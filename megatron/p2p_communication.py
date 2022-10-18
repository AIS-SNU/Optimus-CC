# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ast import comprehension
from bz2 import compress
from functools import reduce
from random import random
import numpy as np
import operator
import torch
import torch.distributed as dist

from megatron import get_args
from megatron import mpu

# For support fp16
from torch.cuda.amp import autocast


# PowerSGD SVD for P2P Communication
class PowerSVD:
    def __init__(self, random_seed, device, reuse_query=True,\
                 acti_rank=12, grad_rank=12, use_error_feedback=True, fp16=False):
        self.rng = np.random.RandomState(random_seed)
        M = 1024 * 1024
        self.precalc_numbers = (
            torch.from_numpy(self.rng.randn(128 * M)).to(device).type(torch.float32)
        )
        if torch.distributed.is_available():
            self.n_workers = 1
            self.rank = torch.distributed.get_rank()
        self.device = device

        """ Gradient Compression """
        # compression_rank
        self.rank = grad_rank
        # matrix P and Q
        self.p_memory = None
        self.p_shape = None
        self.q_memory = None
        self.q_shape = None
        self.memories = None
        """ Activation Compression"""
        # compression_rank
        self.acti_rank = acti_rank
        # matrix P and Q
        self.acti_p_memory = None
        self.acti_p_shape = None
        self.acti_q_memory = None
        self.acti_q_shape = None        
        self.acti_memories = None

        # reuse_query => warm-start (in PowerSGD paper)
        # in most cases it is essential to be True.
        self.reuse_query = reuse_query
        # EF SGD enabled?
        self.use_error_feedback = use_error_feedback
        # support fp16?
        self.fp16 = fp16

        if dist.get_rank() == 0:
            self._init_printer()

    def _init_printer(self):
        print('===== MP PowerSGD Reducer =====')
        print(' >> rank: ', self.rank)
        print(' >> warm_start: ', self.reuse_query)
        print(' >> EF on: ', self.use_error_feedback)
        print('===============================')

    def _set_random(self, vector):
        torch.manual_seed(self.rng.randint(1_000_000_000))
        vector.data[:] = torch.randn(*vector.shape, device=self.device)
        # orthogonalize needs to be done
        # But almost not needed... randn make almost perfect
        orthogonalize(vector)

    # EF memeory is merged into this class
    # return P and Q matrix for communication
    def p2p_grad_compress(self, tensor_in):
        """
        [MP PowerSGD]
        : Grad_in tensor is already 2dim (no multi layer type!)
        """
        # We'll use error feedback but uninitialized
        if self.use_error_feedback and self.memories == None:
            self.memories = torch.zeros_like(tensor_in)
            # if dist.get_rank() == 0:
            #     print(' >> EF Memory initialized')
            #     print(' >> Dtype: ', self.memories.dtype, ' / # elements: ', self.memories.nelement())

        # add EF
        if self.use_error_feedback:
            # if dist.get_rank() == 0 and self.current_iter == 0:
            #     print(' >> EF update into input buffer')
            tensor_in += self.memories
            # if dist.get_rank() == 0 and self.current_iter == 0:
            #     print(' >> EF updated total ', self.memories.nelement(), ' elements')

        # build rank-k approx of every tensor
        # Approx equation
        # M = PQ^T
        # allocate consequtive mem for P's and Q's

        mem_uninitialized = self.p_memory is None
        
        args = get_args()

        n = args.seq_length * args.micro_batch_size
        m = tensor_in.nelement() // n
        # n, m = args.seq_length * args.micro_batch_size, args.hidden_size
        assert args.inter_acti_comp == False, 'We do not support inter activation compression'
        rank = min(n, m, args.inter_grad_comp_rank)
        p_shape = (n, rank)
        q_shape = (m, rank)

        if args.scatter_gather_tensors_in_pipeline:
            tensor_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)
            tensor_chunk_shape = reduce(operator.mul, tensor_shape, 1)
            if tensor_chunk_shape % mpu.get_tensor_model_parallel_world_size() == 0:            
                tensor_chunk_shape = tensor_chunk_shape // \
                                        mpu.get_tensor_model_parallel_world_size()
                assert args.micro_batch_size % 8 == 0, 'our compressed inter communication only support microbatchsize = 8n'
                n = args.seq_length * (args.micro_batch_size // 8)
                m = tensor_chunk_shape // n
                rank = min(n, m, args.inter_grad_comp_rank)
                p_shape=(n, rank)
                q_shape=(m, rank)

        # now we found n,m .... viewing it in 2d!
        tensor_in = tensor_in.view(n, m)

        # Step 1. Calc Matrix Size and Allocate Memory
        p_total_size = 0
        q_total_size = 0
        # matrix = tensor_in.view(tensor_in[0], -1)
        # n, m = matrix.shape # already 2d matrix format
        # rank = min(n, m, self.rank)
        p_total_size += n*rank
        q_total_size += m*rank

        # [Important] Initialization on Device !!!
        if self.p_memory == None: # not initialized
            self.p_memory = torch.empty(p_total_size, device=self.device, dtype=torch.float)
            self.q_memory = torch.empty(q_total_size, device=self.device, dtype=torch.float)

        # for easier implementation, gather pointers
        p_ptr = self.p_memory.view(n, rank)
        q_ptr = self.q_memory.view(m, rank)


        # Step 2. Prepare Q if not initailized
        if self.reuse_query and not mem_uninitialized:
            # if u wanna reuse and already init
            # use prev_Q
            # do not need orthogonalize if properly _set_random...ed!
            orthogonalize(q_ptr)
        else:
            self._set_random(q_ptr)
        
        """
        PowerSGD
        Algorithm 1: Rank-r PowerSGD Compression
        
        All Compression/Decompression is done in Reducer
        """

        # Step 3. (Algo 1: line 3) P <- MQ (Compute P)
        if self.fp16:
            torch.matmul(tensor_in.float(), q_ptr, out=p_ptr)
        else:
            torch.matmul(tensor_in, q_ptr, out=p_ptr)
        
        # We do not need all_reduce for P2POp
        # # Step 4. (Algo 1: line 4) ALL_REDUCE_MEAN(P)
        # all_reduce(self.p_memory, group=self.group)

        # if self.current_iter % 1000 == 0 and dist.get_rank() == 0:
        #     print(' > Compressed P Matrix: ', n_bits(self.p_memory), 'bits')

        # Step 5. (Algo 1: line 5) P_hat <- ORTHOGONALIZE(P)
        orthogonalize(p_ptr)

        # Step 6. (Algo 1: line 6) Q <- M_T P_hat
        if self.fp16:
            torch.matmul(tensor_in.t().float(), p_ptr, out=q_ptr)
        else:
            torch.matmul(tensor_in.t(), p_ptr, out=q_ptr)
        
        # We do not need all_reduce for P2POp
        # Step 7. (Algo 1: line 7) ALL_REDUCE_MEAN(Q)
        # handle = all_reduce(self.q_memory, group=self.group)

        """
        PowerSGD
        Algorithm 2: Distributed Error-feedback SGD with Momentum
        Only Local Error is return by Reducer!
        Main Algorithm is implemented in Main Process
        """
        # Step 8. (Algo 1: line 11) Decompress
        # make temp grad space
        tensor_out = torch.zeros_like(tensor_in)
        if self.fp16:
            with autocast():
                tensor_out.data[:] = torch.mm(p_ptr, q_ptr.t())
                # torch.matmul(p, q.t(), out=out.data[:])
        else:
            torch.matmul(p_ptr, q_ptr.t(), out=tensor_out.data[:])

        # Step 9. (Algo 2: line 9) Memorize Local Errors
        if self.use_error_feedback:
            self.memories.data[:] = tensor_in.view(-1) - tensor_out.view(-1)

        # remove temp tensor space
        del tensor_out

        self.p_shape = p_ptr.shape
        self.q_shape = q_ptr.shape

        return p_ptr, q_ptr
    # it's just helper method for decompress....
    def p2p_grad_recv_decompress(self, p_ptr, q_ptr, out):
        return torch.matmul(p_ptr, q_ptr.t(), out=out.data[:])

    def get_current_grad_error(self):
        return self.memories

    # EF memeory is merged into this class
    # return P and Q matrix for communication
    def p2p_acti_compress(self, tensor_in):
        tensor_in = tensor_in.detach()
        """
        [MP PowerSGD]
        : Grad_in tensor is already 2dim (no multi layer type!)
        """
        # We'll use error feedback but uninitialized
        if self.use_error_feedback and self.acti_memories == None:
            self.acti_memories = torch.zeros_like(tensor_in)
            # if dist.get_rank() == 0:
            #     print(' >> EF Memory initialized')
            #     print(' >> Dtype: ', self.memories.dtype, ' / # elements: ', self.memories.nelement())

        # add EF
        if self.use_error_feedback:
            # if dist.get_rank() == 0 and self.current_iter == 0:
            #     print(' >> EF update into input buffer')
            tensor_in += self.acti_memories
            # if dist.get_rank() == 0 and self.current_iter == 0:
            #     print(' >> EF updated total ', self.memories.nelement(), ' elements')

        # build rank-k approx of every tensor
        # Approx equation
        # M = PQ^T
        # allocate consequtive mem for P's and Q's

        mem_uninitialized = self.acti_p_memory is None

        args = get_args()
        n = args.seq_length * args.micro_batch_size
        m = tensor_in.nelement() // n
        # n, m = args.seq_length * args.micro_batch_size, args.hidden_size
        tensor_in = tensor_in.view(n, m)
        rank = min(n, m, args.inter_acti_comp_rank)
        p_shape = (n, rank)
        q_shape = (m, rank)

        # Step 1. Calc Matrix Size and Allocate Memory
        p_total_size = 0
        q_total_size = 0
        # matrix = tensor_in.view(tensor_in[0], -1)
        # n, m = matrix.shape # already 2d matrix format
        # rank = min(n, m, self.rank)
        p_total_size += n*rank
        q_total_size += m*rank

        # [Important] Initialization on Device !!!
        if self.acti_p_memory == None: # not initialized
            self.acti_p_memory = torch.empty(p_total_size, device=self.device, dtype=torch.float)
            self.acti_q_memory = torch.empty(q_total_size, device=self.device, dtype=torch.float)

        # for easier implementation, gather pointers
        p_ptr = self.acti_p_memory.view(n, rank)
        q_ptr = self.acti_q_memory.view(m, rank)


        # Step 2. Prepare Q if not initailized
        if self.reuse_query and not mem_uninitialized:
            # if u wanna reuse and already init
            # use prev_Q
            # do not need orthogonalize if properly _set_random...ed!
            orthogonalize(q_ptr)
        else:
            self._set_random(q_ptr)
        
        """
        PowerSGD
        Algorithm 1: Rank-r PowerSGD Compression
        
        All Compression/Decompression is done in Reducer
        """

        # Step 3. (Algo 1: line 3) P <- MQ (Compute P)
        if self.fp16:
            torch.matmul(tensor_in.float(), q_ptr, out=p_ptr)
        else:
            torch.matmul(tensor_in, q_ptr, out=p_ptr)
        
        # We do not need all_reduce for P2POp
        # # Step 4. (Algo 1: line 4) ALL_REDUCE_MEAN(P)
        # all_reduce(self.p_memory, group=self.group)

        # if self.current_iter % 1000 == 0 and dist.get_rank() == 0:
        #     print(' > Compressed P Matrix: ', n_bits(self.p_memory), 'bits')

        # Step 5. (Algo 1: line 5) P_hat <- ORTHOGONALIZE(P)
        orthogonalize(p_ptr)

        # Step 6. (Algo 1: line 6) Q <- M_T P_hat
        if self.fp16:
            torch.matmul(tensor_in.t().float(), p_ptr, out=q_ptr)
        else:
            torch.matmul(tensor_in.t(), p_ptr, out=q_ptr)
        
        # We do not need all_reduce for P2POp
        # Step 7. (Algo 1: line 7) ALL_REDUCE_MEAN(Q)
        # handle = all_reduce(self.q_memory, group=self.group)

        """
        PowerSGD
        Algorithm 2: Distributed Error-feedback SGD with Momentum
        Only Local Error is return by Reducer!
        Main Algorithm is implemented in Main Process
        """
        # Step 8. (Algo 1: line 11) Decompress
        # make temp grad space
        tensor_out = torch.zeros_like(tensor_in)
        if self.fp16:
            with autocast():
                tensor_out.data[:] = torch.mm(p_ptr, q_ptr.t())
                # torch.matmul(p, q.t(), out=out.data[:])
        else:
            torch.matmul(p_ptr, q_ptr.t(), out=tensor_out.data[:])

        # Step 9. (Algo 2: line 9) Memorize Local Errors
        if self.use_error_feedback:
            self.acti_memories.data[:] = tensor_in.view(-1) - tensor_out.view(-1)

        # remove temp tensor space
        del tensor_out

        self.acti_p_shape = p_ptr.shape
        self.acti_q_shape = q_ptr.shape

        return p_ptr, q_ptr
    # it's just helper method for decompress....
    def p2p_acti_recv_decompress(self, p_ptr, q_ptr, out):
        return torch.matmul(p_ptr, q_ptr.t(), out=out.data[:])

def _communicate(tensor_send_next, tensor_send_prev, recv_prev, recv_next,
                 tensor_shape,
                 use_ring_exchange=False,
                 dtype_=None,
                 compressor=None,
                 flush=False):
    """
    Communicate tensors between stages. Used as helper method in other
    communication methods that are used in megatron/schedules.py.

    Takes the following arguments:
        tensor_send_next: tensor to send to next rank (no tensor sent if
                          set to None).
        tensor_send_prev: tensor to send to prev rank (no tensor sent if
                          set to None).
        recv_prev: boolean for whether tensor should be received from
                   previous rank.
        recv_next: boolean for whether tensor should be received from
                   next rank.
        tensor_shape: shape of tensor to receive (this method assumes that all
                      tensors sent and received in a single function call are
                      the same shape).
        use_ring_exchange: boolean for whether torch.distributed.ring_exchange()
                           API should be used.
        dtype_: optional, this is used when the tensor that needs to be
                communicated is different from args.params_dtype.
        compressor: optional, this is used when we need to compress inter communication
        flush: optional, this tells you whether this call is for flush stage.
    Returns:
        (tensor_recv_prev, tensor_recv_next)
    """
    args = get_args()


    """
    [WARNING] Watch out this portion
    """
    need_inter_grad_comp = args.inter_grad_comp # we want to inter_grad_comp
    # we want to inter_grad_comp and comp for flush only
    # for not flush only case... do inter_grad_comp for all
    if need_inter_grad_comp and args.inter_grad_comp_epilogue_only:
        if flush: # if current inter comm is flush => compress
            need_inter_grad_comp = True
        else: # if current inter comm is not flush => do no compress
            need_inter_grad_comp = False

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None

    # Some legacy inference code doesn't set the tensor shape, do so now
    # for the normal values for gpt/bert. This could be removed if inference
    # code is changed to provide tensor_shape.
    if tensor_shape is None:
        tensor_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)
    
    # matrix = tensor_shape.view(tensor_shape[0], -1) # already 2d matrix format
    n, m = args.seq_length * args.micro_batch_size, args.hidden_size
    """
    Warning... Use only InterGrad Comp!!!
    """
    assert args.inter_acti_comp == False, 'We do not support inter activation compression'
    rank = min(n, m, args.inter_grad_comp_rank)
    p_shape = (n, rank)
    q_shape = (m, rank)

    override_scatter_gather_tensors_in_pipeline = False
    if args.scatter_gather_tensors_in_pipeline:
        tensor_chunk_shape = reduce(operator.mul, tensor_shape, 1)
        if tensor_chunk_shape % mpu.get_tensor_model_parallel_world_size() == 0:
            tensor_chunk_shape = tensor_chunk_shape // \
                mpu.get_tensor_model_parallel_world_size()
            assert args.micro_batch_size % 8 == 0, 'our compressed inter communication only support microbatchsize = 8n'
            n = args.seq_length * (args.micro_batch_size // 8)
            m = tensor_chunk_shape // n
            rank = min(n, m, args.inter_grad_comp_rank)
            p_shape=(n, rank)
            q_shape=(m, rank)
        else:
            tensor_chunk_shape = tensor_shape
            override_scatter_gather_tensors_in_pipeline = True
    else:
        tensor_chunk_shape = tensor_shape
    dtype = args.params_dtype
    if args.fp32_residual_connection:
        dtype = torch.float

    requires_grad = True
    if args.inter_acti_comp or need_inter_grad_comp:
        requires_grad = False
    if dtype_ is not None:
        dtype = dtype_
        requires_grad = False

    if recv_prev:
        tensor_recv_prev = torch.empty(tensor_chunk_shape,
                                       requires_grad=requires_grad,
                                       device=torch.cuda.current_device(),
                                       dtype=dtype)
        if args.inter_acti_comp:
            # activation recv
            tensor_p_prev = torch.empty(p_shape,
                                        requires_grad=requires_grad,
                                        device=torch.cuda.current_device(),
                                        dtype=dtype)
            tensor_q_prev = torch.empty(q_shape,
                                        requires_grad=requires_grad,
                                        device=torch.cuda.current_device(),
                                        dtype=dtype)
    if recv_next:
        tensor_recv_next = torch.empty(tensor_chunk_shape,
                                       requires_grad=requires_grad,
                                       device=torch.cuda.current_device(),
                                       dtype=dtype)
        if need_inter_grad_comp:
            # grad recv
            tensor_p_next = torch.empty(p_shape,
                                        requires_grad=requires_grad,
                                        device=torch.cuda.current_device(),
                                        dtype=dtype)
            tensor_q_next = torch.empty(q_shape,
                                        requires_grad=requires_grad,
                                        device=torch.cuda.current_device(),
                                        dtype=dtype)

    # Split tensor into smaller chunks if using scatter-gather optimization.
    if not override_scatter_gather_tensors_in_pipeline and \
            args.scatter_gather_tensors_in_pipeline:
        if tensor_send_next is not None:
            tensor_send_next = mpu.split_tensor_into_1d_equal_chunks(tensor_send_next)

        if tensor_send_prev is not None:
            tensor_send_prev = mpu.split_tensor_into_1d_equal_chunks(tensor_send_prev)

    # Send tensors in both the forward and backward directions as appropriate.
    if use_ring_exchange:
        torch.distributed.ring_exchange(tensor_send_prev=tensor_send_prev,
                                        tensor_recv_prev=tensor_recv_prev,
                                        tensor_send_next=tensor_send_next,
                                        tensor_recv_next=tensor_recv_next,
                                        group=mpu.get_pipeline_model_parallel_group())
        # To protect against race condition when using batch_isend_irecv().
        torch.cuda.synchronize()
    else:
        ops = []
        if tensor_send_prev is not None:
            # we compress gradient backward
            if need_inter_grad_comp:
                p, q = compressor.p2p_grad_compress(tensor_send_prev)
                send_prev_op_p = torch.distributed.P2POp(
                    torch.distributed.isend, p,
                    mpu.get_pipeline_model_parallel_prev_rank())
                send_prev_op_q = torch.distributed.P2POp(
                    torch.distributed.isend, q,
                    mpu.get_pipeline_model_parallel_prev_rank())
                ops.append(send_prev_op_p)
                ops.append(send_prev_op_q)
            else:
                send_prev_op = torch.distributed.P2POp(
                    torch.distributed.isend, tensor_send_prev,
                    mpu.get_pipeline_model_parallel_prev_rank())
                ops.append(send_prev_op)
        if tensor_recv_prev is not None:
            # activation recv
            if args.inter_acti_comp:
                recv_prev_op_p = torch.distributed.P2POp(
                    torch.distributed.irecv, tensor_p_prev,
                    mpu.get_pipeline_model_parallel_prev_rank())
                recv_prev_op_q = torch.distributed.P2POp(
                    torch.distributed.irecv, tensor_q_prev,
                    mpu.get_pipeline_model_parallel_prev_rank())
                ops.append(recv_prev_op_p)
                ops.append(recv_prev_op_q)
                # we need to fill tensor_recv_prev by p2p_acti_recv_decompress
            else:
                recv_prev_op = torch.distributed.P2POp(
                    torch.distributed.irecv, tensor_recv_prev,
                    mpu.get_pipeline_model_parallel_prev_rank())
                ops.append(recv_prev_op)
        if tensor_send_next is not None:
            # we compress activation forward
            if args.inter_acti_comp:
                p, q = compressor.p2p_acti_compress(tensor_send_next)
                send_next_op_p = torch.distributed.P2POp(
                    torch.distributed.isend, p,
                    mpu.get_pipeline_model_parallel_next_rank())
                send_next_op_q = torch.distributed.P2POp(
                    torch.distributed.isend, q,
                    mpu.get_pipeline_model_parallel_next_rank())
                ops.append(send_next_op_p)
                ops.append(send_next_op_q)
            else:
                send_next_op = torch.distributed.P2POp(
                    torch.distributed.isend, tensor_send_next,
                    mpu.get_pipeline_model_parallel_next_rank())
                ops.append(send_next_op)
        if tensor_recv_next is not None:
            # gradient recv
            if need_inter_grad_comp:
                # """[debugging]"""
                # print('>>>>>    wainting to recv grad... pp rank: ', str(mpu.get_pipeline_model_parallel_rank()))
                recv_next_op_p = torch.distributed.P2POp(
                    torch.distributed.irecv, tensor_p_next,
                    mpu.get_pipeline_model_parallel_next_rank())
                recv_next_op_q = torch.distributed.P2POp(
                    torch.distributed.irecv, tensor_q_next,
                    mpu.get_pipeline_model_parallel_next_rank())
                ops.append(recv_next_op_p)
                ops.append(recv_next_op_q)
            else:
                recv_next_op = torch.distributed.P2POp(
                    torch.distributed.irecv, tensor_recv_next,
                    mpu.get_pipeline_model_parallel_next_rank())
                ops.append(recv_next_op)
        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
        # To protect against race condition when using batch_isend_irecv().
        torch.cuda.synchronize()
        if args.inter_acti_comp:
            if tensor_recv_prev is not None:
                compressor.p2p_acti_recv_decompress(tensor_p_prev, tensor_q_prev, tensor_recv_prev)
                # print(tensor_recv_prev)
        if need_inter_grad_comp:
            if tensor_recv_next is not None:
                compressor.p2p_grad_recv_decompress(tensor_p_next, tensor_q_next, tensor_recv_next)
                # print('decomp recv: ', tensor_recv_next)

    # If using scatter-gather optimization, gather smaller chunks.
    if not override_scatter_gather_tensors_in_pipeline and \
            args.scatter_gather_tensors_in_pipeline:
        if recv_prev:
            tensor_recv_prev = mpu.gather_split_1d_tensor(
                tensor_recv_prev).view(tensor_shape).requires_grad_()

        if recv_next:
            tensor_recv_next = mpu.gather_split_1d_tensor(
                tensor_recv_next).view(tensor_shape).requires_grad_()

    return tensor_recv_prev, tensor_recv_next


def recv_forward(tensor_shape=None, dtype_=None, timers=None, compressor=None):
    """Receive tensor from previous rank in pipeline (forward receive)."""

    if mpu.is_pipeline_first_stage():
        input_tensor = None
    else:
        """
        [TODO] Inter Device Foward Communication (Activation :))
        [TODO] This forward-recv takes most of the time ...
        [TODO] Mainly compress here!!!
        """
        if timers is not None:
            timers('forward-recv').start()
        input_tensor, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            dtype_=dtype_,
            compressor=compressor)
        if timers is not None:
            timers('forward-recv').stop()
    return input_tensor


def recv_backward(tensor_shape=None, timers=None, compressor=None, flush=False):
    """Receive tensor from next rank in pipeline (backward receive)."""
    if mpu.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if timers is not None:
            timers('backward-recv').start()
        _, output_tensor_grad = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape,
            compressor=compressor,
            flush=flush)
        if timers is not None:
            timers('backward-recv').stop()
    return output_tensor_grad

"""
[TODO] To mitigate recv_foward... maybe we need to fix this method
"""
def send_forward(output_tensor, tensor_shape=None, dtype_=None, timers=None, compressor=None):
    """Send tensor to next rank in pipeline (forward send)."""

    if not mpu.is_pipeline_last_stage():
        if timers is not None:
            timers('forward-send').start()
        _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            tensor_shape=tensor_shape,
            dtype_=dtype_,
            compressor=compressor)
        if timers is not None:
            timers('forward-send').stop()


def send_backward(input_tensor_grad, tensor_shape=None, timers=None, compressor=None, flush=False):
    """Send tensor to previous rank in pipeline (backward send)."""
    if not mpu.is_pipeline_first_stage():
        if timers is not None:
            timers('backward-send').start()
        _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=False,
            tensor_shape=tensor_shape,
            compressor=compressor,
            flush=flush)
        if timers is not None:
            timers('backward-send').stop()


def send_forward_recv_backward(output_tensor, tensor_shape=None, timers=None, compressor=None):
    """Batched send and recv with next rank in pipeline."""
    if mpu.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if timers is not None:
            timers('forward-send-backward-recv').start()
        _, output_tensor_grad = _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape,
            compressor=compressor)
        if timers is not None:
            timers('forward-send-backward-recv').stop()
    return output_tensor_grad


def send_backward_recv_forward(input_tensor_grad, tensor_shape=None, timers=None, compressor=None):
    """Batched send and recv with previous rank in pipeline."""
    if mpu.is_pipeline_first_stage():
        input_tensor = None
    else:
        if timers is not None:
            timers('backward-send-forward-recv').start()
        input_tensor, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            compressor=compressor)
        if timers is not None:
            timers('backward-send-forward-recv').stop()
    return input_tensor


def send_forward_recv_forward(output_tensor, recv_prev, tensor_shape=None, timers=None, compressor=None):
    """Batched recv from previous rank and send to next rank in pipeline."""
    if timers is not None:
        timers('forward-send-forward-recv').start()
    input_tensor, _ = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=recv_prev,
        recv_next=False,
        tensor_shape=tensor_shape,
        compressor=compressor)
    if timers is not None:
        timers('forward-send-forward-recv').stop()
    return input_tensor


def send_backward_recv_backward(input_tensor_grad, recv_next, tensor_shape=None, timers=None, compressor=None, flush=False):
    """Batched recv from next rank and send to previous rank in pipeline."""
    if timers is not None:
        timers('backward-send-backward-recv').start()
    _, output_tensor_grad = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=False,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        compressor=compressor,
        flush=flush)
    if timers is not None:
        timers('backward-send-backward-recv').stop()
    return output_tensor_grad


def send_forward_backward_recv_forward_backward(
        output_tensor, input_tensor_grad, recv_prev,
        recv_next, tensor_shape=None, timers=None, compressor=None):
    """Batched send and recv with previous and next ranks in pipeline."""
    if timers is not None:
        timers('forward-backward-send-forward-backward-recv').start()
    input_tensor, output_tensor_grad = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        compressor=compressor)
    if timers is not None:
        timers('forward-backward-send-forward-backward-recv').stop()
    return input_tensor, output_tensor_grad


@torch.jit.script
def orthogonalize(matrix, eps=torch.tensor(1e-8)):
    n, m = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i : i + 1]
        col /= torch.sqrt(torch.sum(col ** 2)) + eps
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1 :]
            # rest -= torch.matmul(col.t(), rest) * col
            rest -= torch.sum(col * rest, dim=0) * col

@torch.jit.script
def add_error_feedback(t1, t2):
    torch.add(t1, t2, out=t1)

@torch.jit.script
def update_error_feedback(e, t, o):
    torch.add(t, o, alpha=(-1), out=e)

def all_reduce(*args, **kwargs):
    if torch.distributed.is_available() and torch.distributed.get_world_size() > 1:
        return torch.distributed.all_reduce(*args, **kwargs)

def broadcast(*args, **kwargs):
    if torch.distributed.is_available() and torch.distributed.get_world_size() > 1:
        return torch.distributed.broadcast(*args, **kwargs)

def n_bits(tensor):
    return 8 * tensor.nelement() * tensor.element_size()