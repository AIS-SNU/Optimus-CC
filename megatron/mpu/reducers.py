from asyncio.proactor_events import _ProactorSocketTransport
import numpy as np
import torch
import torch.distributed as dist
import os
import time
from typing import List

from megatron import get_args
import megatron.mpu.tensor_buffer as tb

# For support fp16
from torch.cuda.amp import autocast

BITS_PER_BYTE = 8

def current_time_in_ms():
    return int(round(time.time() * 1000))


class MemoryBuffer:

    def __init__(self, numel, dtype):
        self.numel = numel
        self.dtype = dtype
        self.data = torch.zeros(self.numel,
                                dtype=self.dtype,
                                device=torch.cuda.current_device(),
                                requires_grad=False)


    def zero(self):
        """Reset the buffer to zero."""
        self.data.zero_()


    def get(self, shape, start_index):
        """Return a tensor with the input `shape` as a view into the
        1-D data starting at `start_index`."""
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, \
            'requested tensor is out of the buffer range.'
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor
    
    def get_flat_to_end(self, start_index):
        """Return a flat tensor starting at `start_index`."""
        end_index = self.numel
        assert start_index < self.numel, \
            'requested tensor is out of the buffer range.'
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(-1)
        return buffer_tensor    

class Reducer:
    def __init__(self, random_seed, device, group, group_num):
        self.rng = np.random.RandomState(random_seed)
        M = 1024 * 1024
        self.precalc_numbers = (
            torch.from_numpy(self.rng.randn(128 * M)).to(device).type(torch.float32)
        )
        if torch.distributed.is_available():
            self.n_workers = group_num
            self.rank = torch.distributed.get_rank()
        else:
            self.n_workers = 1
            self.rank = 0
        self.device = device
        self.group = group

    def reduce(self, grad_in, grad_out, memory_out):
        """Return communicated bits"""
        raise NotImplementedError()

"""
[FP32 Version only...]
[TODO] Compare this Randomized SVD with PowerSGD Reducer
[TODO] Given codes are not using memory buffer! => change it to use that
"""
class RandomizedSVDReducer(Reducer):
    def __init__(self, random_seed, device, group, group_num, rank=8, m=32, start_iter=10,\
                use_error_feedback=False, fp16=False):
        super().__init__(random_seed, device, group, group_num)
        # compression rank
        self.rank = rank
        # matrix m
        self.m = m
        # matrix n
        self.n = None # not initialized
        # warm-up period for 10% training iteration
        self.start_iter = int(start_iter)
        # track current iteration
        self.current_iter = 0
        # EF SGD enabled?
        self.use_error_feedback = use_error_feedback
        # support fp16?
        self.fp16 = fp16
        # EF memory
        self.memories = None
        if dist.get_rank() == 0:
            self._init_printer()

    def _init_printer(self):
        print('===== Randomized SVD Reducer =====')
        print(' >> rank: ', self.rank)
        print(' >> m: ', self.m)
        print(' >> start_iter: ', self.start_iter)
        print(' >> EF on: ', self.use_error_feedback)
        print(' >> FP16: ', self.fp16)
        print('============================')

    # [TODO] should be changed into MemoryBuffer format
    # arguments: parameters(module.parameters()) / grad_in(grad buffers) / grad_out(reducer buffer) / memory_out(EF memory)
    # EF memeory is merged into this class
    def reduce(self, module, grad_in_buffers, grad_out_buffers):

        if self.current_iter < self.start_iter:
            for _, buffer_ in grad_in_buffers.items():
                buffer_.data /= self.n_workers
                all_reduce(buffer_.data, group=self.group)
            if self.current_iter % 100 == 0 and dist.get_rank() == 0:
                print(' >> Still in Warm-up Period... ', self.current_iter)
            self.current_iter += 1
        else:
            # Simple function to define buffer type.
            def _get_buffer_type(param):
                return param.dtype

            # Collect Views of all tensors for each layer
            grad_in = []
            grad_out = []
            memory_out = []

            type_num_elements = {}
            for param in module.parameters():
                if param.requires_grad:
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] = type_num_elements.get(dtype, 0) \
                                                + param.data.nelement()

            # We'll use error feedback but uninitialized
            if self.use_error_feedback and self.memories == None:
                self.memories = {}
                # if dist.get_rank() == 0:
                #     print(' >> EF Memory initialized')
                for dtype, num_elements in type_num_elements.items():
                    self.memories[dtype] = MemoryBuffer(num_elements, dtype)
                    # if dist.get_rank() == 0:
                    #     print(' >> Dtype: ', dtype, ' / # elements: ', num_elements)                                                + param.data.nelement()

            # # add EF error for each 'DataType'
            if self.use_error_feedback:
                # if dist.get_rank() == 0 and self.current_iter == 0:
                #     print(' >> EF update into input buffer')
                # copied_elements = 0
                for (_, buffer_), (_, e_) in zip(grad_in_buffers.items(), self.memories.items()):
                    # buffer_.data.nan_to_num(nan=1e-8)
                    # e_.data.nan_to_num_(nan=1e-4)
                    # print(e_.data[:100])
                    buffer_.data += e_.data
                    # copied_elements += e_.data.nelement()
                # if dist.get_rank() == 0 and self.current_iter == 0:
                #     print(' >> EF updated total ', copied_elements, ' elements')

            # Assume the back prop order is reverse the params order,
            # store the start index for the gradients.
            for param in module.parameters():
                if param.requires_grad:
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] -= param.data.nelement()
                    # So param.main_grad is view of grad_buffers
                    grad_in.append(grad_in_buffers[dtype].get(
                        param.data.shape, type_num_elements[dtype]))
                    grad_out.append(grad_out_buffers[dtype].get(
                        param.data.shape, type_num_elements[dtype]))
                    if self.use_error_feedback:
                        memory_out.append(self.memories[dtype].get(
                            param.data.shape, type_num_elements[dtype]))

            # [For Rank 1] It's out of Algorithms!!!!
            # rank1 tensors will be reduced un-compressed
            # and rank > 1 tensors should be compressed and reduced
            rank1_tensors = [
                (tensor, out, mem)
                for tensor, out, mem in zip(grad_in, grad_out, memory_out)
                if tensor.ndimension() <= 1
            ]

            # Error Handling... There's a case that rank1 tensors are not exist.
            # Most cases of NLP tasks have more dimension than rank1
            if len(rank1_tensors) == 0:
                process_rank1_tensors = False
            else:
                process_rank1_tensors = True

            high_rank_tensors = [
                (tensor, out, mem)
                for tensor, out, mem in zip(grad_in, grad_out, memory_out)
                if tensor.ndimension() > 1
            ]

            # [TODO] Change order of all reduce rank1 tensors
            # [For Rank 1] Start Communicating Rank 1 Tensors
            # Maybe due to there's no rank1 tensors?
            if process_rank1_tensors:
                rank1_tensor_list = tb.TensorBuffer([tensor for (tensor, _, _) in rank1_tensors], group=self.group)
                rank1_handler = rank1_tensor_list.all_reduce(async_op=True)

            for tensor, out, mem in high_rank_tensors:
                # convert grad(M) into 2d tensor
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape
                rank = min(n, m, self.rank)
                u, s, v = torch.svd_lowrank(matrix, q=rank, niter=1, M=None)
                n_data_parallel_workers = self.n_workers
                if n_data_parallel_workers > 1:
                    u /= n_data_parallel_workers
                    v /= n_data_parallel_workers
                    s /= n_data_parallel_workers
                    h1 = torch.distributed.all_reduce(u, group=self.group, async_op=True)
                    h2 = torch.distributed.all_reduce(v, group=self.group, async_op=True)
                    h3 = torch.distributed.all_reduce(s, group=self.group, async_op=True)
                    h1.wait()
                    h2.wait()
                    h3.wait()
                out.data[:] = torch.einsum("in, n, jn -> ij", u, s, v)
                if self.use_error_feedback:
                    mem.data[:] = tensor - out
            
            if process_rank1_tensors:
                rank1_handler.wait()
                rank1_tensor_list.buffer /= self.n_workers
                rank1_tensor_list.unpack([out for (_, out, _) in rank1_tensors])     
        
        if self.current_iter < self.start_iter:
            # track current iteration
            self.current_iter += 1
            return False
        else:
            # track current iteration
            self.current_iter += 1
            return True

"""
[Finished] Change PowerSGD Reducer for MemoryBuffer Type (contiguous)
[TODO] Error-Feedback should be merged into this class (finished - debugging)
[TODO] warm-up period should be implemented into this class (finished - debugging)
"""
class PowerSGDReducer(Reducer):
    def __init__(self, random_seed, device, group, group_num, n_power_iterations=0, reuse_query=True,\
                 rank=4, start_iter=10, use_error_feedback=False, fp16=False):
        super().__init__(random_seed, device, group, group_num)
        # check if power iteration == 0 or not
        assert n_power_iterations == 0
        # compression_rank
        self.rank = rank
        # matrix P and Q
        self.p_memory = None
        self.q_memory = None
        # reuse_query => warm-start (in PowerSGD paper)
        # in most cases it is essential to be True.
        self.reuse_query = reuse_query
        # warm-up period for 10% training iteration (!!! important !!!)
        self.start_iter = int(start_iter)
        # track current iteration
        self.current_iter = 0
        # EF SGD enabled?
        self.use_error_feedback = use_error_feedback
        # support fp16?
        self.fp16 = fp16
        self.memories = None
        if dist.get_rank() == 0:
            self._init_printer()
        self.args = get_args()

    def _init_printer(self):
        print('===== PowerSGD Reducer =====')
        print(' >> rank: ', self.rank)
        print(' >> warm_start: ', self.reuse_query)
        print(' >> start_iter: ', self.start_iter)
        print(' >> EF on: ', self.use_error_feedback)
        print('============================')

    def _set_random(self, vector):
        torch.manual_seed(self.rng.randint(1_000_000_000))
        vector.data[:] = torch.randn(*vector.shape, device=self.device)
        # orthogonalize needs to be done
        # But almost not needed... randn make almost perfect
        orthogonalize(vector)

    # [TODO] should be changed into MemoryBuffer format
    # arguments: parameters(module.parameters()) / grad_in(grad buffers) / grad_out(reducer buffer) / memory_out(EF memory)
    # EF memeory is merged into this class
    def reduce(self, module, grad_in_buffers, grad_out_buffers, grad_start_idx):
        """
        grad_in, grad_out, memory_out : dictionary of params grads
        return total communicated
        """

        if self.current_iter < self.start_iter:
            for _, buffer_ in grad_in_buffers.items():
                if self.args.emb_comm_opt:
                    non_emb_buffer_ = buffer_.get_flat_to_end(grad_start_idx)
                    non_emb_buffer_ /= self.n_workers
                    all_reduce(non_emb_buffer_, group=self.group)
                else:
                    buffer_.data /= self.n_workers
                    all_reduce(buffer_.data, group=self.group)
            if self.current_iter % 100 == 0 and dist.get_rank() == 0:
                print(' >> Still in Warm-up Period... ', self.current_iter)
        else:
            # Simple function to define buffer type.
            def _get_buffer_type(param):
                return param.dtype

            # Collect Views of all tensors for each layer
            grad_in = []
            grad_out = []
            memory_out = []

            type_num_elements = {}
            for param in module.parameters():
                if param.requires_grad:
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] = type_num_elements.get(dtype, 0) \
                                                + param.data.nelement()

            # We'll use error feedback but uninitialized
            # if self.use_error_feedback and self.memories == None:
            if self.memories == None:
                self.memories = {}
                # if dist.get_rank() == 0:
                #     print(' >> EF Memory initialized')
                for dtype, num_elements in type_num_elements.items():
                    self.memories[dtype] = MemoryBuffer(num_elements, dtype)
                    # if dist.get_rank() == 0:
                    #     print(' >> Dtype: ', dtype, ' / # elements: ', num_elements)

            # # add EF error for each 'DataType'
            if self.use_error_feedback:
                # if dist.get_rank() == 0 and self.current_iter == 0:
                #     print(' >> EF update into input buffer')
                # copied_elements = 0
                for (_, buffer_), (_, e_) in zip(grad_in_buffers.items(), self.memories.items()):
                    # buffer_.data.nan_to_num(nan=1e-8)
                    # e_.data.nan_to_num_(nan=1e-4)
                    # print(e_.data[:100])
                    # buffer_.data += e_.data
                    if self.args.emb_comm_opt:
                        add_error_feedback(buffer_.get_flat_to_end(grad_start_idx), \
                                            e_.get_flat_to_end(grad_start_idx))
                    else:
                        add_error_feedback(buffer_.data, e_.data)
                    # copied_elements += e_.data.nelement()
                # if dist.get_rank() == 0 and self.current_iter == 0:
                #     print(' >> EF updated total ', copied_elements, ' elements')


            # if dist.get_rank() == 0 and self.current_iter == 0:
            #     print(' >> Tensor pointer gathering...')

            # Assume the back prop order is reverse the params order,
            # store the start index for the gradients.
            for name, param in module.named_parameters():
                if self.args.emb_comm_opt and 'word_embeddings' in name:
                        # embeddings are in early in memories... so it's ok to do like this
                        continue
                if param.requires_grad:
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] -= param.data.nelement()
                    # So param.main_grad is view of grad_buffers
                    grad_in.append(grad_in_buffers[dtype].get(
                        param.data.shape, type_num_elements[dtype]))
                    grad_out.append(grad_out_buffers[dtype].get(
                        param.data.shape, type_num_elements[dtype]))
                    # if self.use_error_feedback:
                    #     memory_out.append(self.memories[dtype].get(
                    #         param.data.shape, type_num_elements[dtype]))
                    # [debugging] error on non error feedback case
                    memory_out.append(self.memories[dtype].get(
                        param.data.shape, type_num_elements[dtype]))
                

            # [For Rank 1] It's out of Algorithms!!!!
            # rank1 tensors will be reduced un-compressed
            # and rank > 1 tensors should be compressed and reduced
            rank1_tensors = [
                (tensor, out, mem)
                for tensor, out, mem in zip(grad_in, grad_out, memory_out)
                if tensor.ndimension() <= 1
            ]
            # Error Handling... There's a case that rank1 tensors are not exist.
            # Most cases of NLP tasks have more dimension than rank1
            if len(rank1_tensors) == 0:
                process_rank1_tensors = False
            else:
                process_rank1_tensors = True

            high_rank_tensors = [
                (tensor, out, mem)
                for tensor, out, mem in zip(grad_in, grad_out, memory_out)
                if tensor.ndimension() > 1
            ]

            # build rank-k approx of every tensor
            # Approx equation
            # M = PQ^T
            # allocate consequtive mem for P's and Q's

            mem_uninitialized = self.p_memory is None

            # Step 1. Calc Matrix Size and Allocate Memory
            p_total_size = 0
            q_total_size = 0
            for tensor, _, _ in high_rank_tensors:
                # convert grad(M) into 2d tensor
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape
                rank = min(n, m, self.rank)
                p_total_size += n*rank
                q_total_size += m*rank
            # [Important] Initialization on Device !!!
            if self.p_memory == None: # not initialized
                self.p_memory = torch.empty(p_total_size, device=self.device, dtype=torch.float)
                self.q_memory = torch.empty(q_total_size, device=self.device, dtype=torch.float)
            # for easier implementation, gather pointers
            p_ptrs = []
            q_ptrs = []
            p_idx = 0
            q_idx = 0
            for tensor, _, _ in high_rank_tensors:
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape
                rank = min(n, m , self.rank)
                # torch.tensor.view returns pointer
                p_ptrs.append(self.p_memory[p_idx : p_idx + n*rank].view(n, rank))
                q_ptrs.append(self.q_memory[q_idx : q_idx + m*rank].view(m, rank))
                p_idx += n * rank
                q_idx += m * rank

            # Step 2. Prepare Q if not initailized
            for (tensor, _, _), q, p in zip(high_rank_tensors, q_ptrs, p_ptrs):
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape
                if self.reuse_query and not mem_uninitialized:
                    # if u wanna reuse and already init
                    # use prev_Q
                    # do not need orthogonalize if properly _set_random...ed!
                    # megatron-lm need nan to zero
                    #orthogonalize(q)
                    # q.nan_to_num_(nan=1e-4)
                    pass
                else:
                    self._set_random(q)
            
            """
            PowerSGD
            Algorithm 1: Rank-r PowerSGD Compression
            
            All Compression/Decompression is done in Reducer
            """

            # Step 3. (Algo 1: line 3) P <- MQ (Compute P)
            for (tensor, _, _), q, p in zip(high_rank_tensors, q_ptrs, p_ptrs):
                matrix = tensor.view(tensor.shape[0], -1)
                if self.fp16:
                    torch.matmul(matrix.float(), q, out=p)
                else:
                    torch.matmul(matrix, q, out=p)
                # p.nan_to_num_(nan=1e-4)
            
            # if dist.get_rank() == 0 and self.current_iter == 0:
            # print(self.q_memory.data[:100])
            # print(self.p_memory.data[:1000])

            # Step 4. (Algo 1: line 4) ALL_REDUCE_MEAN(P)
            all_reduce(self.p_memory, group=self.group)

            # if self.current_iter % 1000 == 0 and dist.get_rank() == 0:
            #     print(' > Compressed P Matrix: ', n_bits(self.p_memory), 'bits')

            # it's different from original PowerSGD code...
            # if there's another degradation in accurcy
            # uncomment this line for accuracy regain
            # self.p_memory.data[:] /= self.n_workers

            # [For Rank 1] Start Communicating Rank 1 Tensors
            # Maybe due to there's no rank1 tensors?
            if process_rank1_tensors:
                rank1_tensor_list = tb.TensorBuffer([tensor for (tensor, _, _) in rank1_tensors], group=self.group)
                rank1_handler = rank1_tensor_list.all_reduce(async_op=True)

            # Step 5. (Algo 1: line 5) P_hat <- ORTHOGONALIZE(P)
            for p in p_ptrs:
                orthogonalize(p)
                # p.nan_to_num_(nan=1e-4)

            # Step 6. (Algo 1: line 6) Q <- M_T P_hat
            for p, q, (tensor, _, _) in zip(p_ptrs, q_ptrs, high_rank_tensors):
                matrix = tensor.view(tensor.shape[0], -1)
                if self.fp16:
                    torch.matmul(matrix.t().float(), p, out=q)
                else:
                    torch.matmul(matrix.t(), p, out=q)
                # q.nan_to_num_(nan=1e-4)
            
            # Step 7. (Algo 1: line 7) ALL_REDUCE_MEAN(Q)
            all_reduce(self.q_memory, group=self.group)
            # if self.current_iter % 1000 == 0 and dist.get_rank() == 0:
            #     print(' > Compressed Q Matrix: ', n_bits(self.q_memory), 'bits')
            self.q_memory.data /= self.n_workers

            """
            PowerSGD
            Algorithm 2: Distributed Error-feedback SGD with Momentum
            Only Local Error is return by Reducer!
            Main Algorithm is implemented in Main Process
            """
            out_n_bits = 0
            # Step 8. (Algo 1: line 11) Decompress
            for p, q, (tensor, out, mem) in zip(p_ptrs, q_ptrs, high_rank_tensors):
                # einsum representation
                # out.data[:] = torch.einsum("nr, mr -> nm", (p, q)).view(*tensor.shape)
                if self.fp16:
                    with autocast():
                        out.data[:] = torch.mm(p, q.t())
                        # torch.matmul(p, q.t(), out=out.data[:])
                else:
                    torch.matmul(p, q.t(), out=out.data[:])
                out_n_bits += n_bits(out.data[:])
                # Step 9. (Algo 2: line 9) Memorize Local Errors
                if self.use_error_feedback:
                    update_error_feedback(mem.data[:], tensor, out)
                    # mem.data[:] = tensor - out
            # if self.current_iter % 1000 == 0 and dist.get_rank() == 0:
            #     print(' > Original Matrix: ', out_n_bits, 'bits')
            #     if (n_bits(self.p_memory)+n_bits(self.q_memory)) != 0:
            #         print(' > Compression Ratio: ', \
            #                 out_n_bits/(n_bits(self.p_memory)+n_bits(self.q_memory)))

            # [For Rank 1] Wait for Reducing
            if process_rank1_tensors:
                rank1_handler.wait()
                rank1_tensor_list.buffer /= self.n_workers
                rank1_tensor_list.unpack([out for (_, out, _) in rank1_tensors])     
        
        if self.current_iter < self.start_iter:
            # track current iteration
            self.current_iter += 1
            return False
        else:
            # track current iteration
            self.current_iter += 1
            return True


"""
[TODO] Embedding Gradient All-Reduce
[TODO] Should be changed into proper format and arguemt type
"""
class EmbPowerSGDReducer(Reducer):
    # Do we really need group num for embedding powersgd reducer??
    # [TODO] Figure out the logic and remove/keep group_num argument
    # Embedding Reducer do not need group_num !
    # Just sum them
    def __init__(self, random_seed, device, group, group_num, n_power_iterations=0, reuse_query=True,\
                 rank=4, start_iter=10, use_error_feedback=False, fp16=False):
        assert group_num == 1 # group_num must be 1 in Embedding All-Reduce
        super().__init__(random_seed, device, group, group_num)
        # check if power iteration == 0 or not
        assert n_power_iterations == 0
        # compression_rank
        self.rank = rank
        # matrix P and Q
        self.p_memory = None
        self.q_memory = None
        # reuse_query => warm-start (in PowerSGD paper)
        # in most cases it is essential to be True.
        self.reuse_query = reuse_query
        # warm-up period for 10% training iteration (!!! important !!!)
        self.start_iter = int(start_iter)
        # track current iteration
        self.current_iter = 0
        # EF SGD enabled?
        self.use_error_feedback = use_error_feedback
        # support fp16?
        self.fp16 = fp16
        self.memories = None
        if dist.get_rank() == 0:
            self._init_printer()

    def _init_printer(self):
        print('===== Embedding PowerSGD Reducer =====')
        print(' >> rank: ', self.rank)
        print(' >> warm_start: ', self.reuse_query)
        print(' >> start_iter: ', self.start_iter)
        print(' >> EF on: ', self.use_error_feedback)
        print('============================')

    def _set_random(self, vector):
        torch.manual_seed(self.rng.randint(1_000_000_000))
        vector.data[:] = torch.randn(*vector.shape, device=self.device)
        # orthogonalize needs to be done
        # But almost not needed... randn make almost perfect
        orthogonalize(vector)

    # [TODO] should be changed into MemoryBuffer format
    # arguments: parameters(module.parameters()) / grad_in(grad buffers) / grad_out(reducer buffer) / memory_out(EF memory)
    # EF memeory is merged into this class
    def reduce(self, grad_in):
        """
        grad_in: embedding gradient compression only need grad_in
        """
        if self.current_iter < self.start_iter:
            # no averaging for embedding gradient
            all_reduce(grad_in, group=self.group)
            if self.current_iter % 100 == 0 and dist.get_rank() == 0:
                print(' >> Still in Warm-up Period... ', self.current_iter)
        else:
            """
            [Embedding PowerSGD]
            : Grad_in tensor is already 2dim (simple type)
            """
            # We'll use error feedback but unintialized
            if self.use_error_feedback and self.memories == None:
                self.memories = torch.zeros_like(grad_in)
                if dist.get_rank() == 0:
                    print(' >> EF Memory initialized')
                    print(' >> Dtype: ', self.memories.dtype, ' / # elements: ', self.memories.nelement())
            
            # add Error Feedback
            if self.use_error_feedback:
                grad_in += self.memories

            # build rank-k approx of every tensor (in this case 2d grad matrix)
            # approx eq =>
            # M = PQ^T
            # allocate consequtive mem for P's and Q's

            mem_uninitialized = self.p_memory is None

            # Step 1. Calc Matrix Size and Allocate Memory
            p_total_size = 0
            q_total_size = 0
            n, m = grad_in.shape
            rank = min(n, m, self.rank)
            p_total_size += n * rank
            q_total_size += m * rank

            # initialize on device
            if self.p_memory == None:
                self.p_memory = torch.empty(p_total_size, device=self.device, dtype=torch.float)
                self.q_memory = torch.empty(q_total_size, device=self.device, dtype=torch.float)

            # for easy implementation, gather pointers
            p_ptr = self.p_memory.view(n, rank)
            q_ptr = self.q_memory.view(m, rank)

            # Step 2. prepare q if not initialized
            if self.reuse_query and not mem_uninitialized:
                # if u wanna reuse and already init
                # use prev_Q
                # do not need orthogonalize if properly _set_randomed
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
                torch.matmul(grad_in.float(), q_ptr, out=p_ptr)
            else:
                torch.matmul(grad_in, q_ptr, out=p_ptr)
            
            # Step 4. (Algo 1: line 4) ALL_REDUCE_MEAN(P)
            all_reduce(self.p_memory, group=self.group)

            # if self.current_iter % 1000 == 0 and dist.get_rank() == 0:
            #     print(' > Compressed P Matrix: ', n_bits(self.p_memory), 'bits')

            # it's different from original PowerSGD code...
            # if there's another degradation in accurcy
            # uncomment this line for accuracy regain
            # self.p_memory.data[:] /= self.n_workers

            # Step 5. (Algo 1: line 5) P_hat <- ORTHOGONALIZE(P)
            orthogonalize(p_ptr)

            # Step 6. (Algo 1: line 6) Q <- M_T P_hat
            if self.fp16:
                torch.matmul(grad_in.t().float(), p_ptr, out=q_ptr)
            else:
                torch.matmul(grad_in.t(), p_ptr, out=q_ptr)
            
            # Step 7. (Algo 1: line 7) ALL_REDUCE_MEAN(Q)
            all_reduce(self.q_memory, group=self.group)
            # if self.current_iter % 1000 == 0 and dist.get_rank() == 0:
            #     print(' > Compressed Q Matrix: ', n_bits(self.q_memory), 'bits')

            # no need for averaging !
            # self.q_memory.data /= self.n_workers

            """
            PowerSGD
            Algorithm 2: Distributed Error-feedback SGD with Momentum
            Only Local Error is return by Reducer!
            Main Algorithm is implemented in Main Process
            """
            # out_n_bits = 0
            # Step 8. (Algo 1: line 11) Decompress
            # make temp grad space
            grad_out = torch.zeros_like(grad_in)
            if self.fp16:
                with autocast():
                    grad_out.data[:] = torch.mm(p_ptr, q_ptr.t())
                    # torch.matmul(p, q.t(), out=out.data[:])
            else:
                torch.matmul(p_ptr, q_ptr.t(), out=grad_out.data[:])
            # out_n_bits += n_bits(grad_out.data[:])

            # Step 9. (Algo 2: line 9) Memorize Local Errors
            if self.use_error_feedback:
                self.memories.data[:] = grad_in - grad_out

            # copy to grad_in
            grad_in.data[:] = grad_out.data[:]

            # remove temp grad space
            del grad_out
            # torch.cuda.empty_cache()

            # if self.current_iter % 1000 == 0 and dist.get_rank() == 0:
            #     print(' > Original Matrix: ', out_n_bits, 'bits')
            #     if (n_bits(self.p_memory)+n_bits(self.q_memory)) != 0:
            #         print(' > Compression Ratio: ', \
            #                 out_n_bits/(n_bits(self.p_memory)+n_bits(self.q_memory)))
        
        if self.current_iter < self.start_iter:
            # track current iteration
            self.current_iter += 1
        else:
            # track current iteration
            self.current_iter += 1

# PowerSGD Reducer for Model Parallel
class MPPowerSGDReducer(Reducer):
    def __init__(self, random_seed, device, group, group_num, n_power_iterations=0, reuse_query=True,\
                 rank=4, start_iter=10, use_error_feedback=True, fp16=False):
        super().__init__(random_seed, device, group, group_num)
        # check if power iteration == 0 or not
        assert n_power_iterations == 0
        # compression_rank
        self.rank = rank
        # matrix P and Q
        self.p_memory = None
        self.q_memory = None
        # reuse_query => warm-start (in PowerSGD paper)
        # in most cases it is essential to be True.
        self.reuse_query = reuse_query
        # warm-up period for 10% training iteration (!!! important !!!)
        self.start_iter = int(start_iter)
        # track current iteration
        self.current_iter = 0
        # EF SGD enabled?
        self.use_error_feedback = use_error_feedback
        # support fp16?
        self.fp16 = fp16
        self.memories = None
        # if dist.get_rank() == 0:
        #     self._init_printer()

    def _init_printer(self):
        print('===== MP PowerSGD Reducer =====')
        print(' >> rank: ', self.rank)
        print(' >> warm_start: ', self.reuse_query)
        print(' >> start_iter: ', self.start_iter)
        print(' >> EF on: ', self.use_error_feedback)
        print('===============================')

    def _set_random(self, vector):
        torch.manual_seed(self.rng.randint(1_000_000_000))
        vector.data[:] = torch.randn(*vector.shape, device=self.device)
        # orthogonalize needs to be done
        # But almost not needed... randn make almost perfect
        orthogonalize(vector)

    # [TODO] should be changed into MemoryBuffer format
    # arguments: parameters(module.parameters()) / grad_in(grad buffers) / grad_out(reducer buffer) / memory_out(EF memory)
    # EF memeory is merged into this class
    def reduce(self, grad_in):
        """
        grad_in, grad_out, memory_out : dictionary of params grads
        return total communicated
        """

        if self.current_iter < self.start_iter:
            # no averaging for model parallel
            all_reduce(grad_in, group=self.group)
            if self.current_iter % 100 == 0 and dist.get_rank() == 0:
                print(' >> Still in Warm-up Period... ', self.current_iter)
        else:
            """
            [MP PowerSGD]
            : Grad_in tensor is already 2dim (no multi layer type!)
            """
            # We'll use error feedback but uninitialized
            if self.use_error_feedback and self.memories == None:
                self.memories = torch.zeros_like(grad_in)
                if dist.get_rank() == 0:
                    print(' >> EF Memory initialized')
                    print(' >> Dtype: ', self.memories.dtype, ' / # elements: ', self.memories.nelement())

            # add EF
            if self.use_error_feedback:
                if dist.get_rank() == 0 and self.current_iter == 0:
                    print(' >> EF update into input buffer')
                grad_in += self.memories
                if dist.get_rank() == 0 and self.current_iter == 0:
                    print(' >> EF updated total ', self.memories.nelement(), ' elements')        

            # build rank-k approx of every tensor
            # Approx equation
            # M = PQ^T
            # allocate consequtive mem for P's and Q's

            mem_uninitialized = self.p_memory is None

            # Step 1. Calc Matrix Size and Allocate Memory
            p_total_size = 0
            q_total_size = 0
            n, m = grad_in.shape # already 2d matrix format
            rank = min(n, m, self.rank)
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
                torch.matmul(grad_in.float(), q_ptr, out=p_ptr)
            else:
                torch.matmul(grad_in, q_ptr, out=p_ptr)
            
            # Step 4. (Algo 1: line 4) ALL_REDUCE_MEAN(P)
            all_reduce(self.p_memory, group=self.group)

            if self.current_iter % 1000 == 0 and dist.get_rank() == 0:
                print(' > Compressed P Matrix: ', n_bits(self.p_memory), 'bits')

            # it's different from original PowerSGD code...
            # if there's another degradation in accurcy
            # uncomment this line for accuracy regain
            # self.p_memory.data[:] /= self.n_workers

            # Step 5. (Algo 1: line 5) P_hat <- ORTHOGONALIZE(P)
            orthogonalize(p_ptr)

            # Step 6. (Algo 1: line 6) Q <- M_T P_hat
            if self.fp16:
                torch.matmul(grad_in.t().float(), p_ptr, out=q_ptr)
            else:
                torch.matmul(grad_in.t(), p_ptr, out=q_ptr)
            
            # Step 7. (Algo 1: line 7) ALL_REDUCE_MEAN(Q)
            handle = all_reduce(self.q_memory, group=self.group)
            if self.current_iter % 1000 == 0 and dist.get_rank() == 0:
                print(' > Compressed Q Matrix: ', n_bits(self.q_memory), 'bits')

            # no need for averaging !
            # self.q_memory.data /= self.n_workers

            """
            PowerSGD
            Algorithm 2: Distributed Error-feedback SGD with Momentum
            Only Local Error is return by Reducer!
            Main Algorithm is implemented in Main Process
            """
            out_n_bits = 0
            # Step 8. (Algo 1: line 11) Decompress
            # make temp grad space
            grad_out = torch.zeros_like(grad_in)
            if self.fp16:
                with autocast():
                    grad_out.data[:] = torch.mm(p_ptr, q_ptr.t())
                    # torch.matmul(p, q.t(), out=out.data[:])
            else:
                torch.matmul(p_ptr, q_ptr.t(), out=grad_out.data[:])
            out_n_bits += n_bits(grad_out.data[:])

            # Step 9. (Algo 2: line 9) Memorize Local Errors
            if self.use_error_feedback:
                self.memories.data[:] = grad_in - grad_out

            # copy to grad_in
            grad_in.data.copy_(grad_out)

            # remove temp grad space
            del grad_out
            torch.cuda.empty_cache()

            if self.current_iter % 1000 == 0 and dist.get_rank() == 0:
                print(' > Original Matrix: ', out_n_bits, 'bits')
                if (n_bits(self.p_memory)+n_bits(self.q_memory)) != 0:
                    print(' > Compression Ratio: ', \
                            out_n_bits/(n_bits(self.p_memory)+n_bits(self.q_memory)))

        if self.current_iter < self.start_iter:
            # track current iteration
            self.current_iter += 1
            return False
        else:
            # track current iteration
            self.current_iter += 1
            return True


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