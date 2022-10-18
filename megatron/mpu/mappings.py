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

import numpy as np
import torch
import torch.distributed as dist

# For support fp16
from torch.cuda.amp import autocast

from .initialize import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from .utils import split_tensor_along_last_dim


def _reduce(input_):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size()==1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())

    return input_


def _split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_tensor_model_parallel_rank()
    output = input_list[rank].contiguous()

    return output


def _gather(input_):
    """Gather tensors and concatinate along the last dimension."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=get_tensor_model_parallel_group())

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output

def _reduce_powersgd(input_, comp_state):
    """All-reduce the input tensor across model parallel group."""
    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size()==1:
        return input_

    if comp_state['use_comp']:
        input_2d = input_.view(input_.shape[0], -1)
        if comp_state['e'] == None: # if error feedback memeory is not init
            comp_state['e'] = torch.zeros_like(input_2d)
        input_2d += comp_state['e']
        mem_uninitialized = (comp_state['p'] == None)
        # Step 1. Calc Matrix Size and Allocate Memory
        p_total_size = 0
        q_total_size = 0
        n, m = input_2d.shape # already 2d matrix format
        rank = min(n, m, comp_state['comp_rank'])
        p_total_size += n*rank
        q_total_size += m*rank

        # [Important] Initialization on Device !!!
        if mem_uninitialized: # not initialized
            comp_state['p'] = torch.empty(p_total_size, device=input_.device, dtype=torch.float)
            comp_state['q'] = torch.empty(q_total_size, device=input_.device, dtype=torch.float)
        # for easier implementation, gather pointers
        p_ptr = comp_state['p'].view(n, rank)
        q_ptr = comp_state['q'].view(m, rank)

        # Step 2. Prepare Q if not initailized
        if not mem_uninitialized:
            # if u wanna reuse and already init
            # use prev_Q
            # do not need orthogonalize if properly _set_random...ed!
            orthogonalize(q_ptr)
        else:
            _set_random(q_ptr)            

        """
        PowerSGD
        Algorithm 1: Rank-r PowerSGD Compression
        
        All Compression/Decompression is done in Reducer
        """

        # Step 3. (Algo 1: line 3) P <- MQ (Compute P)
        if input_.dtype != torch.float:
            torch.matmul(input_2d.float(), q_ptr, out=p_ptr)
        else:
            torch.matmul(input_2d, q_ptr, out=p_ptr)
        
        # Step 4. (Algo 1: line 4) ALL_REDUCE_MEAN(P)
        torch.distributed.all_reduce(
                comp_state['p'], group=get_tensor_model_parallel_group())

        # it's different from original PowerSGD code...
        # if there's another degradation in accurcy
        # uncomment this line for accuracy regain
        # self.p_memory.data[:] /= self.n_workers

        # Step 5. (Algo 1: line 5) P_hat <- ORTHOGONALIZE(P)
        orthogonalize(p_ptr)

        # Step 6. (Algo 1: line 6) Q <- M_T P_hat
        if input_.dtype != torch.float:
            torch.matmul(input_2d.t().float(), p_ptr, out=q_ptr)
        else:
            torch.matmul(input_2d.t(), p_ptr, out=q_ptr)
        
        # Step 7. (Algo 1: line 7) ALL_REDUCE_MEAN(Q)
        torch.distributed.all_reduce(
                comp_state['q'], group=get_tensor_model_parallel_group())

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
        acti_out = torch.zeros_like(input_2d)
        if input_.dtype != torch.float:
            with autocast():
                acti_out.data[:] = torch.mm(p_ptr, q_ptr.t())
                # torch.matmul(p, q.t(), out=out.data[:])
        else:
            torch.matmul(p_ptr, q_ptr.t(), out=acti_out.data[:])
        out_n_bits += n_bits(acti_out.data[:])

        # Step 9. (Algo 2: line 9) Memorize Local Errors
        comp_state['e'].data[:] = input_2d - acti_out

        # copy to grad_in
        input_2d.data.copy_(acti_out)        
    else:
        # All-reduce.
        torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())

    return input_

class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_
    
    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class _ReduceFromModelParallelRegionWithPowerSGD(torch.autograd.Function):
    """All-reduce the input from the model parallel region. (with powersgd)"""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)
    
    @staticmethod
    def forward(ctx, input_, comp_state):
        return _reduce_powersgd(input_, comp_state)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output)


# -----------------
# Helper functions.
# -----------------

def copy_to_tensor_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)

def reduce_from_tensor_model_parallel_region_with_powersgd(input_, comp_state):
    return _ReduceFromModelParallelRegionWithPowerSGD.apply(input_, comp_state)


def scatter_to_tensor_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_tensor_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)


def _set_random(vector):
    rng = np.random.RandomState(714)
    torch.manual_seed(rng.randint(1_000_000_000))
    vector.data[:] = torch.randn(*vector.shape, device=vector.device)
    # orthogonalize needs to be done
    # But almost not needed... randn make almost perfect
    orthogonalize(vector)

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

def n_bits(tensor):
    return 8 * tensor.nelement() * tensor.element_size()