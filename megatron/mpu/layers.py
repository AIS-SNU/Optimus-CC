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


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch


import math

import numpy as np

import torch
import torch.nn.functional as F
import torch.nn.init as init
import torch.distributed as dist
from torch.nn.parameter import Parameter

from .initialize import get_tensor_model_parallel_rank
from .initialize import get_tensor_model_parallel_world_size
from .initialize import get_tensor_model_parallel_group
from .mappings import copy_to_tensor_model_parallel_region
from .mappings import gather_from_tensor_model_parallel_region
from .mappings import reduce_from_tensor_model_parallel_region
from .mappings import reduce_from_tensor_model_parallel_region_with_powersgd
from .mappings import scatter_to_tensor_model_parallel_region
from .random import get_cuda_rng_tracker
from .utils import divide
from .utils import split_tensor_along_last_dim
from .utils import VocabUtility
from megatron import get_args

# from .custom_autograd import ColumnParallelLinearWithPowerSGDAsyncAllreduce

from .reducers import MPPowerSGDReducer

# For support fp16
from torch.cuda.amp import autocast


_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {'tensor_model_parallel': False,
                                      'partition_dim': -1,
                                      'partition_stride': 1}


def param_is_not_tensor_parallel_duplicate(param):
    return (hasattr(param, 'tensor_model_parallel') and
            param.tensor_model_parallel) or (
                get_tensor_model_parallel_rank() == 0)


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, 'tensor_model_parallel', is_parallel)
    setattr(tensor, 'partition_dim', dim)
    setattr(tensor, 'partition_stride', stride)


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(destination_tensor, attribute,
                    getattr(source_tensor, attribute))
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)


def _initialize_affine_weight_gpu(weight, init_method,
                                  partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    with get_cuda_rng_tracker().fork():
        init_method(weight)


def _initialize_affine_weight_cpu(weight, output_size, input_size,
                                  per_partition_size, partition_dim,
                                  init_method, stride=1,
                                  return_master_weight=False):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size,
                                dtype=torch.float,
                                requires_grad=False)
    init_method(master_weight)
    args = get_args()
    master_weight = master_weight.to(dtype=args.params_dtype)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size,
                              dim=partition_dim)
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, get_tensor_model_parallel_rank(),
                self.tensor_model_parallel_size)
        self.num_embeddings_per_partition = self.vocab_end_index - \
            self.vocab_start_index

        # Allocate weights and initialize.
        args = get_args()
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                dtype=args.params_dtype))
            _initialize_affine_weight_cpu(
                self.weight, self.num_embeddings, self.embedding_dim,
                self.num_embeddings_per_partition, 0, init_method)
        else:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=1)

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | \
                         (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        return output


class ColumnParallelLinearWithAsyncAllreduce(torch.autograd.Function):
    """
    Column-parallel linear layer execution with asynchronous all-reduce
    execution in backprop.
    """
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        output = torch.matmul(input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        grad_input = grad_output.matmul(weight)
        # Asyncronous all-reduce
        handle = torch.distributed.all_reduce(
                grad_input, group=get_tensor_model_parallel_group(), async_op=True)
        
        # Delay the start of weight gradient computation shortly (3us) to have
        # all-reduce scheduled first and have GPU resources allocated
        _ = torch.empty(1, device=grad_output.device) + 1
        grad_weight = grad_output.t().matmul(input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None
        handle.wait()
        return grad_input, grad_weight, grad_bias

class ColumnParallelLinearWithPowerSGDAsyncAllreduce(torch.autograd.Function):
    """
    Column-parallel linear layer execution with asynchronous all-reduce
    execution in backprop.
    """
    @staticmethod
    def forward(ctx, input, weight, bias, comp_state):
        ctx.save_for_backward(input, weight)
        ctx.comp_state = comp_state
        ctx.use_bias = bias is not None
        output = torch.matmul(input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    """
    [TODO] Backward All-Reduce of Megatron-LM
            Both MLP / Attention layers do all-reduce in this backward function
            Apply Gradient compression method here.
    """
    @staticmethod
    # pass reducer only for gradient compression
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        comp_state = ctx.comp_state
        grad_input = grad_output.matmul(weight)
        
        if comp_state['use_comp']:
            if comp_state['e'] == None: # if error feedback memeory is not init
                comp_state['e'] = torch.zeros_like(grad_input)
            grad_input += comp_state['e']
            mem_uninitialized = (comp_state['p'] == None)
            # Step 1. Calc Matrix Size and Allocate Memory
            p_total_size = 0
            q_total_size = 0
            n, m = grad_input.shape # already 2d matrix format
            rank = min(n, m, comp_state['comp_rank'])
            p_total_size += n*rank
            q_total_size += m*rank

            # [Important] Initialization on Device !!!
            if mem_uninitialized: # not initialized
                comp_state['p'] = torch.empty(p_total_size, device=grad_input.device, dtype=torch.float)
                comp_state['q'] = torch.empty(q_total_size, device=grad_input.device, dtype=torch.float)
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
            if grad_input.dtype != torch.float:
                torch.matmul(grad_input.float(), q_ptr, out=p_ptr)
            else:
                torch.matmul(grad_input, q_ptr, out=p_ptr)
            
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
            if grad_input.dtype != torch.float:
                torch.matmul(grad_input.t().float(), p_ptr, out=q_ptr)
            else:
                torch.matmul(grad_input.t(), p_ptr, out=q_ptr)
            
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
            grad_out = torch.zeros_like(grad_input)
            if grad_input.dtype != torch.float:
                with autocast():
                    grad_out.data[:] = torch.mm(p_ptr, q_ptr.t())
                    # torch.matmul(p, q.t(), out=out.data[:])
            else:
                torch.matmul(p_ptr, q_ptr.t(), out=grad_out.data[:])
            out_n_bits += n_bits(grad_out.data[:])

            # Step 9. (Algo 2: line 9) Memorize Local Errors
            comp_state['e'].data[:] = grad_input - grad_out

            # copy to grad_in
            grad_input.data.copy_(grad_out)

        else:
            # Asyncronous all-reduce
            handle = torch.distributed.all_reduce(
                    grad_input, group=get_tensor_model_parallel_group(), async_op=True)

        # Delay the start of weight gradient computation shortly (3us) to have
        # all-reduce scheduled first and have GPU resources allocated
        _ = torch.empty(1, device=grad_output.device) + 1
        grad_weight = grad_output.t().matmul(input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        # default reduce need handler wait
        if not comp_state['use_comp']:
            handle.wait()
        return grad_input, grad_weight, grad_bias , None

class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip 
                       adding bias but instead return it.
    """

    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add

        args = get_args()

        """
        [TODO] Set Gradient Reducer for ColumnParallelLinear
        """
        if args.mp_grad_comp:
            self.grad_comp = True
            if args.mp_grad_comp_type == 'PowerSGD':
                """
                [TODO] for memory efficiency... use custom powersgd
                """
                self.comp_state = {
                    'use_comp': False,
                    'comp_rank': args.mp_grad_comp_rank,
                    'use_error_feedback': True,
                    'p': None,
                    'q': None,
                    'e': None
                }
                self.start_iter = int(args.train_iters * args.mp_grad_comp_warm_up)
                self.current_iter = 0
        else:
            self.grad_comp = False

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size_per_partition,
                                                self.input_size,
                                                dtype=args.params_dtype))
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight, self.output_size, self.input_size,
                self.output_size_per_partition, 0, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size_per_partition, self.input_size,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=stride)

        if bias:
            if args.use_cpu_initialization:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition, dtype=args.params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=args.params_dtype))
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        self.async_tensor_model_parallel_allreduce = (
                not args.no_async_tensor_model_parallel_allreduce and
                world_size > 1)



    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None

        if self.async_tensor_model_parallel_allreduce:
            input_shape = input_.shape
            input_ = input_.view(input_shape[0] * input_shape[1],input_shape[2])
            # Maxtrix multiply with asynchronouse all-reduce execution
            """
            [TODO] Column Parallel backward all-reduce hook registeration.
            """
            if self.grad_comp:
                if self.current_iter < self.start_iter:
                    self.comp_state['use_comp'] = False
                else:
                    self.comp_state['use_comp'] = True
                output_parallel = ColumnParallelLinearWithPowerSGDAsyncAllreduce.apply(
                    input_, self.weight, bias, self.comp_state)
                self.current_iter += 1
            else:
                output_parallel = ColumnParallelLinearWithAsyncAllreduce.apply(
                        input_, self.weight, bias)
            output_parallel = output_parallel.view(
                    input_shape[0], input_shape[1], output_parallel.shape[1])
        else:
            # Set up backprop all-reduce.
            input_parallel = copy_to_tensor_model_parallel_region(input_)

            # Matrix multiply.
            output_parallel = F.linear(input_parallel, self.weight, bias)

        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

"""
[TODO] This linear layer is used after front MLP/Attention layer.
"""

class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.
    """

    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add

        args = get_args()
        """
        [TODO] Set Gradient Reducer for ColumnParallelLinear
        """
        if args.mp_acti_comp:
            self.acti_comp = True
            if args.mp_acti_comp_type == 'PowerSGD':
                """
                [TODO] for memory efficiency... use custom powersgd
                """
                self.comp_state = {
                    'use_comp': False,
                    'comp_rank': args.mp_acti_comp_rank,
                    'use_error_feedback': True,
                    'p': None,
                    'q': None,
                    'e': None
                }
                self.start_iter = int(args.train_iters * args.mp_acti_comp_warm_up)
                self.current_iter = 0
        else:
            self.acti_comp = False

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size,
                                                self.input_size_per_partition,
                                                dtype=args.params_dtype))
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight, self.output_size, self.input_size,
                self.input_size_per_partition, 1, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size, self.input_size_per_partition,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=1, stride=stride)
        if bias:
            if args.use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size,
                                                  dtype=args.params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size, device=torch.cuda.current_device(),
                    dtype=args.params_dtype))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)



    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
        # All-reduce across all the partitions.
        """
        [TODO] All-Reduce before dropout layer
                Apply activation compression here.
        """
        if self.acti_comp:
            if self.current_iter < self.start_iter:
                self.comp_state['use_comp'] = False
            else:
                self.comp_state['use_comp'] = True
            output_ = reduce_from_tensor_model_parallel_region_with_powersgd(
                        output_parallel, self.comp_state)
            self.current_iter += 1
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)

        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias

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