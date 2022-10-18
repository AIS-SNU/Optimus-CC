import numpy as np

import torch
import torch.distributed as dist

# For support fp16
from torch.cuda.amp import autocast



class ColumnParallelLinearWithPowerSGDAsyncAllreduce(torch.autograd.Function):
    def __init__(self, rank):
        super().__init__()
        self.e = None
        self.p = None
        self.q = None
        self.rank = rank
    """
    Column-parallel linear layer execution with asynchronous all-reduce
    execution in backprop.
    """
    # @staticmethod
    def forward(self, ctx, input, weight, bias, use_comp):
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        ctx.use_comp = use_comp
        output = torch.matmul(input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    """
    [TODO] Backward All-Reduce of Megatron-LM
            Both MLP / Attention layers do all-reduce in this backward function
            Apply Gradient compression method here.
    """
    # @staticmethod
    # pass reducer only for gradient compression
    def backward(self, ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        use_comp = ctx.use_comp
        grad_input = grad_output.matmul(weight)
        
        if use_comp:
            if self.e == None: # if size is same as our useless tensor
                self.e = torch.zeros_like(grad_input)
            grad_input += self.e
            mem_uninitialized = (self.p == None)
            # Step 1. Calc Matrix Size and Allocate Memory
            p_total_size = 0
            q_total_size = 0
            n, m = grad_input.shape # already 2d matrix format
            rank = min(n, m, self.rank)
            p_total_size += n*rank
            q_total_size += m*rank

            # [Important] Initialization on Device !!!
            if mem_uninitialized: # not initialized
                self.p = torch.empty(p_total_size, device=grad_input.device, dtype=torch.float)
                self.q = torch.empty(q_total_size, device=grad_input.device, dtype=torch.float)
            # for easier implementation, gather pointers
            p_ptr = self.p.view(n, rank)
            q_ptr = self.q.view(m, rank)

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
                    self.p, group=get_tensor_model_parallel_group())

            # if dist.get_rank() == 0:
            #     print(' > Compressed P Matrix: ', n_bits(p), 'bits')

            # it's different from original PowerSGD code...
            # if there's another degradation in accurcy
            # uncomment this line for accuracy regain
            # self.p_memory.data[:] /= self.n_workers

            # Step 5. (Algo 1: line 5) P_hat <- ORTHOGONALIZE(P)
            orthogonalize(p_ptr)

            # if dist.get_rank() == 0:
            #     print(p.data[:100])

            # Step 6. (Algo 1: line 6) Q <- M_T P_hat
            if grad_input.dtype != torch.float:
                torch.matmul(grad_input.t().float(), p_ptr, out=q_ptr)
            else:
                torch.matmul(grad_input.t(), p_ptr, out=q_ptr)
            
            # Step 7. (Algo 1: line 7) ALL_REDUCE_MEAN(Q)
            torch.distributed.all_reduce(
                    self.q, group=get_tensor_model_parallel_group())

            # if dist.get_rank() == 0:
            #     print(' > Compressed Q Matrix: ', n_bits(q), 'bits')

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
            self.e.data[:] = grad_input - grad_out

            # copy to grad_in
            grad_input.data.copy_(grad_out)

            # remove temp grad space
            del grad_out
            # torch.cuda.empty_cache()

            # if dist.get_rank() == 0:
            #     print(' > Original Matrix: ', out_n_bits, 'bits')
            #     if (n_bits(p)+n_bits(q)) != 0:
            #         print(' > Compression Ratio: ', \
            #                 out_n_bits/(n_bits(p)+n_bits(q)))
            pass
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
        if not use_comp:
            handle.wait()
        return grad_input, grad_weight, grad_bias , None


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