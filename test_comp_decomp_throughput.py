import torch
import sys
import numpy as np
import torch.distributed as dist
# For support fp16
from torch.cuda.amp import autocast
import time
from torch.profiler import profile, record_function, ProfilerActivity


# PowerSGD SVD for P2P Communication
class PowerSVD:
    def __init__(self, random_seed, device, reuse_query=True,\
                grad_rank=12, use_error_feedback=True, fp16=False):
        self.rng = np.random.RandomState(random_seed)
        M = 1024 * 1024
        self.precalc_numbers = (
            torch.from_numpy(self.rng.randn(128 * M)).to(device).type(torch.float32)
        )
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

        # reuse_query => warm-start (in PowerSGD paper)
        # in most cases it is essential to be True.
        self.reuse_query = reuse_query
        # EF SGD enabled?
        self.use_error_feedback = use_error_feedback
        # support fp16?
        self.fp16 = fp16

        # self._init_printer()

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
    def p2p_grad_compress(self, tensor_in, seq_length, micro_batch_size, hidden_size):
        """
        [MP PowerSGD]
        : Grad_in tensor is already 2dim (no multi layer type!)
        """
        # We'll use error feedback but uninitialized
        if self.use_error_feedback and self.memories == None:
            self.memories = torch.zeros_like(tensor_in)

        # add EF
        if self.use_error_feedback:
            tensor_in += self.memories

        mem_uninitialized = self.p_memory is None

        n = seq_length * micro_batch_size
        m = tensor_in.nelement() // n

        rank = min(n, m, self.rank)
        p_shape = (n, rank)
        q_shape = (m, rank)

        if True:
            tensor_shape = (seq_length, micro_batch_size, hidden_size)
            tensor_chunk_shape = seq_length * micro_batch_size * hidden_size
            if tensor_chunk_shape % 8 == 0:            
                tensor_chunk_shape = tensor_chunk_shape // 8
                # assert args.micro_batch_size % 8 == 0, 'our compressed inter communication only support microbatchsize = 8n'
                n = seq_length * (micro_batch_size // 8)
                m = tensor_chunk_shape // n
                rank = min(n, m, self.rank)
                p_shape=(n, rank)
                q_shape=(m, rank)

        # now we found n,m .... viewing it in 2d!
        tensor_in = tensor_in.view(n, m)

        # Step 1. Calc Matrix Size and Allocate Memory
        p_total_size = 0
        q_total_size = 0

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

        self.p_shape = p_ptr.shape
        self.q_shape = q_ptr.shape

        return p_ptr, q_ptr
    # it's just helper method for decompress....
    def p2p_grad_recv_decompress(self, p_ptr, q_ptr, out):
        return torch.matmul(p_ptr, q_ptr.t(), out=out.data[:])

    def get_current_grad_error(self):
        return self.memories


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



if __name__ == '__main__':
    # 2.5B
    ranks = [8, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    micro_batch_size = 8
    seq_length = 1024
    hidden_size = 1920
    tensor_shape = (micro_batch_size, seq_length, hidden_size)
    n, m = seq_length * micro_batch_size, hidden_size

    for rank in ranks:
        # we use sc21 scatter gather optimization
        n = seq_length * (micro_batch_size // 8) # tp 8
        m = micro_batch_size * seq_length * hidden_size // n
        p_shape = (n, rank)
        q_shape = (m, rank)
        compressor = PowerSVD(random_seed=714, device='cuda:0', reuse_query=True,\
                            grad_rank=rank, use_error_feedback=True, fp16=False)
        tensor_full = torch.empty((micro_batch_size*seq_length*hidden_size//8),
                            requires_grad=False,
                            device='cuda:0',
                            dtype=torch.float)
        tensor_p = torch.empty(p_shape,
                            requires_grad=False,
                            device='cuda:0',
                            dtype=torch.float)
        tensor_q = torch.empty(q_shape,
                            requires_grad=False,
                            device='cuda:0',
                            dtype=torch.float)
        full_element_n = tensor_full.nelement()
        pq_element_n = tensor_p.nelement() + tensor_q.nelement()

        # warm-up trial
        p, q = compressor.p2p_grad_compress(tensor_full, seq_length=seq_length, micro_batch_size=micro_batch_size,\
                                                            hidden_size=hidden_size)
        compressor.p2p_grad_recv_decompress(p, q, tensor_full)


        print('==========   2.5 B    ===============')
        print('Tensor Shape: ', tensor_shape)
        print('P Shape: ', p_shape, '/ Q Shape:', q_shape)
        print('Compression RATIO OvO..: ', pq_element_n/full_element_n*100, '(ratio) / ', full_element_n/pq_element_n, 'x times')
        print('compression.... rank: ', rank)
        print('total element (Byte): ', 8 * tensor_full.nelement() * tensor_full.element_size())

        with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                p, q = compressor.p2p_grad_compress(tensor_full, seq_length=seq_length, micro_batch_size=micro_batch_size,\
                                                                    hidden_size=hidden_size)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        print()
        print('decompression.... rank: ', rank)
        print('total element (Byte): ', 8 * tensor_full.nelement() * tensor_full.element_size())

        with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            compressor.p2p_grad_recv_decompress(p, q, tensor_full)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print('=============================')
        print()

    # 8.3B
    ranks = [8, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    micro_batch_size = 8
    seq_length = 1024
    hidden_size = 3072
    tensor_shape = (micro_batch_size, seq_length, hidden_size)
    n, m = seq_length * micro_batch_size, hidden_size

    for rank in ranks:
        # we use sc21 scatter gather optimization
        n = seq_length * (micro_batch_size // 8) # tp 8
        m = micro_batch_size * seq_length * hidden_size // n
        p_shape = (n, rank)
        q_shape = (m, rank)
        compressor = PowerSVD(random_seed=714, device='cuda:0', reuse_query=True,\
                            grad_rank=rank, use_error_feedback=True, fp16=False)
        tensor_full = torch.empty((micro_batch_size*seq_length*hidden_size//8),
                            requires_grad=False,
                            device='cuda:0',
                            dtype=torch.float)
        tensor_p = torch.empty(p_shape,
                            requires_grad=False,
                            device='cuda:0',
                            dtype=torch.float)
        tensor_q = torch.empty(q_shape,
                            requires_grad=False,
                            device='cuda:0',
                            dtype=torch.float)
        full_element_n = tensor_full.nelement()
        pq_element_n = tensor_p.nelement() + tensor_q.nelement()


        # warm-up trial
        p, q = compressor.p2p_grad_compress(tensor_full, seq_length=seq_length, micro_batch_size=micro_batch_size,\
                                                            hidden_size=hidden_size)
        compressor.p2p_grad_recv_decompress(p, q, tensor_full)


        print('==========   8.3 B    ===============')
        print('Tensor Shape: ', tensor_shape)
        print('P Shape: ', p_shape, '/ Q Shape:', q_shape)
        print('Compression RATIO OvO..: ', pq_element_n/full_element_n*100, '(ratio) / ', full_element_n/pq_element_n, 'x times')
        print('compression.... rank: ', rank)
        print('total element (Byte): ', 8 * tensor_full.nelement() * tensor_full.element_size())

        with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                p, q = compressor.p2p_grad_compress(tensor_full, seq_length=seq_length, micro_batch_size=micro_batch_size,\
                                                                    hidden_size=hidden_size)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        print()
        print('decompression.... rank: ', rank)
        print('total element (Byte): ', 8 * tensor_full.nelement() * tensor_full.element_size())

        with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            compressor.p2p_grad_recv_decompress(p, q, tensor_full)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print('=============================')
        print()



    # 175B
    ranks = [8, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    micro_batch_size = 8
    seq_length = 2048
    hidden_size = 12288
    tensor_shape = (micro_batch_size, seq_length, hidden_size)
    n, m = seq_length * micro_batch_size, hidden_size

    for rank in ranks:
        # we use sc21 scatter gather optimization
        n = seq_length * (micro_batch_size // 8) # tp 8
        m = micro_batch_size * seq_length * hidden_size // n
        p_shape = (n, rank)
        q_shape = (m, rank)
        compressor = PowerSVD(random_seed=714, device='cuda:0', reuse_query=True,\
                            grad_rank=rank, use_error_feedback=True, fp16=False)
        tensor_full = torch.empty((micro_batch_size*seq_length*hidden_size//8),
                            requires_grad=False,
                            device='cuda:0',
                            dtype=torch.float)
        tensor_p = torch.empty(p_shape,
                            requires_grad=False,
                            device='cuda:0',
                            dtype=torch.float)
        tensor_q = torch.empty(q_shape,
                            requires_grad=False,
                            device='cuda:0',
                            dtype=torch.float)
        full_element_n = tensor_full.nelement()
        pq_element_n = tensor_p.nelement() + tensor_q.nelement()


        # warm-up trial
        p, q = compressor.p2p_grad_compress(tensor_full, seq_length=seq_length, micro_batch_size=micro_batch_size,\
                                                            hidden_size=hidden_size)
        compressor.p2p_grad_recv_decompress(p, q, tensor_full)


        print('==========   175 B    ===============')
        print('Tensor Shape: ', tensor_shape)
        print('P Shape: ', p_shape, '/ Q Shape:', q_shape)
        print('Compression RATIO OvO..: ', pq_element_n/full_element_n*100, '(ratio) / ', full_element_n/pq_element_n, 'x times')
        print('compression.... rank: ', rank)
        print('total element (Byte): ', 8 * tensor_full.nelement() * tensor_full.element_size())

        with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                p, q = compressor.p2p_grad_compress(tensor_full, seq_length=seq_length, micro_batch_size=micro_batch_size,\
                                                                    hidden_size=hidden_size)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        print()
        print('decompression.... rank: ', rank)
        print('total element (bit): ', 8 * tensor_full.nelement() * tensor_full.element_size())

        with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            compressor.p2p_grad_recv_decompress(p, q, tensor_full)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print('=============================')
        print()

@torch.jit.script
def add_error_feedback(t1, t2):
    torch.add(t1, t2, out=t1)

@torch.jit.script
def update_error_feedback(e, t, o):
    torch.add(t, o, alpha=(-1), out=e)

def n_bits(tensor):
    return 8 * tensor.nelement() * tensor.element_size()