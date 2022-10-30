#! /bin/bash

export PATH=/apps/mpi/gcc/RHEL8/ucx-1.11.0/bin:/apps/mpi/gcc/RHEL8/openmpi-4.1.0-cu111-ucx1110/bin:$PATH
export LD_LIBRARY_PATH=/apps/mpi/gcc/RHEL8/ucx-1.11.0/lib:/apps/mpi/gcc/RHEL8/openmpi-4.1.0-cu111-ucx1110/lib:/usr/lib64/libibverbs:/usr/lib64:/usr/lib:/usr/local:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=128
export NCCL_IB_TIMEOUT=220
export NCCL_ASYNC_ERROR_HANDLING=0

DATE=$(date +%Y%m%d_%H-%M-%S)
DATA_PATH={GPT_data_path}

GPUS_PER_NODE=8
NNODES=`echo $HOSTLIST | wc -w`
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH={checkpoint_path}_${DATE}
echo "checkpoint path : ${CHECKPOINT_PATH}"



TENSOR_MP_SIZE=8
PIPELINE_MP_SIZE=4


TOTAL_ARGS="--num-layers 24 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --micro-batch-size 8 \
            --global-batch-size 512 \
            --seq-length 1024 \
            --max-position-embeddings 1024 \
            --train-iters 230000 \
            --lr-decay-iters 320000 \
            --save $CHECKPOINT_PATH \
            --load $CHECKPOINT_PATH \
            --data-path $DATA_PATH \
            --vocab-file gpt2-vocab.json \
            --merge-file gpt2-merges.txt \
            --data-impl mmap \
            --split 949,50,1 \
            --distributed-backend nccl \
            --lr 0.00015 \
            --lr-decay-style cosine \
            --min-lr 1.0e-5 \
            --weight-decay 1e-2 \
            --clip-grad 1.0 \
            --lr-warmup-fraction .01 \
            --activations-checkpoint-method uniform \
            --save-interval 100 \
            --eval-interval 100 \
            --eval-iters 10 \
            --tensor-model-parallel-size $TENSOR_MP_SIZE \
            --pipeline-model-parallel-size $PIPELINE_MP_SIZE \
            --experiment_name GPT2-2.5B-CB-FE-SC-Breakdown \
            --DDP-impl local \
            --grad_comp \
            --grad_comp_rank 128 \
            --grad_comp_warm_up 0.0 \
            --selective_grad_comp \
            --selective_grad_comp_way 3 \
            --inter_grad_comp \
            --inter_grad_comp_rank 16 \
            --inter_grad_comp_epilogue_only \
            --emb_comm_opt \
            --use_error_feedback \
            --log-interval 1"

NODE_RANK=`cat $node_rankfile | grep $HOSTNAME | awk '{print $2}'`
NNODES=`cat $node_rankfile | wc -l`

DISTRIBUTED_ARGS="--nproc_per_node 8 --nnodes $NNODES --master_addr $MASTER_ADDR --master_port 12223"

echo "HOSTNAME: $HOSTNAME"
echo "NODE_RANK: $NODE_RANK"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "NNODES: $NNODES"
echo "MASTER_ADDR: $MASTER_ADDR"


python -m torch.distributed.launch $DISTRIBUTED_ARGS --node_rank $NODE_RANK ../pretrain_gpt.py $TOTAL_ARGS



