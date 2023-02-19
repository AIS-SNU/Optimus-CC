#!/bin/bash

DATE=$(date +%Y%m%d_%H-%M-%S)
DATA_PATH="/datasets/Megatron-LM_data/my-gpt2_text_document"
VOCAB_PATH="/datasets/Megatron-LM_data"

GPUS_PER_NODE=8
NNODES=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH="/checkpoints/gpt_345m_opt_cc/"
echo "checkpoint path : ${CHECKPOINT_PATH}"



TENSOR_MP_SIZE=2
PIPELINE_MP_SIZE=2

TOTAL_ARGS="--num-layers 24 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --micro-batch-size 8 \
            --global-batch-size 512 \
            --seq-length 1024 \
            --max-position-embeddings 1024 \
            --train-iters 10 \
            --lr-decay-iters 320000 \
            --save $CHECKPOINT_PATH \
            --load $CHECKPOINT_PATH \
            --data-path $DATA_PATH \
            --vocab-file $VOCAB_PATH/gpt2-vocab.json \
            --merge-file $VOCAB_PATH/gpt2-merges.txt \
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
            --experiment_name GPT2-335M-CB-Breakdown \
            --DDP-impl local \
            --inter_grad_comp \
            --inter_grad_comp_rank 16 \
            --inter_grad_comp_epilogue_only \
            --use_error_feedback \
            --log-interval 1"

DISTRIBUTED_ARGS="--nproc_per_node 8 --nnodes $NNODES --master_addr localhost --master_port 12223"

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NNODES: $NNODES"

torchrun $DISTRIBUTED_ARGS ../../pretrain_gpt.py $TOTAL_ARGS




