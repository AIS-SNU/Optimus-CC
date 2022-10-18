#!/bin/bash

WORLD_SIZE=4

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6125"

TRAIN_DATA="data/mnli/train.tsv"
VALID_DATA="data/mnli/dev_matched.tsv \
            data/mnli/dev_mismatched.tsv"
PRETRAINED_CHECKPOINT=/dataset/{path}/checkpoints/bert_345m
VOCAB_FILE=bert-large-uncased-vocab.txt
CHECKPOINT_PATH=/dataset/{path}/checkpoints/bert_345m_mnli

python -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/main.py \
               --grad_comp \
               --grad_comp_type RandomizedSVD \
               --grad_comp_warm_up 0.005 \
               --grad_comp_rank 4 \
               --task MNLI \
               --seed 1234 \
               --train-data $TRAIN_DATA \
               --valid-data $VALID_DATA \
               --tokenizer-type BertWordPieceLowerCase \
               --vocab-file $VOCAB_FILE \
               --epochs 1 \
               --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
               --tensor-model-parallel-size 2 \
               --pipeline-model-parallel-size 2 \
               --num-layers 24 \
               --hidden-size 1024 \
               --num-attention-heads 16 \
               --micro-batch-size 8 \
               --activations-checkpoint-method uniform \
               --lr 5.0e-5 \
               --lr-decay-style linear \
               --lr-warmup-fraction 0.065 \
               --seq-length 512 \
               --max-position-embeddings 512 \
               --save-interval 500000 \
               --save $CHECKPOINT_PATH \
               --log-interval 10 \
               --eval-interval 100 \
               --eval-iters 50 \
               --weight-decay 1.0e-1 \
