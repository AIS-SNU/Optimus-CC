#!/bin/bash

TENSOR_MODEL_PARALLEL_SIZE=1

VOCAB_FILE=/workspace/datasets/mnli/bert-large-uncased-vocab.txt
CHECKPOINT_PATH=/workspace/checkpoints/bert_345m

WORLD_SIZE=$TENSOR_MODEL_PARALLEL_SIZE python ../tools/mp_chkpt_generator.py \
                                --model-type BERT \
                                --target-pipeline-model-parallel-size 1 \
                                --target-tensor-model-parallel-size 4 \
                                --tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE \
                                --tokenizer-type BertWordPieceLowerCase \
                                --vocab-file $VOCAB_FILE \
                                --num-layers 24 \
                                --hidden-size 1024 \
                                --num-attention-heads 16 \
                                --seq-length 512 \
                                --max-position-embeddings 512 \
                                --load $CHECKPOINT_PATH