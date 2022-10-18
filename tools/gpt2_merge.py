import argparse
from operator import truediv
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import torch
from collections import OrderedDict
from pathlib import Path
import megatron


# Megatron Checkpoint Arguments
MODEL_KEY = 'model'
ARGS_KEY = 'args'
LANGUAGE_MODEL_KEY = 'language_model'
EMBEDDING_KEY = 'embedding'
ENCODER_KEY = 'encoder'
POSITION_EMBEDDINGS_KEY = 'position_embeddings'
WORD_EMBEDDINGS_FOR_HEAD_KEY = 'word_embeddings_for_head'
WORD_EMBEDDINGS_KEY = 'word_embeddings'
FINAL_LAYER_NORM_KEY ='final_layernorm'
CHECKPOINT_VERSION_KEY = 'checkpoint_version'
CHECKPOINT_VERSION_VALUE = 3.0
ITERATION_KEY = 'iteration'

# Additional Arguments for merging
SEQUENTIAL_LAYERS = [
    'input_layernorm.weight', 'input_layernorm.bias',
    'self_attention.dense.bias',
    'post_attention_layernorm.weight', 'post_attention_layernorm.bias',
    'mlp.dense_4h_to_h.bias',
    'position_embeddings.weight',
    'final_layernorm.weight',
    'final_layernorm.bias'
]

LAYER_CONCAT_DIM = {
    'self_attention.dense.weight': 1,
    'mlp.dense_4h_to_h.weight': 1
}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default=None, type=str, help='Input Megatron checkpoint folder')
    parser.add_argument('--output_folder', default=None, type=str, help='Output Megatron checkpoint folder')
    parser.add_argument('--target_tp', default=1, type=int, help='Target TP degree')
    parser.add_argument('--target_pp', default=1, type=int, help='Target PP degree')
    parser.add_argument('--for_release', action='store_true', help='Convert for release purpose, reset some (progress) counters.')
    args = parser.parse_args()
    print(f'args = {args}')
    return args

def _is_in_this_layer_type(name, type_layer_list):
    for layer in type_layer_list:
        if layer.find(name) != -1:
            return True
    return False

def _create_checkpoint_paths(base_folder, iteration, tp_degree, pp_degree):
    path_list = []
    iter_folder = f'iter_{iteration:07d}'
    for i in range(0, tp_degree):
        path_list.append([])
        for j in range(0, pp_degree):
            rank_folder = f'mp_rank_{i:02d}' if pp_degree == 1 else f'mp_rank_{i:02d}_{j:03d}'
            ckpt_path = os.path.join(rank_folder, 'model_optim_rng.pt')
            path_list[i].append(os.path.join(base_folder, iter_folder, ckpt_path))

    return path_list

def _create_megatron_dict():
    language_model_dict = {
        EMBEDDING_KEY: {},
        ENCODER_KEY: {}
    }
    megatron_dict = {
        MODEL_KEY: {LANGUAGE_MODEL_KEY: language_model_dict},
        CHECKPOINT_VERSION_KEY: CHECKPOINT_VERSION_VALUE
    }
    return megatron_dict

def _save_checkpoint(file_path, chkpt_sd):
    dir, _ = os.path.split(file_path)
    os.makedirs(dir, exist_ok=True)
    torch.save(chkpt_sd, file_path)

def _create_latest_file(base_folder, iteration):
    file_path = os.path.join(base_folder, 'latest_checkpointed_iteration.txt')
    os.makedirs(base_folder, exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(str(iteration))

# def _merge_partitions(merged, partitions, partition_dim, stride):
def _merge_partitions(partitions, partition_dim, stride):
    # Number and size of each partition.
    num_partitions = len(partitions)
    per_partition_size = None
    for partition in partitions:
        if per_partition_size is None:
            per_partition_size = partition.size(partition_dim)
        else:
            assert per_partition_size == partition.size(partition_dim)

    def concat_partitions(partitions_):
        with torch.no_grad():
            return torch.cat(partitions_, dim=partition_dim)
            # if (per_partition_size * num_partitions) == merged.size(
            #         partition_dim):
            #     torch.cat(partitions_, dim=partition_dim, out=merged)
            # else:
            #     print('     ***WARNING*** sizes do not match. Will cut '
            #           'the merged partitions by {} along dimension {} '
            #           'to reduce the size from {} to {} ...'.format(
            #               (per_partition_size * num_partitions) - \
            #               merged.size(partition_dim), partition_dim,
            #               per_partition_size * num_partitions,
            #               merged.size(partition_dim)))
            #     merged_ = torch.cat(partitions_, dim=partition_dim)
            #     merged_split = torch.split(merged_, merged.size(partition_dim),
            #                                dim=partition_dim)
            #     merged_ = merged_split[0]
            #     assert merged_.size(partition_dim) == merged.size(partition_dim)
            #    merged.data.copy_(merged_.data)

    # If stride is 1, then do simple concatination.
    if stride == 1:
        merged = concat_partitions(partitions)
        return merged

    # For none unity strides, first split based on stride and then group.
    per_partition_per_stride_size = mpu.utils.divide(per_partition_size, stride)
    # Chunk and build a list.
    chunks = None
    for i, partition in enumerate(partitions):
        chunk = torch.split(partition,
                            per_partition_per_stride_size,
                            dim=partition_dim)

        if chunks is None:
            chunks = [0]*(num_partitions*len(chunk))
        chunks[i::num_partitions] = chunk

    # Concatinate.
    merged = concat_partitions(chunks)
    return merged

def convert():
    print('Convert megatron checkpoint to target configuration megatron checkpoint.')

    args = parse_arguments()
    assert args.input_folder != args.output_folder, 'input path and output path should be different!'
    assert args.target_tp == 1, 'only support tp=1'
    assert args.target_pp == 1, 'only support pp=1'

    print(f'Converting megatron ckpt in {args.input_folder} to target config megatron ckpt in {args.output_folder}')

    prefix = Path(args.input_folder)
    ckpt_name = 'model_optim_rng.pt'

    # load informations from rank 0
    if (prefix / 'mp_rank_00').is_dir():
        model_00 = torch.load((prefix / "mp_rank_00" / ckpt_name).as_posix(), map_location='cpu')
    elif (prefix / 'mp_rank_00_000').is_dir():
        model_00 = torch.load((prefix / "mp_rank_00_000" / ckpt_name).as_posix(), map_location='cpu')
    else:
        print('[Error] Wrong input folder...')
        exit(1)
    
    model_args = model_00[ARGS_KEY]
    iteration = model_00[ITERATION_KEY]
    orig_tp_size = model_00[ARGS_KEY].tensor_model_parallel_size
    orig_pp_size = model_00[ARGS_KEY].pipeline_model_parallel_size

    _create_latest_file(args.output_folder, iteration)
    checkpoint_paths = _create_checkpoint_paths(args.output_folder, iteration, args.target_tp, args.target_pp)

    checkpoint_sd = _create_megatron_dict()

    # need to copy all arguments in originam megatron ckpt

    checkpoint_sd[ITERATION_KEY] = iteration
    
    meg_embedding_sd = OrderedDict()
    # get positional embedding
    meg_embedding_sd[POSITION_EMBEDDINGS_KEY] = {'weight': None}
    meg_embedding_sd[POSITION_EMBEDDINGS_KEY]['weight'] = \
                    torch.tensor(model_00[MODEL_KEY][LANGUAGE_MODEL_KEY][EMBEDDING_KEY][POSITION_EMBEDDINGS_KEY]['weight'])

    meg_encoder_sd = OrderedDict()
    meg_embedding_for_head_sd = OrderedDict()

    w_e_list = [] # word embedding list for merge

    for pp_index in range(orig_pp_size):
        if orig_pp_size == 1:
            layer_rank_num = ''
        else:
            layer_rank_num = f'_{pp_index:03d}'
        encoders = []
        for tp_index in range(orig_tp_size):
            m = torch.load((prefix / f"mp_rank_{0*orig_tp_size + tp_index:02d}{layer_rank_num}" / ckpt_name).as_posix(), map_location='cpu')
            encoders.append(m[MODEL_KEY][LANGUAGE_MODEL_KEY][ENCODER_KEY])
            if tp_index == 0:
                w_e_list.append(m[MODEL_KEY][LANGUAGE_MODEL_KEY][EMBEDDING_KEY][WORD_EMBEDDINGS_KEY]['weight'])
        # each tp partition has same structures
        for (k, v) in encoders[0].items():
            print('original key: ', k)
            print('value: ', v)
            if k.find("layers.") != -1:
                layer_index = (int)(k[7 : k.find(".", 7)])
                new_key = k.replace(
                    "layers.%d." % layer_index,
                    "layers.%d." % (layer_index + pp_index * model_args.num_layers // orig_pp_size))
            else:
                new_key = k
            if _is_in_this_layer_type(new_key, SEQUENTIAL_LAYERS)  and tp_index == 0:
                meg_encoder_sd[new_key] = env
            else:
                partitions = []
                for encoder in encoders:
                    partitions.append(encoder[k])
                dim = v['partition_dim']
                stride = v['partition_stride']
                merged = _merge_partitions(partitions, dim, stride)
                meg_encoder_sd[new_key] = merged
                print('new key: ', new_key)
        print(meg_encoder_sd.keys())


    # merge embedding and update embedding for checkpoint
    meg_embedding_sd[WORD_EMBEDDINGS_KEY]['weight'] = torch.cat(w_e_list, 0)
    checkpoint_sd[MODEL_KEY][LANGUAGE_MODEL_KEY][EMBEDDING_KEY] = meg_embedding_sd

    # update encoder state from 
    checkpoint_sd[MODEL_KEY][LANGUAGE_MODEL_KEY][ENCODER_KEY] = meg_encoder_sd
    
    print(checkpoint_sd)
    # not used for single model
    # checkpoint_sd[MODEL_KEY][WORD_EMBEDDINGS_FOR_HEAD_KEY] = meg_embedding_for_head_sd

    if args.for_release:
        checkpoint_sd[ARGS_KEY].consumed_train_samples = 0
        checkpoint_sd[ARGS_KEY].consumed_valid_samples = 0

    _save_checkpoint(checkpoint_paths[0][0], checkpoint_sd)
    
    


if __name__ == '__main__':
    convert()