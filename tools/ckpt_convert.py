# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import os
import re
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import megatron

import argparse
import configparser
from datetime import datetime
import multiprocessing
from pathlib import Path

import numpy as np
import torch  # pytype: disable=import-error

from megatron import fused_kernels

from megatron import mpu
from megatron.checkpointing import load_checkpoint, save_checkpoint
from megatron.checkpointing import ensure_directory_exists
from megatron.checkpointing import get_checkpoint_name
from megatron.checkpointing import get_checkpoint_version
from megatron.checkpointing import get_checkpoint_tracker_filename
from megatron.global_vars import set_global_variables, get_args
from megatron.global_vars import rebuild_tokenizer


# Sequential layers (shared)
SEQ_LAYERS = [
    'input_layernorm.weight',
    'input_layernorm.bias',
    'self_attention.dense.bias',
    'post_attention_layernorm.weight',
    'post_attention_layernorm.bias',
    'mlp.dense_4h_to_h.bias',
    'final_layernorm.weight',
    'final_layernorm.bias'
]
# Row Parallel Linear (partition_dim=1, stride=1)
ROW_PAR_LAYERS = [
    'self_attention.dense.weight',
    'mlp.dense_4h_to_h.weight'
]

# Col Parallel Linear (partition_dim=0, stride=1)
COL_PAR_LAYERS = [
    'mlp.dense_h_to_4h.weight',
    'mlp.dense_h_to_4h.bias',
    'self_attention.query_key_value.bias',
    'self_attention.query_key_value.weight'
]


def _find_layer_type(key, layer_type_list):
    for layer in layer_type_list:
        if key.find(layer) != -1:
            return True
    return False

def get_model(model_type):
    if model_type == 'BERT':
        from pretrain_bert import model_provider
    elif model_type == 'GPT':
        from pretrain_gpt import model_provider
    elif model_type == 'RACE':
        from tasks.race.finetune import model_provider
    elif model_type == ['MNLI', 'QQP']:
        num_classes = 2
        if model_type == 'MNLI':
            num_classes = 3
        from megatron.model.classification import Classification
        def model_provider():
            return Classification(num_classes=num_classes, num_tokentypes=2)
    else:
        raise Exception('unrecognized model type: {}'.format(model_type))

    model = model_provider()
    # model = model.half()
    # We do not need FP16 version on megatron-cc
    return model

def _gpu_map_location(storage, loc):
    # force cpu -> we only need to convert... do not need to load on gpu
    return storage.cpu()

# This tool is used to support the new megatron model trained by pipeline parallel + tensor parallel
def merge_and_convert_process(merged_model, i, pipeline_para_rank, saved_dir, factor, key, model_args, transformer_model_list, ckpt_ver):
    saved_dir = Path(saved_dir)
    if key.find("layers.") != -1:
        layer_index = (int)(key[7 : key.find(".", 7)])
        saved_key = key.replace(
            "layers.%d." % layer_index,
            "layers.%d." % (layer_index + pipeline_para_rank * model_args.num_layers // model_args.pipeline_model_parallel_size))

        if saved_key.find("self_attention") != -1:
            pass
            # saved_key = saved_key.replace("self_attention", "attention")
            # do not change
    else:
        saved_key = key
    major_device = transformer_model_list[0][key].device

    if _find_layer_type(key, SEQ_LAYERS):
        # shared weights, only need to convert the weights of rank 0
        if i == 0:
            merged_model.state_dict()['language_model.encoder.' + saved_key].data.copy_(\
                                        transformer_model_list[0][key])
            print('language_model.encoder.' + saved_key)
    elif _find_layer_type(key, ROW_PAR_LAYERS):
        # Row Parallel Linear
        # partition_dim = 1, stride = 1
        vals = []
        for k in range(factor):
            vals.append(transformer_model_list[k][key].float().to(major_device))
        merged_model.state_dict()['language_model.encoder.'+saved_key].data.copy_(\
                torch.cat(vals, 1))
        print('language_model.encoder.' + saved_key)
    elif _find_layer_type(key, COL_PAR_LAYERS):
        # Column Parallel Linear
        # partition_dim = 0 stride = 1
        vals = []
        for k in range(factor):
            vals.append(transformer_model_list[k][key].float().to(major_device))
        merged_model.state_dict()['language_model.encoder.'+saved_key].data.copy_(\
                torch.cat(vals, 0))
        print('language_model.encoder.' + saved_key)
    else:
        print(f"[ERROR] cannot find key '{key}'")
        
def split_and_convert_process(i, pipeline_para_rank, saved_dir, factor, key, model_args, transformer_model_list, ckpt_ver):
    return False


class PrivateArgs:
    def __init__(self) -> None:
        self.tensor_model_parallel_size = 1
        self.pipeline_model_parallel_size = 1
        self.num_layers = 24

def convert_checkpoint(args, merged_model):
    saved_dir = Path(args.saved_dir) / f"{args.infer_gpu_num:d}-gpu"
    saved_dir.mkdir(parents=True, exist_ok=True)

    prefix = Path(args.in_file)
    ckpt_name = "model_optim_rng.pt"

    # load position_embedding from rank 0
    if (prefix / "mp_rank_00").is_dir():
        model_00 = torch.load((prefix / "mp_rank_00" / ckpt_name).as_posix(), map_location=_gpu_map_location)
    elif (prefix / "mp_rank_00_000").is_dir():
        model_00 = torch.load((prefix / "mp_rank_00_000" / ckpt_name).as_posix(), map_location=_gpu_map_location)
    else:
        print(f"[ERROR] Cannot find checkpoint in {prefix}.")
        exit(1)

    # print(model_00)

    if 'args' in model_00:
        model_args = model_00['args']
    else:
        model_args = PrivateArgs()
        print('No key args in first rank of pp and tp... may be ckpt version under 3!')

    merged_model.state_dict()['language_model.embedding.position_embeddings.weight'].copy_(\
            model_00["model"]["language_model"]["embedding"]["position_embeddings"]["weight"])
    del model_00

    w_e_list = []
    
    t_gpu_num = model_args.tensor_model_parallel_size
    i_gpu_num = args.infer_gpu_num

    if t_gpu_num > i_gpu_num:
        assert t_gpu_num % i_gpu_num == 0
        is_merge_ckpt = True
        factor = int(t_gpu_num / i_gpu_num)
        # we need to merge 8 gpus into 1 gpu
    else:
        assert i_gpu_num % t_gpu_num == 0
        is_merge_ckpt = False
        factor = int(i_gpu_num / t_gpu_num)

    main_loop = min(t_gpu_num, i_gpu_num) # only once cuz we need single ckpt
    
    for i in range(main_loop):
        for j in range(model_args.pipeline_model_parallel_size): # pp do not need merege... they are just layers
            if model_args.pipeline_model_parallel_size == 1:
                layer_rank_num = ""
            else:
                layer_rank_num = f"_{j:03d}"
            
            transformer_models = []
            if is_merge_ckpt == True: # we only regard merging case
                for k in range(factor): # 8 times... tp 8
                    m = torch.load((prefix / f"mp_rank_{i * factor + k:02d}{layer_rank_num}" / ckpt_name).as_posix(), map_location=_gpu_map_location)
                    transformer_models.append(m["model"]["language_model"]["encoder"]) # append tps...

                    if j == 0:
                        w_e_list.append(m["model"]["language_model"]["embedding"]["word_embeddings"]["weight"])
            else:
                m = torch.load(prefix / f"mp_rank_{i:02d}{layer_rank_num}/" / ckpt_name, map_location=_gpu_map_location)
            
                if j == 0:
                    w_e_list.append(m["model"]["language_model"]["embedding"]["word_embeddings"]["weight"])
                transformer_models.append(m["model"]["language_model"]["encoder"])

            for (k, v) in transformer_models[0].items():
                if is_merge_ckpt == True:
                    merge_and_convert_process(merged_model, i, j, saved_dir, factor, k, model_args, transformer_models, m["checkpoint_version"])
                else:
                    split_and_convert_process(merged_model, i, j, saved_dir, factor, k, model_args, transformer_models, m["checkpoint_version"])

    # cut embedding if dummy token exist...
    orig_emb_len = merged_model.state_dict()['language_model.embedding.word_embeddings.weight'].shape[0]
    merged_model.state_dict()['language_model.embedding.word_embeddings.weight'].data.copy_(torch.cat(w_e_list, 0).data[:orig_emb_len,:])
    


def _set_jit_fusion_options():
    """Set PyTorch JIT layer fusion options."""
    # flags required to enable jit fusion kernels
    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])
    if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10):
        # nvfuser
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(True)
        torch._C._debug_set_autodiff_subgraph_inlining(False)
    else:
        # legacy pytorch fuser
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)

def _compile_dependencies(args):

    # ==================
    # Load fused kernels
    # ==================

    start_time = time.time()
    # Custom kernel constraints check.
    seq_len = args.seq_length
    attn_batch_size = \
        (args.num_attention_heads / args.tensor_model_parallel_size) * \
        args.micro_batch_size
    # Constraints on sequence length and attn_batch_size to enable warp based
    # optimization and upper triangular optimization (for causal mask)
    custom_kernel_constraint = seq_len > 16 and seq_len <=2048 and \
        seq_len % 4 == 0 and attn_batch_size % 4 == 0

    fused_kernels.load(args)

    # Simple barrier to make sure all ranks have passed the
    # compilation phase successfully before moving on to the
    # rest of the program. We think this might ensure that
    # the lock is released.
    print('>>> done with compiling and loading fused kernels. '
              'Compilation time: {:.3f} seconds'.format(
                  time.time() - start_time), flush=True)



def get_mp_merge_args(parser):
    """Provide extra arguments required for merging."""
    group = parser.add_argument_group(title='mp merge')

    group.add_argument('--model-type', type=str, required=True,
                       choices=['BERT', 'GPT', 'RACE', 'MNLI', 'QQP'],
                       help='Type of the mdoel.')
    group.add_argument('--target-pipeline-model-parallel-size', type=int, default=1,
                       help='Degree of pipeline model parallelism in output model.')
    parser.add_argument("-saved_dir", "-o", type=str, help="file name of output file", required=True)
    parser.add_argument("-in_file", "-i", type=str, help="file name of input checkpoint file", required=True)
    parser.add_argument("-infer_gpu_num", "-i_g", type=int, help="How many gpus for inference", required=True)
    parser.add_argument("-processes", "-p", type=int, help="How many processes to spawn for conversion (default: 64)", default=64)
    parser.add_argument("-weight_data_type", type=str, default="fp32", choices=["fp32", "fp16"])

    return parser

if __name__ == "__main__":
    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    os.environ["WORLD_SIZE"] = f'{2**31}'

    # Args
    set_global_variables(extra_args_provider=get_mp_merge_args,
                         args_defaults = {'use_cpu_initialization': True,
                                          'micro_batch_size': 1,
                                          'no_load_optim': True,
                                          'no_load_rng': True,
                                          'no_save_optim': True,
                                          'no_save_rng': True,
                                          'save_interval': 1})
    args = get_args()

    _set_jit_fusion_options()
    _compile_dependencies(args)

    model_type = args.model_type
    orig_tensor_model_parallel_size = args.tensor_model_parallel_size
    orig_pipeline_model_parallel_size = args.pipeline_model_parallel_size


    args.tensor_model_parallel_size = 1
    args.pipeline_model_parallel_size = 1
    tokenizer = rebuild_tokenizer(args)



    print('\n merging model parallel partitions ...')
    print(' > number of tp partitions: {}'.format(orig_tensor_model_parallel_size))
    print(' > number of pp partitions: {}'.format(orig_pipeline_model_parallel_size))
    print(' > Total {} paratitions...!'.format(orig_tensor_model_parallel_size *\
                                    orig_pipeline_model_parallel_size))
    print(' > checkpoint path: {}'.format(args.load))
    print(' > model parameters:')
    print('    number of tokens ................ {} '.format(
        tokenizer.vocab_size))
    print('    number of layers ................ {}'.format(args.num_layers))
    print('    hidden size ..................... {}'.format(args.hidden_size))
    print('    number of attention heads ....... {}'.format(
        args.num_attention_heads))
    print('    maximum position embeddings ..... {}'.format(
        args.max_position_embeddings))

    # Full model.
    print('> building the full model ...')
    mpu.initialize.set_tensor_model_parallel_world_size(1)
    mpu.initialize.set_tensor_model_parallel_rank(0)
    mpu.initialize.set_pipeline_model_parallel_world_size(1)
    mpu.initialize.set_pipeline_model_parallel_rank(0)

    merged_model = get_model(model_type)
    print('> Full model generated !')
    print(merged_model)


    print("\n=============== Argument ===============")
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    print("========================================")
    print(merged_model.state_dict()['language_model.embedding.word_embeddings.weight'])
    start_time = datetime.now()    
    convert_checkpoint(args, merged_model)
    stop_time = datetime.now()
    run_time = (stop_time - start_time)
    print("[INFO] Spend {} (h:m:s) to convert the model".format(run_time))
    print(merged_model.state_dict()['language_model.embedding.word_embeddings.weight'])

    save_checkpoint(300000, [merged_model], None, None)