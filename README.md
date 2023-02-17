# Optimus-CC [![DOI](https://zenodo.org/badge/553339236.svg)](https://zenodo.org/badge/latestdoi/553339236)
[ASPLOS'23] Optimus-CC: Efficient Large NLP Model Training with 3D Parallelism Aware Communication Compression (Accepted, to appear)

Our codes are based on Megatron-LM (https://github.com/NVIDIA/Megatron-LM, v2.5) and PowerSGD (https://github.com/epfml/powersgd).

**UPDATE**: We are working on adopting Opt-CC on new Megatron-LM v3.0 with new PowerSGD code, which is faster :)

## Artifact Evaluation
This repository is for AE (Artifact Evaluation) process of ASPLOS'23.

In ASPLOS23/ folder, scripts for pretraining (TABLE 2), speedup check (TABLE 2, Fig. 10), memory consumption check (Fig. 12), comp/decomp throughput check (Fig. 14), and cosine similarity check (Fig. 11) are available.
We give a detailed guideline for these evaluations in `Evaluation Reproducing` section.
For accuracy check of zero-shot (TABLE 3 and TABLE 4), the process is quite complex, so please refer to `Zero-Shot Task Running` section. Note that training script for TABLE 4 is available in TABLE 2 training script folder.
Other experiments (not main evaluation) for figures can be run by changing options in speedup check scripts.

**UPDATE**: GPT-335M version scripts are added to `ASPLOS23/fig10/` directory to test functionality in a small clutster.

Dataset making is explained in `Dataset Preprocessing`. Make the pretraining dataset based on the guideline and use the binarized dataset.

For detailed arguments and settings, please refer to below explanations.

## Licenses
For the baseline codes (Megatron-LM and PowerSGD), please follow their licenses and guidelines.
For additional codes, please follow the MIT license.

## Environment

We conducted pretraining experiments of 2.5B and 8.3B GPT models on a large data center cluster with NVIDIA A100 GPUs. Our GPU Box (Node) interconnects with 200Gbps Infiniband.

We used the NGC's PyTorch docker container version 20.12, python 3.8, PyTorch 1.8, CUDA 11.1, and NCCL 2.8.3.

In addition, we converted this image into a singularity image to use in the IBM LSF scheduler.
(Use `singularity build --sandbox {singularity_name} {docker tar path}` after save docker into `.tar`.)

Refer to https://github.com/NVIDIA/Megatron-LM/blob/main/README.md for distributed execution.

## Megatron-CC Arguments

Below is the explanation of the arguments to use Megatron-CC's main schemes.

### A. Compressed Backpropagation (CB)

```shell
--inter_grad_comp \ # enable CB
--inter_grad_comp_rank 12 \ # set CB compression rank
--inter_grad_comp_epilogue_only \ # set CB epilogue only
--use_error_feedback # use Lazy Error Propagation (LEP) on CB
```

### B. Fused Embedding Synchronization (FE)

```shell
--emb_comm_opt \ # enable FE
```

### C. Selective Stage Compression (SC)

```shell
# SC need data-parallel gradient compression
--grad_comp \ # enable data-parallel gradient compression
--grad_comp_rank 128 \ # set data-parallel gradient compression rank
--grad_comp_warm_up 0.1 \ # set PowerSGD warm-up period
--use_error_feedback \ # use error feedback on PowerSGD
--selective_grad_comp \ # enable selective stage compression (SC)
--selective_grad_comp_way # set how many stages you want to compress
```

If you want to check the validity of `Lazy Error Propagation (LEP)`'s orthogonality and average, use below arguments.

```shell
--check_orth_error_acti
```

This argument will print out the cosine similarity and averages.

## Evaluation Reproducing

All evaluation reproducing scripts are in `ASPLOS23/`. Basic pretraining scripts are in `examples/`.

These scripts use IBM `LSF` Scheduler, but they can be changed into other scheduler formats (e.g., `Slurm`).

Below are the scripts that can reproduce the important evaluation results in our paper.

Use them by `bsub < lsf_job_sumit.sub` after replace the `{sh_script_for_experiment_to_execute}` in `lsf_job_submit.sub` by adequate `.sh` file.

Replace `{some_argument}` with proper values.

- Table2 and Fig. 9: `ASPLOS23/tbl2_fig9_tbl4/*.sh`
  - The main experiment of our paper. Pretraining the GPT-2.5B and GPT-8.3B model for each scheme. Additionally, the training script for non-lep case is included.
- Fig. 10: `ASPLOS23/fig10/*.sh`
  - Time check of each scheme. This script only checks the overall time of each scheme. To break down the time, we should follow an approach similar to the CPI stack; comment on the communication code for each communication.
- Fig. 11: `ASPLOS23/fig11/*.sh`
  - This script checks the averages and the cosine similarity of errors and intermediate activations.
- Fig. 12: `ASPLOS23/fig12/*.sh`
  - This script shows the maximum memory allocation of baseline, CB, and CB+LEP.
- Fig. 14: `ASPLOS23/fig14/*.md`
  - Instruction for compression and decompression throughput check of GPT-2.5B, 8.3B and 175B.

## Zero-Shot Task Running

To run the zero-shot task, split models should be merged into a single model.

Use `tools/ckpt_convert.py` to make the single model.

For example, if you use TP8, PP4 setting for GPT-2.5B pertaining, use a command like below to make the single model.

```shell
WORLD_SIZE=8 python tools/ckpt_convert.py --model-type GPT --tensor-model-parallel-size 8 --pipeline-model-parallel-size 4 --target-pipeline-model-parallel-size 1 --vocab-file ~/student1/gpt2-vocab.json --merge-file ~/student1/gpt2-merges.txt --num-layers 52 --hidden-size 1920 --num-attention-heads 24 --seq-length 1024 --max-position-embeddings 1024 --load ~/student1/GPT-2.5B-Baseline/ --save ~/student1/GPT-2.5B-Baseline/merge/ --experiment_name merge_baseline --tokenizer-type GPT2BPETokenizer --activations-checkpoint-method uniform --data-impl mmap --DDP-impl local -i ~/student1/GPT-2.5B-Baseline/iter_0300000/ -o ~/student1/GPT-2.5B-Baseline/merged/ -i_g 1 -p 1
```

Now, we prepared a single model for running zero-shot tasks.

We'll use `lm-evaluation-harness` (https://github.com/EleutherAI/lm-evaluation-harness) for running zero-shot tasks, so we need to convert the Megatron-LM checkpoint to HF (HuggingFace) checkpoint. Clone `transformers` GitHub and run the below code for converting.

```shell
python transformers-4.17.0/src/transformers/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py --config_file config.json {checkpoint_path}
```

Clone `lm-evaluation-harness` GitHub and run below code for running zero-shot tasks.

```shell
python main.py --model gpt2 --model_args pretrained={checkpoint_path} --device cuda:0 --tasks lambada,hellaswag,piqa,mathqa,winogrande,race
```

Now, you can get zero-shot task results.



## Dataset Preprocessing

Dataset preprocessing uses codes in `tools/` and `tools/openwebtext/`.

### Libraries to install

```
    pip install ftfy langdetect numpy torch pandas nltk sentencepiece boto3 tqdm regex bs4 newspaper3k htmlmin tldextract 
    git clone https://github.com/mattilyra/LSH
    cd LSH
    python setup.py install
```

### Download the dataset

1. Download the deduplicated URLs from [jcpeterson](https://mega.nz/#F!EZZD0YwJ!9_PlEQzdMVLaNdKv_ICNVQ!cc4RgQQZ)
2. Remove blacklisted URLs.

```
python blacklist_urls.py <path to the dowloaded deduplicated URLs> <filename for clean urls. e.g. clean_urls.txt>
```

3. Download the content from the clean urls with [openwebtext's utilities](https://github.com/eukaryote31/openwebtext/blob/master/download.py). 

4. Merge the contents into one loose json file with 1 json per newline of the format `{'text': text, 'url': unique_url}`. It is important for the url to be unique.

### Prepare the data for GPT training

1. Perform ftfy, english detection and remove documents with less than 128 tokens. This step can be sharded and run on shards.

```
python cleanup_dataset.py <input data file> <output cleaned data filename>
```

Additional cleanup (e.g. remove documents less than 512 characters or dataset specific cleaning like stories, realnews datasets) can be done using `cleanup_fix_dataset.py`. More details can be found by running `python cleanup_fix_dataset.py --help`.

2. Using LSH, find possible duplicates and store then in a file for later processing. The code supports saving and loading fingerprints for recurrent deduplications, and is also multithreaded for faster processing. More details are can be found by `python find_duplicate.py --help`.

```
python find_duplicates.py --inputs <pairlist list of input cleaned data files and keys, e.g. cc.json cc_id news.json news_id> --output <output possible duplicate urls filename>
```

3. Based on similarity measure defind inside function `is_similar` (default: 0.9), group urls that are similar. Basically, for each group, only one url we should keep and remove the rest.

```
python group_duplicate_urls.py <possible duplicate urls file> <output file containing similar urls>
```

4. Remove similar documents that were detected in the last step.

```
python remove_group_duplicates.py <file containing simialr documents> <cleaned data file> <outputfile containing deduplicate data>
```

5. Shuffle the dataset.

```
shuf <cleaned deduped data file> -o train_data.json
```

### Deduplicating ngrams

To deduplicate the downstream tasks (e.g. lambada, squad) from the training dataset, we run the following command.

```
python filter_ngrams.py --tasks <name of the task, e.g. lambada, squad> --dedup-dataset <training dataset to deduplicate> <json key> --output <output training dataset>
```

We use 13-grams by default for the deduplication. When we find a 13-gram match in a training document, we split the document into two pieces and remove the 13-gram along with 200 characters from the both side of the 13-gram. We also remove any splitted document with less than 200 characters or if a document got splitted more than 10 times. These parameters can be changed using corresponding arguments.

Only for the lambada task, we need to provide the path, `--lambada-path <path of the lambada test data>`.

Several other features (e.g. save and load dictionary) have been added, look at `python filter_ngrams.py --help` for details.
