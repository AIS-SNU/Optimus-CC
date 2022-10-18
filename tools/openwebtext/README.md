# Datasets

We have used Wikipedia, Realnews, Openwebtext, CC-stories for GPT-2 pretraining. We do not host any data here. Wikipedia[^1], Realnews[^2] and Openwebtext[^3]  is available at [huggingface](https://huggingface.co/datasets).  CC-stories datasets is not publicly available. We've got CC-stories from the author of original paper[^4] . CC-stories needed to be converted to json files, since the data was provided in the text file format. We applied six steps for each of data. The six steps are as follows.  



1) cleaning up data

   - We used  `cleanup_dataset.py`  for Wikipedia and Openwebtext. As [NVIDIA](https://github.com/NVIDIA/Megatron-LM/tree/main/tools/openwebtext) [^5] guided in their repository, we used `cleanup_fix_dataset.py`  for Realnews and CC-stories. For realnews and CC-stories data, some lines of `cleanup_fix_dataset.py` need to be enabled. 

     - ```
       		if "general_cleaning" in args.tasks:
                   cleaned_text = re.sub(r"  +|\b\n+ |\b\n+", " ", text)
                   #cleaned_text = re.sub(r"\n\n+", "\n\n", text) # used this for Gutenberg dataset
                   #cleaned_text = re.sub(r"\n", "\n\n", text) # Used this for realnews
       
                   # stories datasets
                   #cleaned_text = re.sub(r" \'", "'", text)
                   #cleaned_text = re.sub(r" \!", "!", cleaned_text)
                   #cleaned_text = re.sub(r" \.", ".", cleaned_text)
                   #cleaned_text = re.sub(r" \?", "?", cleaned_text)
                   #cleaned_text = re.sub(r" - ", "-", cleaned_text)
                   ##cleaned_text = re.sub(r"\" ", "\"", cleaned_text)
                   #cleaned_text = re.sub(r" @ ", "@", cleaned_text)
       
                   output['general_cleaning'] = True
                   return output, cleaned_text, document, False
       ```

       

2) creating a unique key for each documents.

   - we need unique keys to group duplicate documents. So we attached unique keys for each document by using `generate_key.py`
     - ``` python generate_key.py <input data file> <output data filename>```

3) grouping duplicate documents

   - We first grouped duplicate documents which have jaccard similarity greater than 0.5.

4) removing duplicate documents

   - We removed documents which have jaccard similarity greater than 0.7.

5) shuffling documents

6) deduplicating LAMBADA tasks



After applying above six steps for each of data, we concatenated all resulting json files and ran `preprocess_data.py`. Finally, we could get binary files which is used to pretrain GPT2. Following documents is the same as the one described by [NVIDIA github](https://github.com/NVIDIA/Megatron-LM/tree/main/tools/openwebtext).



# Libraries to install

```
    pip install ftfy langdetect numpy torch pandas nltk sentencepiece boto3 tqdm regex bs4 newspaper3k htmlmin tldextract 
    git clone https://github.com/mattilyra/LSH
    cd LSH
    python setup.py install
```



# Prepare the data for GPT training:

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

# Deduplicating ngrams

To deduplicate the downstream tasks (e.g. lambada, squad) from the training dataset, we run the following command.

```
python filter_ngrams.py --tasks <name of the task, e.g. lambada, squad> --dedup-dataset <training dataset to deduplicate> <json key> --output <output training dataset>
```

We use 13-grams by default for the deduplication. When we find a 13-gram match in a training document, we split the document into two pieces and remove the 13-gram along with 200 characters from the both side of the 13-gram. We also remove any splitted document with less than 200 characters or if a document got splitted more than 10 times. These parameters can be changed using corresponding arguments.

Only for the lambada task, we need to provide the path, `--lambada-path <path of the lambada test data>`.

Several other features (e.g. save and load dictionary) have been added, look at `python filter_ngrams.py --help` for details.

[^1]: Megatron-LM. URL: https : / / github. com / NVIDIA / Megatron - LM / tree / main / megatron/.
[^2]: Jacob Devlin et al. “Bert: Pre-training of Deep Bidirectional Transformers for Language Understanding”. In: arXiv preprint arXiv:1810.04805 (2018).
[^3]: Rowan Zellers et al. “Defending Against Neural Fake News”. In: NeurIPS. 2019.
[^4]: Alec Radford et al. “Better Language Models and Their Implications”. In: OpenAI blog (2019).
[^5]: Trieu H Trinh and Quoc V Le. “A Simple Method for Commonsense Reason- ing”. In: arXiv preprint arXiv:1806.02847 (2018).



