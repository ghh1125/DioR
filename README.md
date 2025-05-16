# DioR: Adaptive Cognitive Detection and Contextual Retrieval Optimization for Dynamic Retrieval-Augmented Generation

---

🚀 **Exciting News**! 

✨ We are **thrilled** to announce that our paper, titled **"DioR: Adaptive Cognitive Detection and Contextual Retrieval Optimization for Dynamic Retrieval-Augmented Generation"**, has been **accepted** for presentation at **ACL 2025 Main**! 🎉
🎉
---

# Code

You should organize them in the following format.

```
DioR
    ├── Sgpt_file  
    ├── data  \\  Create and import according to the following command
    ├── paraphrase-MiniLM-L6-v2   \\ SBERT model
    ├── SGPT   \\ SGPT model
    ├── config
    ├── src
    ├── result \\ Create and import according to the following command
    ├── prep_esastic.py
    ├── rnn_hallucination_model_0.pth
    ├── train.sh
```

## Install environment

```bash
conda create -n DioR python=3.9
conda activate DioR
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Download LLaMA2-7B

```
https://huggingface.co/meta-llama/Llama-2-7b
```


### Download SBERT,SGPT

```
./huggingface/paraphrase-MiniLM-L6-v2
```
and
```
./hugggingface/SGPT
```

### Build BM25 index


```bash
mkdir -p data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
pushd data/dpr
gzip -d psgs_w100.tsv.gz
popd
```

```bash
cd data
wget -O elasticsearch-7.17.9.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz  # download Elasticsearch
tar zxvf elasticsearch-7.17.9.tar.gz
rm elasticsearch-7.17.9.tar.gz 
cd elasticsearch-7.17.9
nohup bin/elasticsearch &  # run Elasticsearch in background
cd ../..
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki  # build index
```

There is an issue with psgsw_100.tsv (empty data exists), please delete it and save it as psgsw_100_fixed.tsv
```
python ./scr/fixed.py
```
### Build SGPT index (in file ./SGPT/encode_result/)

```
python ./Sgpt_file/sgpt_file.py
```

### Download Dataset

For 2WikiMultihopQA:

Download the [2WikiMultihop](https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1) dataset from its repository <https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1>. Unzip it and move the folder to `data/2wikimultihopqa`.

For StrategyQA:

```bash
wget -O data/strategyqa_dataset.zip https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip
mkdir -p data/strategyqa
unzip data/strategyqa_dataset.zip -d data/strategyqa
rm data/strategyqa_dataset.zip 
```

For HotpotQA:

```bash
mkdir -p data/hotpotqa
wget -O data/hotpotqa/hotpotqa-dev.json http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

For IIRC:

```bash
wget -O data/iirc.tgz https://iirc-dataset.s3.us-west-2.amazonaws.com/iirc_train_dev.tgz
tar -xzvf data/iirc.tgz
mv iirc_train_dev/ data/iirc
rm data/iirc.tgz
```



### Run

```
bash train.sh
```

### Evaluate


```bash
python ./src/evaluate1.py --dir path_to_folder(result/[result path and name]])
```


# Citation

```
@article{guo2025dior,
  title={DioR: Adaptive Cognitive Detection and Contextual Retrieval Optimization for Dynamic Retrieval-Augmented Generation},
  author={Guo, Hanghui and Zhu, Jia and Di, Shimin and Shi, Weijie and Chen, Zhangze and Xu, Jiajie},
  journal={arXiv preprint arXiv:2504.10198},
  year={2025}
}
```
