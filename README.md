## Introduction

**XLNet** is a new unsuperivsed language representation learning method based on a novel generalized permutation language modeling objective. Additionally, XLNet employs [Transformer-XL](https://arxiv.org/abs/1901.02860) as the backbone model, exhibiting excellent performance for language tasks involving long context. Overall, XLNet achieves state-of-the-art (SOTA) results on various downstream language tasks including question answering, natural language inference, sentiment analysis, and document ranking.

For a detailed description of technical details and experimental results, please refer to our paper:

​        [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)

​        Zhilin Yang\*, Zihang Dai\*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le 

​        (*: equal contribution) 

​        Preprint 2019




## Results

As of June 19, 2019, XLNet outperforms BERT on 20 tasks and achieves state-of-the-art results on 18 tasks. Below are some comparison between XLNet-Large and BERT-Large, which have similar model sizes:

### Results on Reading Comprehension

Model | [RACE accuracy](http://www.qizhexie.com/data/RACE_leaderboard.html) | SQuAD1.1 EM | SQuAD2.0 EM
--- | --- | --- | ---
BERT | 72.0 | 84.1 | 78.98
XLNet | **81.75** | **88.95** | **86.12**

We use SQuAD dev results in the table to exclude other factors such as using additional training data or other data augmentation techniques. See [SQuAD leaderboard](https://rajpurkar.github.io/SQuAD-explorer/) for test numbers.

### Results on Text Classification

Model | IMDB | Yelp-2 | Yelp-5 | DBpedia | Amazon-2 | Amazon-5
--- | --- | --- | --- | --- | --- | ---
BERT | 4.51 | 1.89 | 29.32 | 0.64 | 2.63 | 34.17
XLNet | **3.79** | **1.55** | **27.80** | **0.62** | **2.40** | **32.26**

The above numbers are error rates.

### Results on GLUE

Model | MNLI | QNLI | QQP | RTE | SST-2 | MRPC | CoLA | STS-B
--- | --- | --- | --- | --- | --- | --- | --- | ---
BERT | 86.6 | 92.3 | 91.3 | 70.4 | 93.2 | 88.0 | 60.6 | 90.0
XLNet | **89.8** | **93.9** | **91.8** | **83.8** | **95.6** | **89.2** | **63.6** | **91.8**

We use single-task dev results in the table to exclude other factors such as multi-task learning or using ensembles.

## Pre-trained models

### Released Models

As of <u>June 19, 2019</u>, the following model has been made available:
* **[`XLNet-Large, Cased`](https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip)**: 24-layer, 1024-hidden, 16-heads

Each .zip file contains three items:
*   A TensorFlow checkpoint (`xlnet_model.ckpt`) containing the pre-trained weights (which is actually 3 files).
*   A [Sentence Piece](https://github.com/google/sentencepiece) model (`spiece.model`) used for (de)tokenization.
*   A config file (`xlnet_config.json`) which specifies the hyperparameters of the model.


### Future Release Plan

We also plan to continuously release more pretrained models under different settings, including:
* **Base models (very soon)**: We will release an XLNet-Base by the end of June, 2019.
* **Uncased models (soon)** : For now, cased XLNet-Large is generally better than uncased XLNet-Large. We are still investigating this observation. When a conclusion is reached, we will release the uncased models.
* A pretrained model that is **finetuned on Wikipedia**. This can be used for tasks with Wikipedia text such as SQuAD and HotpotQA.
* Pretrained models with other hyperparameter configurations, targeting specific downstream tasks.
* Pretrained models that benefit from new techniques.

### Subscribing to XLNet on Google Groups

To receive notifications about updates, announcements and new releases, we recommend subscribing to the XLNet on [Google Groups](https://groups.google.com/forum/#!forum/xlnet).



## Fine-tuning with XLNet

As of <u>June 19, 2019</u>, this code base has been tested with TensorFlow 1.13.1 under Python2.

### Memory Issue during Finetuning

- Most of the SOTA results in our paper were produced on TPUs, which generally have more RAM than common GPUs. As a result, it is currently very difficult (costly) to re-produce most of the `XLNet-Large` SOTA results in the paper using GPUs with 12GB - 16GB of RAM, because a 16GB GPU is only able to hold a <u>single sequence with length 512</u> for `XLNet-Large`. Therefore, a large number (ranging from 32 to 128, equal to `batch_size`) of GPUs are required to reproduce many results in the paper.
- We are experimenting with gradient accumulation to potentially relieve the memory burden, which could be included in a near-future update.

Given the memory issue mentioned above, using the default finetuning scripts (`run_classifier.py` and `run_squad.py`), we benchmarked the maximum batch size on a single **16GB** GPU with TensorFlow **1.13.1**:

| System        | Seq Length | Max Batch Size |
| ------------- | ---------- | -------------- |
| `XLNet-Base`  | 64         | 120            |
| ...           | 128        | 56             |
| ...           | 256        | 24             |
| ...           | 512        | 8              |
| `XLNet-Large` | 64         | 16             |
| ...           | 128        | 8              |
| ...           | 256        | 2              |
| ...           | 512        | 1              |

In most cases, it is possible to reduce the batch size `train_batch_size` or the maximum sequence length `max_seq_length` to fit in given hardware. The decrease in performance depends on the task and the available resources.


### Text Classification/Regression

The code used to perform classification/regression finetuning is in `run_classifier.py`. It also contains examples for standard one-document classification, one-document regression, and document pair classification. Here, we provide two concrete examples of how `run_classifier.py` can be used.

From here on, we assume XLNet-Large and XLNet-base has been downloaded to `$LARGE_DIR` and `$BASE_DIR` respectively.


#### (1) STS-B: sentence pair relevance regression (with GPUs)

- Download the [GLUE data](https://gluebenchmark.com/tasks) by running [this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) and unpack it to some directory `$GLUE_DIR`.

- Perform **multi-GPU** (4 V100 GPUs) finetuning with XLNet-Large by running

  ```shell
  CUDA_VISIBLE_DEVICES=0,1,2,3 python run_classifier.py \
    --do_train=True \
    --do_eval=False \
    --task_name=sts-b \
    --data_dir=${GLUE_DIR}/STS-B \
    --output_dir=proc_data/sts-b \
    --model_dir=exp/sts-b \
    --uncased=False \
    --spiece_model_file=${LARGE_DIR}/spiece.model \
    --model_config_path=${LARGE_DIR}/model_config.json \
    --init_checkpoint=${LARGE_DIR}/xlnet_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=8 \
    --num_hosts=1 \
    --num_core_per_host=4 \
    --learning_rate=5e-5 \
    --train_steps=1200 \
    --warmup_steps=120 \
    --save_steps=600 \
    --is_regression=True
  ```

- Evaluate the finetuning results with a single GPU by

  ```shell
  CUDA_VISIBLE_DEVICES=0 python run_classifier.py \
    --do_train=False \
    --do_eval=True \
    --task_name=sts-b \
    --data_dir=${GLUE_DIR}/STS-B \
    --output_dir=proc_data/sts-b \
    --model_dir=exp/sts-b \
    --uncased=False \
    --spiece_model_file=${LARGE_DIR}/spiece.model \
    --model_config_path=${LARGE_DIR}/model_config.json \
    --max_seq_length=128 \
    --eval_batch_size=8 \
    --num_hosts=1 \
    --num_core_per_host=1 \
    --eval_all_ckpt=True \
    --is_regression=True

  # Expected performance: "eval_pearsonr 0.916+ "
  ```

**Notes**:

- In the context of GPU training, `num_core_per_host` denotes the number of GPUs to use.
- In the multi-GPU setting, `train_batch_size` refers to the <u>per-GPU batch size</u>.
- `eval_all_ckpt` allows one to evaluate all saved checkpoints (save frequency is controlled by `save_steps`) after training finishes and choose the best model based on dev performance.
- `data_dir` and `output_dir` refer to the directories of the "raw data" and "preprocessed tfrecords" respectively, while `model_dir` is the working directory for saving checkpoints and tensorflow events.
- To try out <u>XLNet-base</u>, one can simply set `--train_batch_size=32` and `--num_core_per_host=1`, along with according changes in `init_checkpoint` and `model_config_path`.
- For GPUs with smaller RAM, please proportionally decrease the `train_batch_size` and increase `num_core_per_host` to use the same training setting.
- **Important**: we separate the training and evaluation into "two phases", as using multi GPUs to perform evaluation is tricky (one has to correctly separate the data across GPUs). To ensure correctness, we only support single-GPU evaluation for now.


#### (2) IMDB: movie review sentiment classification (with TPU V3-8)

- Download and unpack the IMDB dataset by running

  ```shell
  wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
  tar zxvf aclImdb_v1.tar.gz
  ```

- Launch a Google cloud TPU V3-8 instance (see the [Google Cloud TPU tutorial](https://cloud.google.com/tpu/docs/tutorials/mnist) for how to set up Cloud TPUs).

- Set up your Google storage bucket path `$GS_ROOT` and move the IMDB dataset and pretrained checkpoint into your Google storage.

- Perform TPU finetuning with XLNet-Large by running

  ```shell
  python run_classifier.py \
    --use_tpu=True \
    --tpu=${TPU_NAME} \
    --do_train=True \
    --do_eval=True \
    --eval_all_ckpt=True \
    --task_name=imdb \
    --data_dir=${IMDB_DIR} \
    --output_dir=${GS_ROOT}/proc_data/imdb \
    --model_dir=${GS_ROOT}/exp/imdb \
    --uncased=False \
    --spiece_model_file=${LARGE_DIR}/spiece.model \
    --model_config_path=${GS_ROOT}/${LARGE_DIR}/model_config.json \
    --init_checkpoint=${GS_ROOT}/${LARGE_DIR}/xlnet_model.ckpt \
    --max_seq_length=512 \
    --train_batch_size=32 \
    --eval_batch_size=8 \
    --num_hosts=1 \
    --num_core_per_host=8 \
    --learning_rate=2e-5 \
    --train_steps=4000 \
    --warmup_steps=500 \
    --save_steps=500 \
    --iterations=500

  # Expected performance: "eval_accuracy 0.962+ "
  ```

**Notes**:

- To obtain the SOTA on the IMDB dataset, using sequence length 512 is **necessary**. Therefore, we show how this can be done with a TPU V3-8.
- Alternatively, one can use a sequence length smaller than 512, a smaller batch size, or switch to XLNet-base to train on GPUs. But performance drop is expected.
- Notice that the `data_dir` and `spiece_model_file` both use a local path rather than a Google Storage path. The reason is that data preprocessing is actually performed locally. Hence, using local paths leads to a faster preprocessing speed.

### SQuAD2.0

The code for the SQuAD dataset is included in `run_squad.py`.

To run the code:

(1) Download the SQuAD2.0 dataset into `$SQUAD_DIR` by:

```shell
mkdir -p ${SQUAD_DIR} && cd ${SQUAD_DIR}
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
```

(2) Perform data preprocessing using the script `scripts/prepro_squad.sh`.

- This will take quite some time in order to accurately map character positions (raw data) to sentence piece positions (used for training).

- For faster parallel preprocessing, please refer to the flags `--num_proc` and `--proc_id` in `run_squad.py`.

(3) Perform training and evaluation.

For the best performance, XLNet-Large uses <u>sequence length 512</u> and <u>batch size 48</u> for training.

- As a result, reproducing the best result with GPUs is quite difficult.

- For training with one TPU v3-8, one can simply run the script `scripts/tpu_squad_large.sh` after both the TPU and Google storage have been setup.
- `run_squad.py` will automatically perform threshold searching on the dev set of squad and output the score. With `scripts/tpu_squad_large.sh`, the expected F1 score should be around 88.6 (median of our multiple runs).

Alternatively, one can use XLNet-Base with GPUs (e.g. three V100). One set of reasonable hyper-parameters can be found in the script `scripts/gpu_squad_base.sh`.


### RACE reading comprehension

The code for the reading comprehension task [RACE](https://www.cs.cmu.edu/~glai1/data/race/) is included in `run_race.py`.

- Notably, the average length of the passages in RACE is over 300 tokens (not peices), which is <u>significantly longer</u> than other popular reading comprehension datasets such as SQuAD.
- Also, many questions can be very difficult and requires complex reasoning for machines to solve (see [one example here](misc/race_example.md)).


To run the code:

(1) Download the RACE dataset from the [official website](https://www.cs.cmu.edu/~glai1/data/race/) and unpack the raw data to `$RACE_DIR`.

(2) Perform training and evaluation:

- The SOTA performance (accuracy 81.75) of RACE is produced using XLNet-Large with sequence length 512 and batch size 32, which requires a large TPU v3-32 in the pod setting. Please refer to the script `script/tpu_race_large_bsz32.sh` for this setting.
- Using XLNet-Large with sequence length 512 and batch size 8 on a TPU v3-8 can give you an accuracy of around 80.3 (see `script/tpu_race_large_bsz8.sh`).



## Custom Usage of XLNet

For finetuning, it is likely that you will be able to modify existing files such as `run_classifier.py`, `run_squad.py` and `run_race.py` for your task at hand. However, we also provide an abstraction of XLNet to enable more flexible usage. Below is an example:

```python
import xlnet

# some code omitted here...
# initialize FLAGS
# initialize instances of tf.Tensor, including input_ids, seg_ids, and input_mask

# XLNetConfig contains hyperparameters that are specific to a model checkpoint.
xlnet_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)

# RunConfig contains hyperparameters that could be different between pretraining and finetuning.
run_config = xlnet.create_run_config(is_training=True, is_finetune=True, FLAGS=FLAGS)

# Construct an XLNet model
xlnet_model = xlnet.XLNetModel(
    xlnet_config=xlnet_config,
    run_config=run_config,
    input_ids=input_ids,
    seg_ids=seg_ids,
    input_mask=input_mask)

# Get a summary of the sequence using the last hidden state
summary = xlnet_model.get_pooled_out(summary_type="last")

# Get a sequence output
seq_out = xlnet_model.get_sequence_output()

# build your applications based on `summary` or `seq_out`
```



## Pretraining with XLNet

Refer to `train.py` for pretraining on TPUs and `train_gpu.py` for pretraining on GPUs. First we need to preprocess the text data into tfrecords.

```shell
python data_utils.py \
	--bsz_per_host=32 \
	--num_core_per_host=16 \
	--seq_len=512 \
	--reuse_len=256 \
	--input_glob=*.txt \
	--save_dir=${SAVE_DIR} \
	--num_passes=20 \
	--bi_data=True \
	--sp_path=spiece.model \
	--mask_alpht=6 \
	--mask_beta=1 \
	--num_predict=85
```

where `input_glob` defines all input text files, `save_dir` is the output directory for tfrecords, and `sp_path` is a [Sentence Piece](https://github.com/google/sentencepiece) model. Here is our script to train the Sentence Piece model

```bash
spm_train \
	--input=$INPUT \
	--model_prefix=sp10m.cased.v3 \
	--vocab_size=32000 \
	--character_coverage=0.99995 \
	--model_type=unigram \
	--control_symbols=<cls>,<sep>,<pad>,<mask>,<eod> \
	--user_defined_symbols=<eop>,.,(,),",-,–,£,€ \
	--shuffle_input_sentence \
	--input_sentence_size=10000000
```

Special symbols are used, including `control_symbols` and `user_defined_symbols`. We use `<eop>` and `<eod>` to denote End of Paragraph and End of Document respectively.

The input text files to `data_utils.py` must use the following format:
* Each line is a sentence.
* An empty line means End of Document.
* (Optional) If one also wants to model paragraph structures, `<eop>` can be inserted at the end of certain lines (without any space) to indicate that the corresponding sentence ends a paragraph.

For example, the text input file could be:
```
This is the first sentence.
This is the second sentence and also the end of the paragraph.<eop>
Another paragraph.

Another document starts here.
```

After preprocessing, we are ready to pretrain an XLNet. Below are the hyperparameters used for pretraining XLNet-Large:

```shell
python train.py
  --corpus_info_path=$DATA/corpus_info.json \
  --record_info_dir=$DATA/tfrecords \
  --train_batch_size=2048 \
  --seq_len=512 \
  --reuse_len=256 \
  --perm_size=256 \
  --n_layer=24 \
  --d_model=1024 \
  --d_embed=1024 \
  --n_head=16 \
  --d_head=64 \
  --d_inner=4096 \
  --untie_r=True \
  --mask_alpha=6 \
  --mask_beta=1 \
  --num_predict=85
```

where we only list the most important flags and the other flags could be adjusted based on specific use cases.

