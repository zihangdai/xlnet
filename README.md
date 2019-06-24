## WORKS ON COLAB TPU

## Introduction

**XLNet** is a new unsuperivsed language representation learning method based on a novel generalized permutation language modeling objective. Additionally, XLNet employs [Transformer-XL](https://arxiv.org/abs/1901.02860) as the backbone model, exhibiting excellent performance for language tasks involving long context. Overall, XLNet achieves state-of-the-art (SOTA) results on various downstream language tasks including question answering, natural language inference, sentiment analysis, and document ranking.

For a detailed description of technical details and experimental results, please refer to the paper:

​        [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)

​        Zhilin Yang\*, Zihang Dai\*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le 

​        (*: equal contribution) 

​        Preprint 2019


## Colab Notebook:
A colab notebook available here:  has been integrated with this repo to allow for quick prototyping on the Google TPUs available at colab.


## Pre-trained models

### Released Models

As of <u>June 19, 2019</u>, the following model has been made available:
* **[`XLNet-Large, Cased`](https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip)**: 24-layer, 1024-hidden, 16-heads

Each .zip file contains three items:
*   A TensorFlow checkpoint (`xlnet_model.ckpt`) containing the pre-trained weights (which is actually 3 files).
*   A [Sentence Piece](https://github.com/google/sentencepiece) model (`spiece.model`) used for (de)tokenization.
*   A config file (`xlnet_config.json`) which specifies the hyperparameters of the model.


**Notes**:

- In the context of GPU training, `num_core_per_host` denotes the number of GPUs to use.
- In the multi-GPU setting, `train_batch_size` refers to the <u>per-GPU batch size</u>.
- `eval_all_ckpt` allows one to evaluate all saved checkpoints (save frequency is controlled by `save_steps`) after training finishes and choose the best model based on dev performance.
- `data_dir` and `output_dir` refer to the directories of the "raw data" and "preprocessed tfrecords" respectively, while `model_dir` is the working directory for saving checkpoints and tensorflow events.
- To try out <u>XLNet-base</u>, one can simply set `--train_batch_size=32` and `--num_core_per_host=1`, along with according changes in `init_checkpoint` and `model_config_path`.
- For GPUs with smaller RAM, please proportionally decrease the `train_batch_size` and increase `num_core_per_host` to use the same training setting.

