# RoSTER
This repo is an adapted clone of the https://github.com/yumeng5/RoSTER, which is based on the source code used for [**Distantly-Supervised Named Entity Recognition with Noise-Robust Learning and Language Model Augmented Self-Training**](https://arxiv.org/abs/2109.05003), published in EMNLP 2021.

## Requirements

At least one GPU with minimum 15GB of memory is required to run the training code. The evaluation can be run on a weaker GPU. 

Before running, you need to first install the required packages by typing following commands:

```
$ pip3 install -r requirements.txt
```

## Reproducing the Results

There are following bash scripts used to run experiments:

* baseline.sh - runs baseline RoSTER training
* baseline_eval.sh - runs evaluation on test set
* roberta_base.sh - runs RoBERTa with self augmentations and soft labels training 
* baseline_10_epochs.sh - runs baseline RoSTER training with 20 noise robust training epochs 

Bash scripts take two in-line arguments:
* `-d|--dataset`: The training dataset. If used in evaluation, then it defines which trained model to use. Any present dataset without suffix '_test' can be used here. 
* `-e|--eval_dataset`: The evaluation dataset. Currently, there three options avaible: lotr_test, conll_test, gold_lotr_test

## Slurm Cluster 

We have three custom slurm jobs templates: 

* main_and_eval.job - runs RoSTER training and evaluation
* main_and_eval_10e.job - runs RoSTER training and evaluation with 20 noise robust training epochs 
* eval.job - evaluate RoSTER model
* roberta.job - runs RoBERTa with self augmentations and soft labels training and evaluation
* eval_roberta.job runs RoBERTa with self augmentations and soft labels evaluation

Jobs take two in-line arguments:
* `-d|--dataset`: The training dataset. If used in evaluation, then it defines which trained model to use. Any present dataset without suffix '_test' can be used here. 
* `-e|--eval_dataset`: The evaluation dataset. Currently, there three options avaible: lotr_test, conll_test, gold_lotr_test
## Command Line Arguments

The meanings of the command line arguments will be displayed upon typing
```
python src/train.py -h
```
The following arguments are important and need to be set carefully:

* `train_batch_size`: The **effective** training batch size **after** gradient accumulation. Usually `32` is good for different datasets.
* `gradient_accumulation_steps`: Increase this value if your GPU cannot hold the training batch size (while keeping `train_batch_size` unchanged).
* `eval_batch_size`: This argument only affects the speed of the algorithm; use as large evaluation batch size as your GPUs can hold.
* `max_seq_length`: This argument controls the maximum length of sequence fed into the model (longer sequences will be truncated). Ideally, `max_seq_length` should be set to the length of the longest document (`max_seq_length` cannot be larger than `512` under RoBERTa architecture), but using larger `max_seq_length` also consumes more GPU memory, resulting in smaller batch size and longer training time. Therefore, you can trade model accuracy for faster training by reducing `max_seq_length`.
* `noise_train_epochs`, `ensemble_train_epochs`, `self_train_epochs`: They control how many epochs to train the model for noise-robust training, ensemble model training and self-training, respectively. Their default values will be a good starting point for most datasets, but you may increase them if your dataset is small (e.g., `Wikigold` dataset) and decrease them if your dataset is large (e.g., `OntoNotes` dataset).
* `q`, `tau`: Hyperparameters used for noise-robust training. Their default values will be a good starting point for most datasets, but you may use higher values if your dataset is more noisy and use lower values if your dataset is cleaner.
* `noise_train_update_interval`, `self_train_update_interval`: They control how often to update training label weights in noise-robust training and compute soft labels in soft-training, respectively. Their default values will be a good starting point for most datasets, but you may use smaller values (more frequent updates) if your dataset is small (e.g., `Wikigold` dataset).

## Citations

Please cite the following paper if you find the code helpful for your research.
```
@inproceedings{meng2021distantly,
  title={Distantly-Supervised Named Entity Recognition with Noise-Robust Learning and Language Model Augmented Self-Training},
  author={Meng, Yu and Zhang, Yunyi and Huang, Jiaxin and Wang, Xuan and Zhang, Yu and Ji, Heng and Han, Jiawei},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  year={2021},
}
```
