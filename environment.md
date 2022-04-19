# FedNLP: A Research Platform for Federated Learning in Natural Language Processing
```
git clone https://github.com/FedML-AI/FedNLP.git
```

## Installation
<!-- http://doc.fedml.ai/#/installation -->
After `git clone`-ing this repository, please run the following command to install our dependencies.

```bash
conda create -n fednlp python=3.7
conda activate fednlp
# conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -n fednlp
# 这里装1.8的相关版本
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt 
pip uninstall transformers
pip install -e transformers/
cd FedML; git submodule init; git submodule update; cd ../;


## Data Preparation
In order to set up correct data to support federated learning, we provide some processed data files and partition files. Users can download them for further training conveniently.

If users want to set up their own dataset, they can refer the scripts under `data/raw_data_loader`. We already offer a bunch of examples, just follow one of them to prepare your owned data!
### download our processed files from Amazon S3.
Dwnload files for each dataset using these two scripts `data/download_data.sh` and `data/download_partition.sh`.

We provide two files for each dataset: data files are saved in  **data_files**, and partition files are in directory **partiton_files**. You need to put the downloaded `data_files` and `partition_files` in the `data` folder here. Simply put, we will have `data/data_files/*_data.h5` and `data/partition_files/*_partition.h5` in the end.

```
# 文件路径修改一下
sh ~/FedNLP/data/download_data.sh
sh ~/FedNLP/data/download_partition.sh
```
## Experiments for Centralized Learning (Sanity Check)

### Transformer-based models 

First, please use this command to test the dependencies.
```bash
# Test the environment for the fed_transformers
python -m model.fed_transformers.test
```

Run Text Classification model with `distilbert`:

```bash 
DATA_NAME=20news
CUDA_VISIBLE_DEVICES=1 python -m experiments.centralized.transformer_exps.main_tc \
    --dataset ${DATA_NAME} \
    --data_file ~/fednlp_data/data_files/${DATA_NAME}_data.h5 \
    --partition_file ~/fednlp_data/partition_files/${DATA_NAME}_partition.h5 \
    --partition_method niid_label_clients=100_alpha=1.0 \
    --model_type distilbert \
    --model_name distilbert-base-uncased  \
    --do_lower_case True \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --max_seq_length 256 \
    --learning_rate 1e-1 \
    --epochs 20 \
    --evaluate_during_training_steps 500 \
    --output_dir /tmp/${DATA_NAME}_fed/ \
    --n_gpu 1
```


## Citation
Please cite our FedNLP and FedML paper if it helps your research.
```bib
@inproceedings{fednlp2021,
  title={FedNLP: A Research Platform for Federated Learning in Natural Language Processing},
  author={Bill Yuchen Lin and Chaoyang He and ZiHang Zeng and Hulin Wang and Yufen Huang and M. Soltanolkotabi and Xiang Ren and S. Avestimehr},
  year={2021},
  booktitle={arXiv cs.CL 2104.08815},
  url={https://arxiv.org/abs/2104.08815}
}
```

```
@article{chaoyanghe2020fedml,
  Author = {He, Chaoyang and Li, Songze and So, Jinhyun and Zhang, Mi and Wang, Hongyi and Wang, Xiaoyang and Vepakomma, Praneeth and Singh, Abhishek and Qiu, Hang and Shen, Li and Zhao, Peilin and Kang, Yan and Liu, Yang and Raskar, Ramesh and Yang, Qiang and Annavaram, Murali and Avestimehr, Salman},
  Journal = {arXiv preprint arXiv:2007.13518},
  Title = {FedML: A Research Library and Benchmark for Federated Machine Learning},
  Year = {2020}
}
```

 
