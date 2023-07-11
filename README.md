


# Efficient Federated Learning for Modern NLP

FedAdapter (old name: AdaFL) is an adapter-based efficient FedNLP framework for accelerating *federated learning* (FL) in *natural language processing* (NLP).

FedAdapter is built atop FedNLP beta (commit id: 27f3f97), the document file and instruction could be found in [README_FedNLP.md](./README_FedNLP.md)

# Installation
## Docker (Recommanded)
Install docker in your machine. Then run the following command to build the docker image.
```bash
docker pull caidongqi/adafl:1.0.1
docker run -it --gpus all --network host caidongqi/adafl:1.0.1 bash
```
We recommand using VSCode docker extension to develop in the docker container.
## Step-by-step installation
After `git clone`-ing this repository, please run the following command to install our dependencies.

```bash
conda create -n fednlp python=3.7
conda activate fednlp
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements.txt 
# some nail wheels, we install them manually
conda install mpi4py=3.0.3=py37hf046da1_1
conda install six==1.15.0


cd FedML; git submodule init; git submodule update; cd ../;

```


## System requirement
Our system is implemented on:

 `Linux Phoenix22 5.4.0-122-generic #138~18.04.1-Ubuntu SMP Fri Jun 24 14:14:03 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux`

Each simulated client needs ~2GB GPU memory, ~10GB RAM memory.

# Download data
```bash
cd data
# remember to modify the path in download_data.sh and download_partition.sh
bash download_data.sh
bash download_partition.sh
cd ..
```

# Main experiemnts
```bash
# TC tasks
conda activate fednlp
cd experiments/distributed/transformer_exps/run_tc_exps
python trial_error.py \
    --dataset 20news\
    --round -1 \
    --depth 0 \
    --width 8 \
    --time_threshold 60 \
    --max_round 3000 \
```
The log files would be placed into `experiments/distributed/transformer_exps/run_tc_exps/results/reproduce/20news-Trail-1-90`. Tmp model would be saved into `experiments/distributed/transformer_exps/run_tc_exps/tmp`. The final accuracy results would be stored in `experiments/distributed/transformer_exps/run_tc_exps/results/20news-depth-1-freq-90.log`.

# Reproduce main results in the paper
We process the result log via `exps_data/draw-performance-baseline.ipynb`
to get the final pictures in the manuscript.

Our experiment results could be downloaded from [Google drive](exps_data/download_data.sh).

You can also refer to our artifact evaluation instructions in [ae.pdf](./ae.pdf).
<!-- 

# Notes to be merged
# train locally
`experiments/distributed/transformer_exps/initializer.py` line 123
local_files_only=False

# Remove adapter
[configuration.py](../../cdq/.conda/envs/fednlp/lib/python3.7/site-packages/transformers/adapters/configuration.py) line 143 leave_out

[bert.py](../../cdq/.conda/envs/fednlp/lib/python3.7/site-packages/transformers/adapters/models/bert.py) line 75 leave_out


# TC
[run_text_classification.sh](experiments/distributed/transformer_exps/run_tc_exps/run_text_classification.sh)





## Layer Freeze
1. Check GPU memory `nvidia-smi`
2. Modify [gpu_mapping file](experiments/distributed/transformer_exps/run_tc_exps/gpu_mapping.yaml)
3. change run command as [run_text_classification_freeze.sh](experiments/distributed/transformer_exps/run_tc_exps/run_text_classification_freeze.sh)

## Adapter
1. Modify [base.py](../../cdq/.conda/envs/fednlp/lib/python3.7/site-packages/transformers/adapters/heads/base.py) line 125
2. Modify [initializer.py](./initializer.py) line 46 && line 71-78
3. adapter size [modeling.py](../../cdq/.conda/envs/fednlp/lib/python3.7/site-packages/transformers/adapters/modeling.py) line 81

## Cache
Modify rpi ubuntu file related with `# CDQ`

## Adaptive
Modify [tc_transformer_trainer.py](training/tc_transformer_trainer.py) line 290 function freeze_model_parameters

## Round Number
Modify [tc_transformer_trainer.py](training/tc_transformer_trainer.py) line 71

! Note: check line 80 whether random is activated

## Wandb Name
Modify [fedavg_main_tc.py](experiments/distributed/transformer_exps/run_tc_exps/fedavg_main_tc.py) line 76

## Speedup 
### Aggregation
Modify [fedavg_main_tc.py](experiments/distributed/transformer_exps/run_tc_exps/fedavg_main_tc.py): 
set args.is_mobile = 0

### Evaluation 
Modify [fed_trainer_transformer.py](training/fed_trainer_transformer.py) line 31

# ST
[run_seq_tagging.sh](experiments/distributed/transformer_exps/run_st_exps/run_seq_tagging.sh)

## Layer Freeze
1. Check GPU memory `nvidia-smi`
2. Modify [gpu_mapping file](experiments/distributed/transformer_exps/run_tc_exps/gpu_mapping.yaml)
3. change [st_transformer_trainer.py](training/st_transformer_trainer.py) line 254

## Adapter
1. Modify [base.py](../../cdq/.conda/envs/fednlp/lib/python3.7/site-packages/transformers/adapters/heads/base.py) line 125
2. Modify [initializer.py](./initializer.py) line 46 && line 71-78

## Cache
Modify rpi ubuntu file related with `# CDQ`

## Adaptive
Modify [tc_transformer_trainer.py](training/tc_transformer_trainer.py) line 290 function freeze_model_parameters

## Round Number
Modify [st_transformer_trainer.py](training/st_transformer_trainer.py) line 71

## Wandb Name
Modify [fedavg_main_st.py](experiments/distributed/transformer_exps/run_st_exps/fedavg_main_st.py) line 76

## Speedup Aggregation
Modify [fedavg_main_st.py](experiments/distributed/transformer_exps/run_st_exps/fedavg_main_st.py): 
set args.is_mobile = 0 -->

