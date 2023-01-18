# AutoFedNLP: An efficient FedNLP framework

AutoFedNLP is an adapter-based efficient FedNLP framework for accelerating *federated learning* (FL) in *natural language processing* (NLP).

AutoFedNLP is built atop FedNLP beta (commit id: 27f3f97), the document file and instruction could be found in [README_FedNLP.md](./README_FedNLP.md)

# Step-by-step installation
After `git clone`-ing this repository, please run the following command to install our dependencies.

```bash
conda create -n fednlp python=3.7
conda activate fednlp
# conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -n fednlp
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt 
pip uninstall transformers
pip install -e transformers/
pip install adapter-transformers==2.3.0a0

cd FedML; git submodule init; git submodule update; cd ../;

```


## System requirement
Our system is implemented on:

 `Linux Phoenix22 5.4.0-122-generic #138~18.04.1-Ubuntu SMP Fri Jun 24 14:14:03 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux`

Each simulated client needs ~2GB GPU memory, ~10GB RAM memory.

# Reproduce main results in the paper
```bash
# TC tasks
conda activate fednlp
cd experiments/transformer_exps/run_tc_exps
python trial_error_w&d.py \
    --dataset agnews \
    --round -1 \
    --depth 1 \
    --width 8 \
    --time_threshold 90 \
    --max_round 3000 \
```
The log files would be placed into `experiments/distributed/transformer_exps/run_tc_exps/results/reproduce/20news-Trail-1-90`. Tmp model would be saved into `/home/cdq/FedNLP/experiments/distributed/transformer_exps/run_tc_exps/tmp`. The final accuracy results would be stored in `experiments/distributed/transformer_exps/run_tc_exps/results/20news-depth-1-freq-90.log`.

We process the result log via `exps_data/draw-performance-baseline.ipynb`
to get the final pictures in the manuscript.

Our experiment results could be downloaded from [Google drive](exps_data/download_data.sh).
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

## Citation
Please cite our paper if it helps your research.
```bib
@inproceedings{aufofednlp2022,
  title={AutoFedNLP: An efficient FedNLP framework},
  author={Dongqi Cai, Yaozong Wu, Shangguang Wang, Felix Xiaozhu Lin and Mengwei Xu},
  year={2022},
  booktitle={arXiv cs.LG 2205.10162},
  url={https://arxiv.org/abs/2205.10162}
}
```