# Remove adapter
[configuration.py](../../../../../cdq/.conda/envs/fednlp/lib/python3.7/site-packages/transformers/adapters/configuration.py) line 143 leave_out

[bert.py](../../../../../cdq/.conda/envs/fednlp/lib/python3.7/site-packages/transformers/adapters/models/bert.py) line 75 leave_out


# TC
[run_text_classification.sh](../../../../../cdq/FedNLP/experiments/distributed/transformer_exps/run_tc_exps/run_text_classification.sh)





## Layer Freeze
1. Check GPU memory `nvidia-smi`
2. Modify [gpu_mapping file](../../../../../cdq/FedNLP/experiments/distributed/transformer_exps/run_tc_exps/gpu_mapping.yaml)
3. change run command as [run_text_classification_freeze.sh](../../../../../cdq/FedNLP/experiments/distributed/transformer_exps/run_tc_exps/run_text_classification_freeze.sh)

## Adapter
1. Modify [base.py](../../../../../cdq/.conda/envs/fednlp/lib/python3.7/site-packages/transformers/adapters/heads/base.py) line 125
2. Modify [initializer.py](./initializer.py) line 46 && line 71-78
3. adapter size [modeling.py](../../../../../cdq/.conda/envs/fednlp/lib/python3.7/site-packages/transformers/adapters/modeling.py) line 81

## Cache
Modify rpi ubuntu file related with `# CDQ`

## Adaptive
Modify [tc_transformer_trainer.py](../../../../../cdq/FedNLP/training/tc_transformer_trainer.py) line 290 function freeze_model_parameters

## Round Number
Modify [tc_transformer_trainer.py](../../../../../cdq/FedNLP/training/tc_transformer_trainer.py) line 71

! Note: check line 80 whether random is activated

## Wandb Name
Modify [fedavg_main_tc.py](../../../../../cdq/FedNLP/experiments/distributed/transformer_exps/run_tc_exps/fedavg_main_tc.py) line 76

## Speedup 
### Aggregation
Modify [fedavg_main_tc.py](../../../../../cdq/FedNLP/experiments/distributed/transformer_exps/run_tc_exps/fedavg_main_tc.py): 
set args.is_mobile = 0

### Evaluation 
Modify [fed_trainer_transformer.py](../../../../../cdq/FedNLP/training/fed_trainer_transformer.py) line 31

# ST
[run_seq_tagging.sh](../../../../../cdq/FedNLP/experiments/distributed/transformer_exps/run_st_exps/run_seq_tagging.sh)

## Layer Freeze
1. Check GPU memory `nvidia-smi`
2. Modify [gpu_mapping file](../../../../../cdq/FedNLP/experiments/distributed/transformer_exps/run_tc_exps/gpu_mapping.yaml)
3. change [st_transformer_trainer.py](../../../../../cdq/FedNLP/training/st_transformer_trainer.py) line 254

## Adapter
1. Modify [base.py](../../../../../cdq/.conda/envs/fednlp/lib/python3.7/site-packages/transformers/adapters/heads/base.py) line 125
2. Modify [initializer.py](./initializer.py) line 46 && line 71-78

## Cache
Modify rpi ubuntu file related with `# CDQ`

## Adaptive
Modify [tc_transformer_trainer.py](../../../../../cdq/FedNLP/training/tc_transformer_trainer.py) line 290 function freeze_model_parameters

## Round Number
Modify [st_transformer_trainer.py](../../../../../cdq/FedNLP/training/st_transformer_trainer.py) line 71

## Wandb Name
Modify [fedavg_main_st.py](../../../../../cdq/FedNLP/experiments/distributed/transformer_exps/run_st_exps/fedavg_main_st.py) line 76

## Speedup Aggregation
Modify [fedavg_main_st.py](../../../../../cdq/FedNLP/experiments/distributed/transformer_exps/run_st_exps/fedavg_main_st.py): 
set args.is_mobile = 0