# cd ~/FedNLP/experiments/centralized/transformer_exps/ && conda activate fednlp && sh run_text_classification.sh

GPU_NUM=2
echo $GPU_NUM
DATA_NAME=20news
CUDA_VISIBLE_DEVICES=$GPU_NUM python -m main_tc \
    --dataset ${DATA_NAME} \
    --data_file /data/cdq/fednlp_data/data_files/${DATA_NAME}_data.h5 \
    --partition_file /data/cdq/fednlp_data/partition_files/${DATA_NAME}_partition.h5 \
    --partition_method uniform \
    --model_type bert \
    --model_name bert-base-uncased  \
    --do_lower_case True \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --max_seq_length 16 \
    --learning_rate 5e-5 \
    --epochs 200 \
    --evaluate_during_training_steps 500 \
    --output_dir /tmp/${DATA_NAME}_fed/ \
    --n_gpu 1