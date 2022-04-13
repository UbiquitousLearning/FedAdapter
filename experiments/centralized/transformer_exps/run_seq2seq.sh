# cd ~/FedNLP/experiments/centralized/transformer_exps/ && conda activate fednlp && sh run_seq2seq.sh

DATA_NAME=wmt_de-en
CUDA_VISIBLE_DEVICES=4 python -m main_ss \
    --dataset ${DATA_NAME} \
    --data_file /data/cdq/fednlp_data/data_files/${DATA_NAME}_data.h5 \
    --partition_file /data/cdq/fednlp_data/partition_files/${DATA_NAME}_partition.h5 \
    --partition_method uniform \
    --model_type bart \
    --model_name facebook/bart-base  \
    --do_lower_case True \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --max_seq_length 256 \
    --learning_rate 3e-4 \
    --epochs 10 \
    --evaluate_during_training_steps 100 \
    --output_dir /tmp/${DATA_NAME}_fed/ \
    --n_gpu 1


# bash experiments/centralized/transformer_exps/run_seq2seq.sh > centralized_giga.log 2>&1 &
# tail -f centralized_giga.log