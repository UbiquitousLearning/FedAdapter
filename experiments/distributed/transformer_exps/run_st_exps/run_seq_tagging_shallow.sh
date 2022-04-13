# cd /home/cdq/FedNLP/experiments/distributed/transformer_exps/run_st_exps && conda activate fednlp && sh run_seq_tagging.sh FedAvg niid_label_clients=30_alpha=1.0 0.1 1 0.5 20000 5
FL_ALG=$1
PARTITION_METHOD=$2
C_LR=$3
S_LR=$4
MU=$5
ROUND=$6
WORKER_NUM=$7
LAYERS=$8
DEPTH=$9
TIME=$10

LOG_FILE="fedavg_transformer_st.log"

CI=0

DATA_DIR=/data/cdq/fednlp_data/
DATA_NAME=onto  # dataname
PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
python -m fedavg_main_st \
  --gpu_mapping_file "gpu_mapping.yaml" \
  --gpu_mapping_key cdq-${WORKER_NUM} \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --ci $CI \
  --dataset "${DATA_NAME}" \
  --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
  --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
  --partition_method $PARTITION_METHOD \
  --fl_algorithm $FL_ALG \
  --model_type bert \
  --model_name bert-base-uncased \
  --do_lower_case True \
  --train_batch_size 4 \
  --eval_batch_size 4 \
  --max_seq_length 128 \
  --lr $C_LR \
  --server_lr $S_LR \
  --fedprox_mu $MU \
  --epochs 1 \
  --output_dir "./tmp/fedavg_${DATA_NAME}_output_shallow-${DEPTH}-${TIME}/" \
  --fp16 \
  --freeze_layers $LAYERS
  # 2> ${LOG_FILE} &

# sh run_seq_tagging.sh FedAvg "niid_cluster_clients=100_alpha=5.0" 1e-5 0.1 0.5 30

# sh run_seq_tagging.sh FedProx "niid_cluster_clients=100_alpha=5.0" 1e-5 0.1 0.5 30

# sh run_seq_tagging.sh FedOPT "niid_cluster_clients=100_alpha=5.0" 1e-5 0.1 0.5 30