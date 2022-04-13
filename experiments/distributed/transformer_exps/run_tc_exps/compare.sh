# Origin
# Modify wandb name!!
# sh run_text_classification.sh FedAvg "uniform" 0.1 1 3000 5 > ./results/New-BERT/client5/fednlp_tc_origin.log 2>&1 &

# Freeze
# Modify wandb name!!
sh run_text_classification_freeze.sh FedAvg "uniform" 0.1 1 3000 5 e,0,1,2,3,4,5,6,7,8,9,10 > ./results/New-BERT/client5/fednlp_tc_freeze-0-10.log 2>&1 &

sh run_text_classification_freeze.sh FedAvg "uniform" 0.1 1 3000 5 e,0,1,2,3,4,5,6,7,8,9 > ./results/New-BERT/client5/fednlp_tc_freeze-0-9.log 2>&1 &

# A-Freeze
# Modify wandb name!!
sh run_text_classification_freeze.sh FedAvg "uniform" 0.1 1 3000 5 e > ./results/New-BERT/client5/fednlp_tc_A-Freeze-e.log 2>&1 &

sh run_text_classification_freeze.sh FedAvg "uniform" 0.1 1 3000 5 e,0 > ./results/New-BERT/client5/fednlp_tc_A-Freeze-0.log 2>&1 &

sh run_text_classification_freeze.sh FedAvg "uniform" 0.1 1 3000 5 e,0,1,2 > ./results/New-BERT/client5/fednlp_tc_A-Freeze-0-2.log 2>&1 &

sh run_text_classification_freeze.sh FedAvg "uniform" 0.1 1 3000 5 e,0,1,2,3,4,5,6 > ./results/New-BERT/client5/fednlp_tc_A-Freeze-0-6.log 2>&1 &

sh run_text_classification_freeze.sh FedAvg "uniform" 0.1 1 3000 5 e,0,1,2,3,4,5,6,7,8,9 > ./results/New-BERT/client5/fednlp_tc_A-Freeze-0-9.log 2>&1 &

sh run_text_classification_freeze.sh FedAvg "uniform" 0.1 1 3000 5 e,0,1,2,3,4,5,6,7,8,9,10 > ./results/New-BERT/client5/fednlp_tc_A-Freeze-0-10.log 2>&1 &

sh run_text_classification_freeze.sh FedAvg "uniform" 0.1 1 3000 5 e,0,1,2,3,4,5,6,7,8,10,11 > ./results/New-BERT/client5/fednlp_tc_A-Freeze-0-11.log 2>&1 &