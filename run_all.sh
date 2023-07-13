# BERT-base
# 20NEWS (provided in ae.pdf)
cd /app/experiments/distributed/transformer_exps/run_tc_exps
python trial_error.py \
    --dataset 20news\
    --round -1 \
    --depth 0 \
    --width 8 \
    --time_threshold 60 \
    --max_round 3000

# AGNEWS
cd /app/experiments/distributed/transformer_exps/run_tc_exps
python trial_error.py \
    --dataset agnews\
    --round -1 \
    --depth 0 \
    --width 8 \
    --time_threshold 90 \
    --max_round 3000

# SEMEVAL
cd /app/experiments/distributed/transformer_exps/run_tc_exps
python trial_error.py \
    --dataset semeval_2010_task8\
    --round -1 \
    --depth 0 \
    --width 8 \
    --time_threshold 90 \
    --max_round 3000

# ONTO
cd /app/experiments/distributed/transformer_exps/run_st_exps
python trial_error.py \
    --dataset onto\
    --round -1 \
    --depth 1 \
    --width 8 \
    --time_threshold 100 \
    --max_round 3000

# DistilBERT-base
mv /app/experiments/distributed/transformer_exps/run_tc_exps/run_text_classification_freeze.sh /app/experiments/distributed/transformer_exps/run_tc_exps/run_text_classification_freeze_bert.sh
mv /app/experiments/distributed/transformer_exps/run_tc_exps/run_text_classification_freeze_distilbert.sh /app/experiments/distributed/transformer_exps/run_tc_exps/run_text_classification_freeze.sh
mv /app/experiments/distributed/transformer_exps/run_st_exps/run_seq_tagging_trial.sh /app/experiments/distributed/transformer_exps/run_st_exps/run_seq_tagging_trial_bert.sh
mv /app/experiments/distributed/transformer_exps/run_st_exps/run_seq_tagging_trial_distilbert.sh /app/experiments/distributed/transformer_exps/run_st_exps/run_seq_tagging_trial.sh
# 20NEWS
cd /app/experiments/distributed/transformer_exps/run_tc_exps
python trial_error.py \
    --dataset 20news\
    --round -1 \
    --depth 0 \
    --width 8 \
    --time_threshold 60 \
    --max_round 3000

# AGNEWS
cd /app/experiments/distributed/transformer_exps/run_tc_exps
python trial_error.py \
    --dataset agnews\
    --round -1 \
    --depth 0 \
    --width 8 \
    --time_threshold 90 \
    --max_round 3000

# SEMEVAL
cd /app/experiments/distributed/transformer_exps/run_tc_exps
python trial_error.py \
    --dataset semeval_2010_task8\
    --round -1 \
    --depth 0 \
    --width 8 \
    --time_threshold 90 \
    --max_round 3000

# ONTO
cd /app/experiments/distributed/transformer_exps/run_st_exps
python trial_error.py \
    --dataset onto\
    --round -1 \
    --depth 1 \
    --width 8 \
    --time_threshold 100 \
    --max_round 3000