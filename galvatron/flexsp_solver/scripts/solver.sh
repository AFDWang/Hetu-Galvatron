
CLUSTER_SIZE=64
MODEL_SIZE="gpt-7b"
# MODEL_SIZE="gpt-13b"
# MODEL_SIZE="gpt-30b"
# DATASET="github_1"
DATASET="common_crawl"
# DATASET="wikipedia"
# DATASET="wikipedia_1"
SEQ_LIMIT=32
# SEQ_LIMIT=256
# SEQ_LIMIT=192
# SEQ_LIMIT=128
# SEQ_LIMIT=96
# SEQ_LIMIT=64
# SEQ_LIMIT=48
# GBS=1024
GBS=512
# GBS=256
# GBS=128
# ITER_NUM=100
ITER_NUM=40
METHOD="adaptive"
METHOD="flexSP"

ARGS="
    --cluster_size $CLUSTER_SIZE \
    --memory_limit_gb 30 \
    --model_size $MODEL_SIZE \
    --time_limit 10 \
    --dataset $DATASET \
    --seq_limit_k $SEQ_LIMIT \
    --global_batch_size $GBS \
    --method_type $METHOD \
    --chunk_alg sort_consec \
    --iter_num $ITER_NUM \
    --show_strategy 0 \
    --start_iter 38 \
    --even_bucketing \
    --start_iter 0
"
# --redist-without-empty-group

python solver.py $ARGS