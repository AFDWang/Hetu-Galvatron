
CLUSTER_SIZE=64
GBS=512
ITER_NUM=40
MEMORY=30
TIMELIMIT=10
LOGDIR="solver_logs_10.3"

for MODEL_SIZE in "gpt-7b" "gpt-13b" "gpt-30b"; do
# for MODEL_SIZE in "gpt-7b"; do
for SEQ_LIMIT in 384 192; do
for DATASET in 'github_1' 'common_crawl' 'wikipedia_1'; do
for METHOD in 'static' 'adaptive' 'flexSP'; do
ARGS="
    --cluster_size $CLUSTER_SIZE \
    --memory_limit_gb $MEMORY \
    --model_size $MODEL_SIZE \
    --time_limit $TIMELIMIT \
    --dataset $DATASET \
    --seq_limit_k $SEQ_LIMIT \
    --global_batch_size $GBS \
    --method_type $METHOD \
    --chunk_alg sort_consec \
    --iter_num $ITER_NUM
"
nohup python solver.py --show_strategy 0 $ARGS 1>${LOGDIR}/${MODEL_SIZE}_${DATASET}_${SEQ_LIMIT}k_${METHOD}.log 2>&1 &
nohup python solver.py --show_strategy 1 $ARGS 1>${LOGDIR}_strategy/${MODEL_SIZE}_${DATASET}_${SEQ_LIMIT}k_${METHOD}.log 2>&1 &
done
done
done
done



# DATASET="github_1"
# METHOD="static"
# SEQ_LIMIT=384
# # SEQ_LIMIT=192
# # for MODEL_SIZE in "gpt-7b" "gpt-13b" "gpt-30b"; do
# for MODEL_SIZE in "gpt-7b"; do
# for CLUSTER_SIZE in 64; do
# ARGS="
#     --cluster_size $CLUSTER_SIZE \
#     --memory_limit_gb $MEMORY \
#     --model_size $MODEL_SIZE \
#     --time_limit $TIMELIMIT \
#     --dataset $DATASET \
#     --seq_limit_k $SEQ_LIMIT \
#     --global_batch_size $GBS \
#     --method_type $METHOD \
#     --chunk_alg sort_consec \
#     --iter_num $ITER_NUM \
#     --start_iter 3
# "

# python solver.py $ARGS
# done
# done


