export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
# export NCCL_SOCKET_IFNAME=ib0
export NODE_RANK=$RANK

LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"

export PROFILE_LAUNCHER="$LAUNCHER"
export PROFILE_TRAINER="train_dist_random.py"

MODEL_ARGS_SIZE15B="
    --model_size gpt-1.5b \
    --set_model_config_manually 0 \
    --vocab_size 50257 \
    --hidden_size 1600 \
    --num_attention_heads 32 \
    --seq_length 1024"

MODEL_ARGS_SIZE27B="
    --model_size gpt-2.7b \
    --set_model_config_manually 0 \
    --vocab_size 50257 \
    --hidden_size 2560 \
    --num_attention_heads 32 \
    --seq_length 2048"

MODEL_ARGS_SIZE67B="
    --model_size gpt-6.7b \
    --set_model_config_manually 0 \
    --vocab_size 50257 \
    --hidden_size 4096 \
    --num_attention_heads 32 \
    --seq_length 2048"

PROFILE_ARGS_BF16="
    --profile_mode sequence \
    --profile_type memory \
    --profile_batch_size 8 \
    --profile_min_seq_length 512 \
    --profile_max_seq_length 8192 \
    --layernum_min 1 \
    --layernum_max 2 \
    --max_tp_deg 8 \
    --profile_dp_type zero3 \
    --mixed_precision bf16 \
    --sequence_parallel \
    --use-flash-attn"

# python3 profiler.py ${MODEL_ARGS_SIZE15B} ${PROFILE_ARGS_BF16}
# python3 profiler.py ${MODEL_ARGS_SIZE27B} ${PROFILE_ARGS_BF16}
python3 profiler.py ${MODEL_ARGS_SIZE67B} ${PROFILE_ARGS_BF16}