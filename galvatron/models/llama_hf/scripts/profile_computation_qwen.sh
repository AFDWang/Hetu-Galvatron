export NUM_NODES=1
export NUM_GPUS_PER_NODE=1
export MASTER_ADDR=localhost
export MASTER_PORT=29000
# export NCCL_SOCKET_IFNAME=ib0
export NODE_RANK=0

LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --master_port ${MASTER_PORT}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"

export PROFILE_LAUNCHER="$LAUNCHER"
export PROFILE_TRAINER="train_dist_random.py"

MODEL_ARGS="
    --model_size qwen2.5-3b \
    --set_model_config_manually 0 \
    --vocab_size 151936 \
    --hidden_size 1536 \
    --ffn_hidden_size 8960 \
    --num_attention_heads 12 \
    --seq_length 4096"

# MODEL_ARGS="
#     --model_size qwen2.5-3b \
#     --set_model_config_manually 0 \
#     --vocab_size 151936 \
#     --hidden_size 2048 \
#     --num_attention_heads 16 \
#     --ffn_hidden_size 11008 \
#     --seq_length 32768"

PROFILE_ARGS="
    --profile_mode sequence \
    --profile_type computation \
    --profile_batch_size 1 \
    --profile_min_seq_length 4096 \
    --profile_max_seq_length 8192 \
    --profile_seq_length_step 1024 \
    --layernum_min 1 \
    --layernum_max 2 \
    --mixed_precision bf16 \
    --use-flash-attn"

# models in flash_attn cannot use fp32 without flash_attn
# PROFILE_ARGS="
#     --profile_mode static \
#     --profile_type computation \
#     --profile_batch_size 4 \
#     --layernum_min 12 \
#     --layernum_max 24 \
#     --mixed_precision fp32"

python3 profiler.py ${MODEL_ARGS} ${PROFILE_ARGS} 