export NUM_NODES=1
export NUM_GPUS_PER_NODE=1
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
# export NCCL_SOCKET_IFNAME=ib0
export NODE_RANK=$RANK

LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"

export PROFILE_LAUNCHER="$LAUNCHER"
export PROFILE_TRAINER="train_dist.py"

MODEL_ARGS="
    --model_size llama-7b \
    --set_model_config_manually 0 \
    --vocab_size 32000 \
    --hidden_size 4096 \
    --num_attention_heads 32 \
    --seq_length 2048"

PROFILE_ARGS="
    --profile_type computation \
    --profile_batch_size 2 \
    --layernum_min 6 \
    --layernum_max 12 \
    --mixed_precision bf16 \
    --use-flash-attn"

# models in flash_attn cannot use fp32 without flash_attn
# PROFILE_ARGS="
#     --profile_type computation \
#     --profile_batch_size 4 \
#     --layernum_min 12 \
#     --layernum_max 24 \
#     --mixed_precision fp32"

python3 profiler.py ${MODEL_ARGS} ${PROFILE_ARGS}