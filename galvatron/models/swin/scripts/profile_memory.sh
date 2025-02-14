export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=localhost
export MASTER_PORT=$MASTER_PORT
# export NCCL_SOCKET_IFNAME=ib0
export NODE_RANK=0

LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"

export PROFILE_LAUNCHER="$LAUNCHER"
export PROFILE_TRAINER="train_dist_random.py"

MODEL_ARGS="
    --model_size swin-huge \
    --set_model_config_manually 0 \
    --set_layernum_manually 0 \
    --set_seqlen_manually 0 \
    --mlp_ratio 4 \
    --drop_path_rate 0.1 \
    --embed_dim 320 \
    --depths 1 1 1 1 \
    --num_heads 2 2 2 2 \
    --window_size 7 \
    --image_size 224 \
    --patch_size 16 \
    --num_channels 3 \
    --num_classes 1000"

PROFILE_ARGS_BF16="
    --profile_mode static \
    --profile_type memory \
    --profile_batch_size 8 \
    --layernum_min 1 \
    --layernum_max 2 \
    --max_tp_deg 8 \
    --profile_dp_type zero3 \
    --mixed_precision bf16"

# PROFILE_ARGS_BF16="
#     --profile_mode sequence \
#     --profile_type memory \
#     --profile_batch_size 8 \
#     --profile_min_seq_length 2048 \
#     --profile_max_seq_length 4096 \
#     --layernum_min 1 \
#     --layernum_max 2 \
#     --max_tp_deg 8 \
#     --profile_dp_type zero3 \
#     --mixed_precision bf16 \
#     --sequence_parallel \
#     --use-flash-attn"

# PROFILE_ARGS_FP32="
#     --profile_type memory \
#     --profile_batch_size 8 \
#     --layernum_min 1 \
#     --layernum_max 2 \
#     --max_tp_deg 8 \
#     --profile_dp_type zero3 \
#     --mixed_precision fp32"

python3 profiler.py ${MODEL_ARGS} ${PROFILE_ARGS_BF16}