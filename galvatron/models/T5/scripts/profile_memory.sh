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
    --model_size t5-3B \
    --set_model_config_manually 0 \
    --set_layernum_manually 0 \
    --set_seqlen_manually 0 \
    --vocab_size 32000 \
    --hidden_size 4096 \
    --num_encoder_layers 8 \
    --num_decoder_layers 8 \
    --num_attention_heads 32 \
    --encoder_seq_length 512 \
    --decoder_seq_length 512"

PROFILE_ARGS_BF16="
    --profile_mode static \
    --profile_type memory \
    --profile_batch_size 8 \
    --profile_seq_length_list 512,512 \
    --layernum_min 1 \
    --layernum_max 2 \
    --max_tp_deg 8 \
    --profile_dp_type zero3 \
    --mixed_precision bf16 \
    --sequence_parallel \
    --use-flash-attn"

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