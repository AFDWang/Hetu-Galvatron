export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=localhost
export MASTER_PORT=29009
# export NCCL_SOCKET_IFNAME=ib0
export NODE_RANK=$RANK

LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --master_port ${MASTER_PORT}"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"

export PROFILE_LAUNCHER="$LAUNCHER"
export PROFILE_TRAINER="train_dist_random.py"

# MODEL_ARGS="
#     --model_size llama-7b \
#     --set_model_config_manually 0 \
#     --vocab_size 32000 \
#     --hidden_size 4096 \
#     --num_attention_heads 32 \
#     --seq_length 2048"

MODEL_ARGS="
    --model_size qwen2.5-1.5b \
    --set_model_config_manually 0 \
    --vocab_size 151936 \
    --hidden_size 1536 \
    --num_attention_heads 12 \
    --ffn_hidden_size 8960 \
    --seq_length 8192"

# MODEL_ARGS="
#     --model_size qwen2.5-3b \
#     --set_model_config_manually 0 \
#     --vocab_size 151936 \
#     --hidden_size 2048 \
#     --num_attention_heads 16 \
#     --ffn_hidden_size 11008 \
#     --seq_length 32768"

PROFILE_ARGS_BF16="
    --profile_mode static \
    --profile_type memory \
    --profile_batch_size 8 \
    --profile_seq_length_list 8192 \
    --layernum_min 1 \
    --layernum_max 2 \
    --max_tp_deg 2 \
    --profile_dp_type zero3 \
    --mixed_precision bf16 \
    --sequence_parallel \
    --use-flash-attn"

# PROFILE_ARGS_BF16="
#     --profile_mode sequence \
#     --profile_type memory \
#     --profile_batch_size 8 \
#     --profile_min_seq_length 4096 \
#     --profile_max_seq_length 8192 \
#     --profile_seq_length_step 1024 \
#     --layernum_min 1 \
#     --layernum_max 2 \
#     --max_tp_deg 2 \
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