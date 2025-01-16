export NUM_NODES=1
export NUM_GPUS_PER_NODE=1
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

PROFILE_ARGS="
    --profile_mode batch \
    --profile_type computation \
    --profile_min_batch_size 1 \
    --profile_max_batch_size 8 \
    --profile_batch_size_step 1 \
    --profile_seq_length_list 512,512 \
    --layernum_min 2 \
    --layernum_max 4 \
    --mixed_precision bf16 \
    --use-flash-attn"

# PROFILE_ARGS="
#     --profile_mode static \
#     --profile_type computation \
#     --profile_batch_size 12 \
#     --profile_seq_length_list 512,512 \
#     --layernum_min 12 \
#     --layernum_max 24 \
#     --mixed_precision bf16 \
#     --use-flash-attn"

python3 profiler.py ${MODEL_ARGS} ${PROFILE_ARGS}