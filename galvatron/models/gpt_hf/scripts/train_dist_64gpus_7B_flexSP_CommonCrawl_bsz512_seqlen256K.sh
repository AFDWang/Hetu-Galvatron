export NUM_NODES=8
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=6666
export NODE_RANK=$RANK
export CUDA_DEVICE_MAX_CONNECTIONS=1

LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"
LAUNCHER="${LAUNCHER} --master_addr ${MASTER_ADDR}"
LAUNCHER="${LAUNCHER} --master_port ${MASTER_PORT}"
LAUNCHER="${LAUNCHER} --node_rank ${NODE_RANK}"

TRAINER="train_dist.py"
tp_deg=1
MODEL_ARGS="
    --model_size gpt-7b \
    --set_model_config_manually 0 \
    --set_layernum_manually 0 \
    --vocab_size 50257 \
    --hidden_size 1600 \
    --num_hidden_layers 12 \
    --num_attention_heads 32 \
    --seq_length 262144"

TRAIN_ARGS="
    --global_train_batch_size  512 \
    --epochs 10 \
    --lr 1e-4 \
    --adam_weight_decay 0.01 \
    --dropout_prob 0.1 \
    --check_loss 1 \
    --profile 1 \
    --save_profiled_memory 0"

PARALLEL_ARGS="
    --pp_deg 1 \
    --global_tp_deg $tp_deg \
    --global_tp_consec 1 \
    --sdp 1 \
    --global_checkpoint 0 \
    --selective_checkpoint 1 \
    --vocab_tp $tp_deg \
    --chunks 1 \
    --pipeline_type pipedream_flush \
    --default_dp_type zero2 \
    --mixed_precision bf16 \
    --sequence-parallel \
    --use-flash-attn \
    --use-ulysses \
    --use-packing \
    --use-flexSP \
    --dataset common_crawl \
    --initialize_on_meta 1"
${LAUNCHER} ${TRAINER} ${MODEL_ARGS} ${TRAIN_ARGS} ${PARALLEL_ARGS}
