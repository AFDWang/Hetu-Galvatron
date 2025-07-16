export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=localhost
export MASTER_PORT=29007
export NODE_RANK=$RANK
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"
LAUNCHER="${LAUNCHER} --master_addr ${MASTER_ADDR}"
LAUNCHER="${LAUNCHER} --master_port ${MASTER_PORT}"
LAUNCHER="${LAUNCHER} --node_rank ${NODE_RANK}"
TRAINER="train_dist_random.py"

# MODEL_ARGS="
#     --model_size llama-7b \
#     --set_model_config_manually 0 \
#     --set_layernum_manually 0 \
#     --set_seqlen_manually 1 \
#     --vocab_size 32000 \
#     --hidden_size 4096 \
#     --num_hidden_layers 8 \
#     --num_attention_heads 32 \
#     --seq_length 2048"

MODEL_ARGS="
    --model_size qwen2.5-1.5b \
    --set_model_config_manually 0 \
    --set_layernum_manually 0 \
    --set_seqlen_manually 1 \
    --vocab_size 151936 \
    --hidden_size 1536 \
    --ffn_hidden_size 8960 \
    --num_hidden_layers 28 \
    --num_attention_heads 12 \
    --seq_length 8192"

TRAIN_ARGS="
    --global_train_batch_size 512 \
    --train-iters 25 \
    --lr 1e-4 \
    --adam_weight_decay 0.01 \
    --dropout_prob 0.1 \
    --check_loss 0 \
    --profile 1 \
    --save_profiled_memory 0"

PARALLEL_ARGS="
    --pp_deg  \
    --global_tp_deg 1 \
    --global_tp_consec 1 \
    --sdp 0 \
    --global_checkpoint 0 \
    --vocab_tp 1 \
    --chunks 1 \
    --pipeline_type pipedream_flush \
    --default_dp_type zero2 \
    --mixed_precision bf16 \
    --sequence-parallel \
    --use-flash-attn \
    --initialize_on_meta 1 \
    --galvatron_config_path /home/galvatron_dev/lqs/new_galvatron/Hetu-Galvatron/galvatron/models/llama_hf/configs/galvatron_config_qwen2.5-1.5b_1nodes_8gpus_per_node_23GB_bf16_bsz512_[tpconsec_off].json"

${LAUNCHER} ${TRAINER} ${MODEL_ARGS} ${TRAIN_ARGS} ${PARALLEL_ARGS}