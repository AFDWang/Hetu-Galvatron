export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=localhost
export MASTER_PORT=$MASTER_PORT
export NODE_RANK=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"
LAUNCHER="${LAUNCHER} --master_addr ${MASTER_ADDR}"
LAUNCHER="${LAUNCHER} --master_port ${MASTER_PORT}"
LAUNCHER="${LAUNCHER} --node_rank ${NODE_RANK}"

TRAINER="train_dist_random.py"

MODEL_ARGS="
    --model_size swin-large \
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

TRAIN_ARGS="
    --global_train_batch_size 16 \
    --train-iters 25 \
    --lr 1e-4 \
    --adam_weight_decay 0.01 \
    --dropout_prob 0.1 \
    --check_loss 0 \
    --profile 1 \
    --save_profiled_memory 0"

PARALLEL_ARGS="
    --pp_deg 2 \
    --global_tp_deg 1 \
    --global_tp_consec 1 \
    --sdp 0 \
    --global_checkpoint 0 \
    --vocab_tp 1 \
    --chunks 1 \
    --pipeline_type pipedream_flush \
    --default_dp_type zero2 \
    --mixed_precision bf16 \
    --initialize_on_meta 1" 
    # --galvatron_config_path ./configs/galvatron_config_llama-7b_2nodes_8gpus_per_node_40GB_bf16_example.json"

${LAUNCHER} ${TRAINER} ${MODEL_ARGS} ${TRAIN_ARGS} ${PARALLEL_ARGS}