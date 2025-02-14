export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=localhost
export MASTER_PORT=$MASTER_PORT
export NODE_RANK=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_HCA=mlx5_2,mlx5_5
LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"
LAUNCHER="${LAUNCHER} --master_addr ${MASTER_ADDR}"
LAUNCHER="${LAUNCHER} --master_port ${MASTER_PORT}"
LAUNCHER="${LAUNCHER} --node_rank ${NODE_RANK}"

TRAINER="train_dist.py"
DATA_PATH=/home/pkuhetu/lxy/dataset/imagenet

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

TRAIN_ARGS="
    --global_train_batch_size 32 \
    --train-iters 20 \
    --eval-iters 1 \
    --lr 0.0001 \
    --lr-decay-iters 20 \
    --lr-decay-style linear \
    --min-lr 0.00001 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --check_loss 0 \
    --profile 1 \
    --no_async_grad_reduce \
    --save_profiled_memory 0"

DATA_ARGS="
    --data-path $DATA_PATH $DATA_PATH \
    --split 949,50,1 \
    --tokenizer-type NullTokenizer \
    --vocab-size 0
"

# CKPT_ARGS="
#     --load /home/pkuhetu/lxy/checkpoints/llama2-7b-chat-hf-split
# "

# CKPT_ARGS="
#     --save /home/pkuhetu/lxy/checkpoints/galvatron_save_llama
#     --save-interval 10
# "

# CKPT_ARGS="
#     --load /home/pkuhetu/lxy/checkpoints/galvatron_save_llama \
#     --load_iteration 10 \
#     --distributed_checkpoint
# "

PARALLEL_ARGS="
    --pp_deg 2 \
    --global_tp_deg 1 \
    --global_tp_consec 1 \
    --sdp 1 \
    --global_checkpoint 1 \
    --vocab_tp 1 \
    --chunks 2 \
    --pipeline_type pipedream_flush \
    --default_dp_type zero2 \
    --mixed_precision bf16 \
    --initialize_on_meta 1 \
    --galvatron_config_path ./configs/galvatron_config_swin-huge_1nodes_8gpus_per_node_34GB_bf16_[tpconsec_off].json"

${LAUNCHER} ${TRAINER} ${MODEL_ARGS} ${TRAIN_ARGS} ${PARALLEL_ARGS} ${DATA_ARGS} ${CKPT_ARGS}