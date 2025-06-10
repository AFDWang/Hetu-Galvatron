export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=localhost
export MASTER_PORT=29000
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
DATA_PATH=/home/galvatron_dev/lqs/megatron_data/my_qwen_text 
VOCAB_FILE=/home/galvatron_dev/lqs/megatron_data/qwen2.5_tokenizer/vocab.json
TOKENIZER_MODEL=/home/galvatron_dev/lqs/megatron_data/qwen2.5_tokenizer

MODEL_ARGS="
    --model_size qwen2.5-3b \
    --set_model_config_manually 0 \
    --set_layernum_manually 0 \
    --set_seqlen_manually 1 \
    --vocab_size 32000 \
    --hidden_size 4096 \
    --num_hidden_layers 8 \
    --num_attention_heads 32 \
    --seq_length 8192"

TRAIN_ARGS="
    --global_train_batch_size 64 \
    --train-iters 20 \
    --eval-iters 1 \
    --lr 1.25e-6 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --lr-warmup-fraction 0.1 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1.0e-5 \
    --init-method-std 0.01 \
    --adam_weight_decay 0.01 \
    --dropout_prob 0.1 \
    --check_loss 0 \
    --profile 1 \
    --save_profiled_memory 0"

DATA_ARGS="
    --data-path /home/galvatron_dev/lqs/megatron_data/my_qwen_text/my-qwen2.5_text_text_sentence \
    --split 949,50,1 \
    --tokenizer-type NullTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --num-workers 0
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
    --sdp 0 \
    --global_checkpoint 1 \
    --vocab_tp 1 \
    --chunks 16 \
    --pipeline_type pipedream_flush \
    --default_dp_type zero2 \
    --mixed_precision bf16 \
    --sequence-parallel \
    --use-flash-attn \
    --initialize_on_meta 1 \
    --galvatron_config_path /home/galvatron_dev/lqs/Hetu-Galvatron/galvatron/models/llama_hf/configs/galvatron_config_qwen2.5-3b_1nodes_8gpus_per_node_23GB_bf16_bsz1024_[tpconsec_off].json "

${LAUNCHER} ${TRAINER} ${MODEL_ARGS} ${TRAIN_ARGS} ${PARALLEL_ARGS} ${DATA_ARGS} ${CKPT_ARGS}