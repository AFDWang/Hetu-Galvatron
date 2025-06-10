export NUM_NODES=1
export NUM_GPUS_PER_NODE=8

MODEL_SIZE="qwen2.5-1.5b"
MEMORY=24
SEQ=4096
FINE_GRAINED=1
MODEL_ARGS="
    --model_size ${MODEL_SIZE} \
    --set_model_config_manually 0 \
    --set_layernum_manually 0 \
    --set_seqlen_manually 1 \
    --vocab_size 151936 \
    --hidden_size 1536 \
    --ffn_hidden_size 8960 \
    --num_hidden_layers 28 \
    --num_attention_heads 12 \
    --seq_length ${SEQ}"

BSZ_ARGS="
    --min_bsz 1024 \
    --max_bsz 1024 \
    --bsz_scale 1 \
    --settle_bsz 1024 \
    --recommend_min_bsz 0
"

SEARCH_SPACE_ARGS="
    --search_space full \
    --sp_space tp+sp \
    --disable_dp 0 \
    --disable_tp 0 \
    --disable_pp 0 \
    --disable_sdp 0 \
    --disable_ckpt 0 \
    --disable_vtp 0 \
    --disable_tp_consec 1 \
    --max_tp_deg 2 \
    --max_pp_deg 8 \
    --fine_grained_mode ${FINE_GRAINED} \
    --sequence_parallel \
    --time_profile_mode sequence \
    --memory_profile_mode sequence
"
#已去掉no async grad reduce
SEARCH_ARGS="
    ${BSZ_ARGS} \
    ${SEARCH_SPACE_ARGS} \
    ${MODEL_ARGS} \
    --num_nodes ${NUM_NODES} \
    --num_gpus_per_node ${NUM_GPUS_PER_NODE} \
    --memory_constraint $MEMORY \
    --mixed_precision bf16 \
    --pipeline_type pipedream_flush \
    --default_dp_type zero2 \
    --parallel_search \
    --worker 8
"

BACKGROUND=1

if [ $BACKGROUND -eq 1 ]; then
    echo "Search in background..."
    OUTPUT_FILE="Search_${MODEL_SIZE}_${MEMORY}GB_${NUM_NODES}Nodes_${NUM_GPUS_PER_NODE}GPUs_per_node_${SEQ}_${FINE_GRAINED}.log"
    nohup python3 search_dist.py ${SEARCH_ARGS} 1> ${OUTPUT_FILE} 2>&1 &
else
    echo "Search in foreground..."
    python3 search_dist.py ${SEARCH_ARGS}
fi