export NUM_NODES=1
export NUM_GPUS_PER_NODE=8

MODEL_SIZE="t5-3B"
MEMORY=34
FINE_GRAINED=1
MODEL_ARGS="
    --model_size ${MODEL_SIZE} \
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

BSZ_ARGS="
    --min_bsz 1024 \
    --max_bsz 1024 \
    --bsz_scale 1 \
    --settle_bsz -1 \
    --settle_chunk -1 \
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
    --max_tp_deg 8 \
    --max_pp_deg 16 \
    --fine_grained_mode ${FINE_GRAINED} \
    --time_profile_mode batch \
    --memory_profile_mode static \
    --no_async_grad_reduce \
    --sequence_parallel
"

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
    --embed_sdp 0
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