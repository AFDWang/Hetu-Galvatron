
INPUT_PATH=/home/pkuhetu/lxy/checkpoints/galvatron_save_llama/
OUTPUT_PATH=/home/pkuhetu/lxy/checkpoints/g2h_llama

CHECKPOINT_ARGS="
    --input_checkpoint $INPUT_PATH \
    --output_dir $OUTPUT_PATH \
    --model_config llama-7b \
    --load_iteration 10
"

python checkpoint_convert_g2h.py --model_type llama ${CHECKPOINT_ARGS}