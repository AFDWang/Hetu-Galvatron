
INPUT_PATH=/home/pkuhetu/lxy/checkpoints/g2h_llama
OUTPUT_PATH=/home/pkuhetu/lxy/checkpoints/h2g_llama

CHECKPOINT_ARGS="
    --input_checkpoint $INPUT_PATH \
    --output_dir $OUTPUT_PATH
"

python checkpoint_convert_h2g.py --model_type llama ${CHECKPOINT_ARGS}