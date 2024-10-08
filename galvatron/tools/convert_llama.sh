
INPUT_PATH=/home/pkuhetu/lxy/checkpoints/llama2-7b-chat-hf
OUTPUT_PATH=/home/pkuhetu/lxy/checkpoints/llama2-7b-chat-hf-split

CHECKPOINT_ARGS="
    --input_checkpoint $INPUT_PATH \
    --output_dir $OUTPUT_PATH
"

python checkpoint_convert.py --model_type llama ${CHECKPOINT_ARGS}