
INPUT_PATH=/home/pkuhetu/lxy/checkpoints/Cerebras-GPT-6.7B
OUTPUT_PATH=/home/pkuhetu/lxy/checkpoints/Cerebras-GPT-6.7B-split

CHECKPOINT_ARGS="
    --input_checkpoint $INPUT_PATH \
    --output_dir $OUTPUT_PATH
"

python checkpoint_convert.py --model_type gpt ${CHECKPOINT_ARGS}