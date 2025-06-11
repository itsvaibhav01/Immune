#!/bin/bash

# python3 -W ignore minigpt_inference.py \
# --input_csv ./datasets/JailbreakV-28K/JailBreakV_28K.csv \
# --output_file ./output/result_minigpt4-vicuna-default-jailbreak.jsonl \
# --generator default

python3 -W ignore llava_inference.py \
--model-path /path/to/weights/llava-v1.5-7b \
--input_csv ./csv/file/path \
--output_file ./output/file/path \
--generator enhanced \
--max_new_token 64
