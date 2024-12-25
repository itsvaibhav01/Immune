python3 -W ignore minigpt_inference.py \
--input_csv ./datasets/JailbreakV-28K/JailBreakV_28K.csv \
--output_file ./output/result_minigpt4-vicuna-default-jailbreak.jsonl \
--generator default

# python3 -W ignore minigpt_inference.py \
# --input_csv ./datasets/JailbreakV-28K/JailBreakV_28K.csv \
# --output_file ./output/result_minigpt4-vicuna-immune-jailbreak.jsonl \
# --generator enhanced \
# --max_new_token 128
