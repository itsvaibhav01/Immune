import argparse
import os
import random

import numpy as np
import pandas as pd 
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
from PIL import Image
import json

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from minigpt_utils import prompt_wrapper, generator, generator_attack, time_decorator

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(filename)s - Line: %(lineno)d - Function: %(funcName)s - %(message)s',
    handlers=[
        logging.StreamHandler() 
    ]
)

def rtp_read(text_file):
    dataset = []
    lines = open(text_file).read().split("\n")
    for li in lines:
        obj = json.loads(li)
        dataset.append(obj['text'])
    return dataset

def read_jailbreak_file(csv_file):
    dataset = []

    if 'csv' in csv_file.suffix:
        df = pd.read_csv(csv_file)

    elif 'json' in csv_file.suffix:
        with open(csv_file) as outf:
            lines = json.load(outf)
        df = pd.DataFrame(lines).T

    for idx, row in df.iterrows():
        dataset.append({'jailbreak_query': row['question'], 'redteam_query':row['question']})
    return dataset, df


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")

    parser.add_argument("--mode", type=str, default='VisualChatBot',
                        choices=[ "TextOnly", "VisualChatBot" ],
                        help="Inference Mode: TextOnly: Text model only (Vicuna) \n VisualChatBot: Vision model + Text model (MiniGPT4) ")


    parser.add_argument("--image_file", type=str, default='./adversarial_images/clean.jpeg',
                        help="Image file")
    parser.add_argument("--output_file", type=str, default='./result_minigpt4-llama2-clean-gen-tq.jsonl',
                        help="Output file.")
    
    parser.add_argument("--generator", type=str, default="enhanced", help="select generator either the `original` one or our `enhanced` one.")

    parser.add_argument("--input_csv", type=str, help="path of cvs file for processing inputs")

    parser.add_argument("--input_jsonl", type=str, default="harmful_corpus/test_prompts.jsonl", help="path of jsonl file for processing inputs")

    parser.add_argument("--max_new_token", type=int, default=64, help="max tokens to generate in output")

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

logging.info('>>> Initializing Models')

args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
logging.info(f"device: {device}")
model = model_cls.from_config(model_config).to(device)
model = model.float()

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
logging.info('Initialization Finished')

if args.generator == "enhanced":
    my_generator = generator.Generator(model=model, device=device, max_new_tokens=args.max_new_token)
else:
    my_generator = generator_attack.Generator(model=model, device=device)



# ========================================
#             Inference
# ========================================

##  TODO: expose interface.
if args.input_csv != None:
    input_csv_pth = Path(args.input_csv).resolve()
    datasets, df = read_jailbreak_file(args.input_csv)

else :
    datasets = rtp_read(args.input_jsonl)

if args.mode == "TextOnly":
    ## vicuna official system message.
    prefix = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
    prefix += "###Human:%s ###Assistant:"
    img_prompt = []

elif args.mode == "VisualChatBot":
    prefix = prompt_wrapper.minigpt4_chatbot_prompt
    img = Image.open(args.image_file).convert('RGB')
    img_prompt = [vis_processor(img).unsqueeze(0).to(model.device)]

text_prompt = prefix % ('Please extend the following sentence: %s')
logging.info(f"Instructions: {text_prompt}")


prompt = prompt_wrapper.Prompt(model=model, img_prompts=[img_prompt], device=device)

# Setup stats logging
stats_saved_pth = f'./{Path(args.output_file).parent}/elapsed'
os.makedirs(stats_saved_pth, exist_ok=True)


out = []
with torch.no_grad():
    for i, user_message_obj in enumerate(datasets):
        user_message = user_message_obj['jailbreak_query']
        if args.input_csv != None:
            # get image embeddings 
            prefix = prompt_wrapper.minigpt4_chatbot_prompt
            img = Image.open(f"{input_csv_pth.parent}/images/{df.iloc[i].image_path}").convert('RGB')
            img_prompt = [vis_processor(img).unsqueeze(0).to(model.device)]
            prompt = prompt_wrapper.Prompt(model=model, img_prompts=[img_prompt], device=device)

        logging.info(f" ----- {i} ----")
        logging.info(" -- prompt: ---")
        logging.info(user_message)

        prompt.update_text_prompt([user_message])
        response, _ = my_generator.generate(prompt, redteam_query=user_message_obj['redteam_query'])

        logging.info(" -- continuation: ---")
        logging.info(response)
        out.append({'prompt': user_message, 'continuation': response})
        logging.info("-------------------")

        # saving execution stats to json
        time_decorator.save_execution_stats(f"{stats_saved_pth}/{Path(args.output_file).name}")


with open(args.output_file, 'w') as f:
    f.write(json.dumps({
        "args": vars(args)
    }))
    f.write("\n")

    for li in out:
        f.write(json.dumps(li))
        f.write("\n")

if args.input_csv != None:
    df['response'] = [li['continuation'] for li in out]
    output_path = Path(args.output_file).resolve()
    df.to_json(f"{output_path.parent}/{output_path.stem}-gen-{args.generator}.jsonl")
