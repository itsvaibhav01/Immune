import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.utils import save_image
from minigpt_utils import visual_attacker, prompt_wrapper

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry



def parse_args():

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg_path", default="eval_configs/minigpt4_llama2_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--n_iters", type=int, default=5, help="specify the number of iterations for attack.")
    parser.add_argument('--eps', type=int, default=32, help="epsilon of the attack budget")
    parser.add_argument('--alpha', type=int, default=1, help="step_size of the attack")
    parser.add_argument("--constrained", default=False, action='store_true')
    parser.add_argument("--img-file", default="adversarial_images/clean.jpeg", help="path to image file.")


    parser.add_argument("--save_dir", type=str, default='output',
                        help="save directory")

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


print('>>> Initializing Models')

args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
print("device:", device)
model = model_cls.from_config(model_config).to(device)
model.eval()

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

print('[Initialization Finished]\n')



if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)



import csv

file = open("harmful_corpus/derogatory_corpus.csv", "r")
data = list(csv.reader(file, delimiter=","))
file.close()
targets = []
num = len(data)
for i in range(num):
    targets.append(data[i][0])


my_attacker = visual_attacker.Attacker(args, model, targets, device=device, is_rtp=False)

template_img = 'adversarial_images/clean.jpeg'
img = Image.open(template_img).convert('RGB')
img = vis_processor(img).unsqueeze(0).to(model.device)


text_prompt_template = prompt_wrapper.minigpt4_chatbot_prompt_no_text_input



if not args.constrained:


    adv_img_prompt = my_attacker.attack_unconstrained(text_prompt_template,
                                                            img=img, batch_size=4,
                                                            num_iter=5000, alpha=args.alpha/255)

else:
    adv_img_prompt = my_attacker.attack_constrained(text_prompt_template,
                                                            img=img, batch_size=4,
                                                            num_iter=args.n_iters, alpha=args.alpha / 255,
                                                            epsilon=args.eps / 255)

save_image(adv_img_prompt, '%s/bad_prompt.bmp' % args.save_dir)
print('[Done]')
