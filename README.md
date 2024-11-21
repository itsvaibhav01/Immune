# IMMUNE <img src="./static/images/favicon.svg" alt="Favicon" style="width: 1em; height: 1em; vertical-align: middle;">:  Improving Safety Against Jailbreaks in Multi-modal LLMs via Inference-Time Alignment


Immune, a novel inference-time defense framework, enhances the safety of Multimodal Large Language Models (MLLMs) against jailbreak attacks. By leveraging a safe reward model during decoding, Immune effectively rejects harmful queries while preserving the model’s original utility and performance.

![teaser index](./static/images/teaser_index.png)

## Abstract

With the widespread deployment of Multimodal Large Language Models (MLLMs) for visual-reasoning tasks, improving their safety has become crucial. Recent research indicates that despite training-time safety alignment, these models remain vulnerable to jailbreak attacks—carefully crafted image-prompt pairs that compel the model to generate harmful content. In this work, we first highlight a critical safety gap, demonstrating that alignment achieved solely through safety training may be insufficient against jailbreak attacks. To address this vulnerability, we propose Immune, an inference-time defense framework that leverages a safe reward model during decoding to defend against jailbreak attacks. Additionally, we provide a rigorous mathematical characterization of Immune, offering provable guarantees against jailbreaks. Extensive evaluations on diverse jailbreak benchmarks using recent MLLMs reveal that Immune effectively enhances model safety while preserving the model’s original capabilities. For instance, against text-based jailbreak attacks on LLaVA-1.6, Immune reduces the attack success rate by 57.82% and 16.78% compared to the base MLLM and state-of-the-art defense strategy, respectively.

## Code 

Immune leverages a safety alignment framework during inference, combining model logits with a safety score generated by a reward model. This approach dynamically adjusts token probabilities to suppress harmful responses, ensuring ethical outputs while maintaining the model’s performance and utility.

![teaser algo](./static/images/teaser_algo.jpg)

---
### MiniGPT-4 with Vicuna-7B: Setup and Usage Guide
#### **Environment Setup**

To set up the environment, create a virtual environment as specified in the [MiniGPT-4 documentation](https://github.com/Vision-CAIR/MiniGPT-4?tab=readme-ov-file#installation). Execute the following commands:

```bash
conda env create -f minigptv-environment.yml
conda activate minigptv
```

#### **Model Setup**

##### Step 1: Download Vicuna-7B Weights
Obtain the model weights for Vicuna-7B from the [Vision-CAIR/vicuna-7b repository](https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main). Ensure `git-lfs` is installed to clone the repository.

```bash
git clone https://huggingface.co/Vision-CAIR/vicuna-7b
```

##### **Step 2: Update Model Path**
Locate the configuration file `minigpt4/configs/models/minigpt4_vicuna0.yaml`. Update the path to the Vicuna-7B model weights in **line 18** with the path where the model was downloaded in Step 1.

##### **Step 3: Download MiniGPT-4 Head**
Download the MiniGPT-4 head for Vicuna from [this link](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view) and save it in an accessible directory.

##### **Step 4: Update MiniGPT-4 Head Path**
Locate the configuration file `eval_configs/minigpt4_eval.yaml`. Update the path to the MiniGPT-4 head in **line 8** with the directory where you saved the file in Step 3.

---

### **Running Inference**

##### **General Command Syntax**
The `minigpt_inference.py` script allows for flexible inference using different input formats and configurations. Here’s how to structure your commands:
- `cfg-path`: Path to the YAML configuration file (e.g., `eval_configs/minigpt4_eval.yaml`).
- `input_jsonl`: Path to the `.jsonl` file containing prompts.
- `input_csv`: Path to the `.csv` file containing prompts.
- `image_file`: Path to the input image file.
- `output_file`: Path to save the output in `.json` or `.xlsx`.
- `generator`: Specify `enhanced` for the updated generator or `default` for the baseline.
- `max_new_token`: Maximum number of output tokens when using the `enhanced` generator.

##### **Examples of Inference Commands**
1. **Inference with `jsonl` file and single image**:
   ```bash
   python3 -W ignore minigpt_inference.py \
   --image_file ./adversarial_images/bad_prompt_temp_1790.bmp \
   --output_file ./output/result_minigpt4-llama2-p64_bp_1790.jsonl \
   --generator enhanced
   ```

2. **Inference for JailbreakV-28K with a CSV file**:
   Output will be recorded in an `.xlsx` file in the same directory.
   ```bash
   python3 -W ignore minigpt_inference.py \
   --input_csv ./datasets/JailbreakV-28K/test_jailbreak.csv \
   --output_file ./output/result_minigpt4-llama2-test-jailbreak.jsonl \
   --generator enhanced \
   --max_new_token 300
   ```

3. **Inference using the default generator**:
   ```bash
   python3 -W ignore minigpt_inference.py \
   --input_csv ./datasets/JailbreakV-28K/test_jailbreak.csv \
   --output_file ./output/result_minigpt4-llama2-test-jailbreak.jsonl \
   --generator default
   ```

---

### **Performing Adversarial Attacks**

To generate adversarial prompts, place an image named `clean.jpeg` in the `adversarial_images` folder. Then, run the following command. The adversarial output will be saved in the `output` folder as `bad_prompt.bmp`.

```bash
python3 -W ignore minigpt_visual_attack.py --gpu_id 1 --n_iters 1000 --eps 64
```


## Evaluation Datasets

Our method has been evaluated on the following datasets, which encompass diverse scenarios and benchmarks for assessing the safety and robustness of Multimodal Large Language Models (MLLMs):

1. **[FigStep](https://github.com/ThuCCSLab/FigStep)**:  
   A dataset designed for evaluating the interpretability and reasoning capabilities of models with step-by-step instructions and multi-image reasoning.

2. **[JailBreakV-28K](https://github.com/EddyLuo1232/JailBreakV_28K)**:  
   A comprehensive benchmark of 28,000 adversarial text-image pairs crafted to evaluate models' susceptibility to jailbreak attacks and harmful content generation.

3. **[MM-SafetyBench](https://github.com/isXinLiu/MM-SafetyBench)**:  
   A benchmark focusing on assessing the safety alignment of MLLMs against adversarial and harmful inputs, with scenarios targeting ethical and safety considerations.

4. **[Visual Adversarial Examples for Jailbreaking LLMs](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models)**:  
   A dataset featuring adversarial visual prompts designed to test the vulnerability of LLMs in generating harmful or unethical outputs when paired with malicious queries.

## Comparison of Defense Strategies

The image below demonstrates how Immune outperforms other defense strategies in neutralizing adversarial attacks. While existing methods fail to handle harmful prompts effectively, Immune provides safe, ethical, and bias-free responses, ensuring robust safety alignment.

![teaser 1](./static/images/teaser1.jpg)
![teaser 2](./static/images/teaser2.jpg)
![teaser 3](./static/images/teaser3.jpg)




