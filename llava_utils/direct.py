from typing import List
import torch
from torch.nn import functional as F
from tqdm import tqdm
import sys, os
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, LlamaForCausalLM, LlamaForSequenceClassification
import pdb
import numpy as np
from transformers import LlamaTokenizer
import logging
from minigpt_utils.time_decorator import timeit

import numpy as np
def factors(x):
    return [i for i in range(1,x+1) if x%i==0]

def auto_size(seq_len, topk):
    estimated = (28672/(seq_len*1.5)) -11.52605
    # hack
    possible_facs = factors(topk)
    if np.all(~(np.array(possible_facs[::-1]) < estimated)): return 1
    return possible_facs[::-1][np.argmax(np.array(possible_facs[::-1]) < estimated)]
###

def create_attention_mask(seq_len, bsz=1):
    return torch.ones((bsz, seq_len))

# From huggingface
def rcache(past_key_values, beam_idx):
    reordered_past = ()
    for layer_past in past_key_values:
        reordered_past += (
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
        )
    return reordered_past

def even_chunk(data, chunk_size=10):
    assert data.shape[0] % chunk_size == 0, "chunk_size must evenly divide the topk"
    for i in range(0, data.shape[0], chunk_size):
        yield data[i:(i+chunk_size)]


class TQ_direct:
    @timeit
    def __init__(self, model, tokenizer, llm_dev="cuda:0", rm_dev="cuda:1", device="cpu", torch_dtype=torch.float16, reward_model="lomahony/eleuther-pythia6.9b-hh-dpo"):
        self.llm_dev = llm_dev
        self.rm_dev = rm_dev
        self.device = device
        logging.info("Loading Direct Transfer Code")

        logging.info("Loading LLM...")
        self.LLM = model

        logging.info(f"Loading tokenizer...")
        self.tokenizer = tokenizer

        logging.info("Loading RM...")
        self.RM = AutoModelForSequenceClassification.from_pretrained(reward_model, num_labels=1, torch_dtype=torch_dtype)
        self.RM = self.RM.to(self.rm_dev)
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model)
        self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token
        self.RM.eval()
      
       
    def get_input_ids(self, prompt: str, add_special_tokens=True) -> torch.Tensor:
        tokens = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids.to(self.llm_dev)
        return tokens
    
    def tokens_to_text(self, tokens: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
    
    def generate_greedy_step_large(self, mout, input_ids, pre_screen_beam_width=2, weight=0., rm_cached=None, chunk_size=10, debug=True, _use_cache=True):
        out_logits = mout.logits[:, -1]

        prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=pre_screen_beam_width)

        expanded_tis = torch.unsqueeze(input_ids, 1).repeat(1, pre_screen_beam_width, 1)
        if debug: logging.info(f"{expanded_tis.shape=}")

        to_rm_eval = torch.dstack((expanded_tis, prescreen_tokens))
        if debug: logging.info(f"{to_rm_eval.shape=}")

        if debug: logging.info(f"{out_logits.shape[0] * pre_screen_beam_width=}")
        flat_trme = to_rm_eval.view(out_logits.shape[0] * pre_screen_beam_width, -1)
        if debug: logging.info(f"{flat_trme.shape=}")
        
        new_rm_cached = None
        current_best_score = None
        current_best_tokens = None
        if debug: logging.info(f"{prescreen_logits.flatten().shape=}")
        for chunk, chunk_logits in zip(even_chunk(flat_trme.to(self.rm_dev), chunk_size), even_chunk(prescreen_logits.flatten(), chunk_size)):
            pkv = None if not _use_cache else rm_cached

            rm_out = self.RM(**self.LLM.prepare_inputs_for_generation(input_ids=chunk, attention_mask=create_attention_mask(chunk.shape[1], chunk.shape[0]).to(self.rm_dev), past_key_values=pkv, use_cache=True))
            current_rm_cached = rm_out.past_key_values
            rewards = rm_out.logits.flatten().to(self.llm_dev)
            del rm_out
            if debug: logging.info(f"{rewards=}")
            if debug: logging.info(f"{rewards.shape=}")
            new_scores = rewards * weight + chunk_logits
            if debug: logging.info(f"{new_scores=}")
            
            _, top_k_ids = torch.topk(new_scores, dim=-1, k=1)
            current_score = new_scores[top_k_ids[0]].item()
            if debug: logging.info(f"{current_score=} {current_best_score=} ")
            if (current_best_score is None) or (current_score > current_best_score):
                if debug: logging.info(f"Updated!!")
                
                current_best_score = current_score
                current_best_tokens = chunk.to(self.llm_dev)[top_k_ids]
                new_rm_cached = self.LLM._reorder_cache(current_rm_cached, top_k_ids.repeat(chunk_size,))
            
        if debug: logging.info(f"{new_scores.shape=}")
        
        return current_best_tokens, new_rm_cached

    @timeit
    def generate_step(self, mout, input_ids, prompt_token_len, pre_screen_beam_width=5, max_new_tokens=5, weight=0.5, method="topk", temperature=1, prompt_forward=False, debug=True, scores_output=[], rm_cached=None):
        out_logits = mout.logits[:, -1]
        prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=pre_screen_beam_width)
        if debug: logging.info(f'prescreen_logits: {prescreen_logits}')
        if debug: logging.info(f'prescreen_tokens: {prescreen_tokens}')
        
        expanded_tis = torch.unsqueeze(input_ids, 1).repeat(1, pre_screen_beam_width, 1)
        if debug: logging.info(f"{expanded_tis.shape=}")

        to_rm_eval = torch.dstack((expanded_tis, prescreen_tokens))
        if debug: logging.info(f"{to_rm_eval.shape=}")

        if debug: logging.info(f"{out_logits.shape[0] * pre_screen_beam_width=}")
        flat_trme = to_rm_eval.view(out_logits.shape[0] * pre_screen_beam_width, -1)
        if debug: logging.info(f"{flat_trme.shape=}")
        
        if prompt_forward:
            output = [self.tokenizer.decode(r.squeeze()) for r in flat_trme]
        else:
            output = [self.tokenizer.decode(r.squeeze()) for r in flat_trme[:, prompt_token_len:]]
            output = [self.redteam_query + elm for elm in output]

        if debug:
            logging.info("-------------------------")
            for possible_generations in output:
                logging.info(f"Possible continuation: {possible_generations}")
            logging.info("-------------------------")

        if rm_cached is None:
            rm_texts_tokens = self.reward_tokenizer(output, return_tensors='pt', padding=True)
            rm_out = self.RM(**rm_texts_tokens, past_key_values=None, use_cache=True)

        else:
            next_text = [self.tokenizer.decode(r.squeeze()) for r in flat_trme[:, -1:]]
            rm_texts_tokens_next = self.reward_tokenizer(next_text, return_tensors='pt', padding=False, add_special_tokens=False)
            rm_out = self.RM(**rm_texts_tokens_next, past_key_values=rm_cached, use_cache=True)


        rm_cached = rm_out['past_key_values']
        rewards = rm_out.logits.flatten().to(self.llm_dev)

        if debug: logging.info(f"{rewards=}")
        del rm_out
        if debug: logging.info(f"{rewards.shape=}")

        # ToDo: Both the logits need to be normalized before adding 
        new_scores = rewards * weight + prescreen_logits.flatten()
        if debug: logging.info(f"new_scores: {new_scores}")
        if debug: logging.info(f"{new_scores.shape=}")

        if method == "greedy":
            top_score, top_k_ids = torch.topk(new_scores, dim=-1, k=1)
          
        elif method == "topk":
            assert input_ids.shape[0] == 1
            new_scores = new_scores / temperature
            scores = F.softmax(new_scores, dim=-1)
            if debug: logging.info(f"normalized scores: {scores}")
            top_k_ids = torch.multinomial(scores, num_samples=1)
            top_score = new_scores[top_k_ids]

        elif method == "topp":
            top_p = 0.9
            assert input_ids.shape[0] == 1
            new_scores = new_scores / temperature
            scores = F.softmax(new_scores, dim=-1)
            if debug: logging.info(f"normalized scores: {scores}")
            sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_scores, dim=-1)
            # Create a mask for tokens with cumulative probability <= top_p
            cumulative_mask = cumulative_probs <= top_p
            cumulative_mask[0] = True
            probs_to_keep = sorted_scores[cumulative_mask]
            indices_to_keep = sorted_indices[cumulative_mask]
            probs_to_keep = probs_to_keep / probs_to_keep.sum()
            top_k_id = torch.multinomial(probs_to_keep, num_samples=1)
            top_k_ids = indices_to_keep[top_k_id]
            top_score = new_scores[top_k_ids]
        else:
            raise ValueError(f"Invalid method '{method}'")
        scores_output.append(top_score.item())  
        if debug: logging.info(f"top_k_ids: {top_k_ids}")
        if debug: logging.info(f"{top_k_ids.shape=}")
        if debug: logging.info(f"{rewards[top_k_ids]=}")
        if debug: logging.info(f"greedy output: {flat_trme[top_k_ids]}")

        return flat_trme[top_k_ids], scores_output, rm_cached

    @timeit
    def generate(self, input_ids, redteam_query, image, image_sizes, stopping_criteria=None, weight=0.5, topk=5, max_new_token=128, method="topp", temperature=1.0, chunk_size=5, prompt_forward=False, debug=True):
        tokens = input_ids['input_ids']
        if debug: logging.info(f"Input tokens shape: {tokens.shape}")
        initial_len = tokens.shape[1]
        print(tokens.shape)
        if chunk_size == "auto":
            chunk_size = auto_size(initial_len + max_new_token, topk)
            logging.info(f"auto {chunk_size=}, {topk=}, {initial_len=}!")
        
        if tokens.shape[-1] > self.LLM.config.to_dict().get("max_sequence_length", 2048):
            logging.info("The sequence of tokens is too long!!! Returning none!")
            return None
        
        if tokens.shape[-1] > self.RM.config.to_dict().get("max_sequence_length", 2048):
            logging.info("The sequence of tokens is too long!!! Returning none!")
            return None
        
        # fixing the initial tokens lenght
        prompt_token_len = tokens.shape[-1]
        # template for reward model inference
        redteam_query = "Given the image, " + redteam_query 
        tokens_rm = self.get_input_ids(redteam_query, add_special_tokens=True)

        scores = []
        rm_cached = None
        cached = None
        logging.info(f"Max new tokens  = {max_new_token}")
        iterator_obj = range(max_new_token)
        if debug: iterator_obj = tqdm(iterator_obj)
        for _ in iterator_obj:
            if debug: logging.info(f"{type(cached)=}")
            if debug: logging.info(f"{type(rm_cached)=}")

            with torch.no_grad():
                if cached is None:
                    mout = self.LLM(tokens, images=image, past_key_values=None, use_cache=False, return_dict=True)
                    cached = mout.past_key_values
                else:
                    mout = self.LLM(new_token, past_key_values=cached, use_cache=True, return_dict=True)
                    cached = mout.past_key_values

                tokens_rm, scores, rm_cached = self.generate_step(mout, tokens_rm, prompt_token_len, topk, weight, method, temperature, prompt_forward, debug, scores, rm_cached, redteam_query)
                new_token = tokens_rm[:, -1:]
                if debug: logging.info(f'tokens.shape: {tokens.shape}')
                if debug: logging.info(f"Selected new token: {new_token}")
                del mout
            
            tokens = torch.cat((tokens, new_token), dim=1)
            if stopping_criteria != None:
                if stopping_criteria(tokens_rm, scores): break

        return tokens_rm, scores