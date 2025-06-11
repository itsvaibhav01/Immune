import torch
import logging
from transformers import StoppingCriteria, StoppingCriteriaList
from llava.conversation import conv_llava_v1, SeparatorStyle
from llava_utils.direct_immune_api_rewared import TQ_direct
import pdb

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = [tokenizer(keyword).input_ids for keyword in keywords]
        self.keyword_ids = [keyword_id[0] for keyword_id in self.keyword_ids if type(keyword_id) is list and len(keyword_id) == 1]
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            for keyword_id in self.keyword_ids:
                if output_ids[0, -1] == keyword_id:
                    return True
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


class Generator:

    def __init__(self, model, tokenizer, max_new_tokens=128, temperature=0.2, device='cuda:0'):
        print('enhanced gen setup')
        self.model = model
        self.device = device
        self.tokenizer = tokenizer

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.stop_str = conv_llava_v1.sep if conv_llava_v1.sep_style != SeparatorStyle.TWO else conv_llava_v1.sep2
        self.keywords = [self.stop_str]

        self.tq_direct = TQ_direct(
            model=self.model, 
            tokenizer=self.tokenizer, 
            llm_dev="cuda", 
            rm_dev="cuda",
            device="cuda",
            reward_model= "LxzGordon/URM-LLaMa-3.1-8B", 
        )


    def generate(self, prompt, image, image_sizes, redteam_query):

        input_ids = prompt.input_ids[0]

        stopping_criteria = KeywordsStoppingCriteria(self.keywords, self.tokenizer, input_ids)

        output_ids, scores =  self.tq_direct.generate(
            input_ids, 
            redteam_query,
            image, 
            image_sizes, 
            stopping_criteria,
            max_new_token=self.max_new_tokens, 
            prompt_forward=False,
            debug=False,
        )
    
        outputs = self.tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

        return outputs
