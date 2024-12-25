import torch
import logging
from minigpt_utils.time_decorator import timeit
from transformers import StoppingCriteria, StoppingCriteriaList, LogitsProcessorList
from minigpt_utils.direct import TQ_direct

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


class Generator:
    def __init__(self, model, max_new_tokens=300, weight=5.0, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, device='cuda:0'):
        print("Running enhanced decoding!!!".center(100, "*"))
        self.model = model
        self.device = device

        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.min_length = min_length
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.temperature = temperature

        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device),
                          torch.tensor([13]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
     
        self.tq_direct = TQ_direct(
            model=self.model.llama_model, 
            tokenizer=self.model.llama_tokenizer, 
            llm_dev="cuda:0", 
            rm_dev="cuda:0",
            device="cuda:0",
            reward_model= "LxzGordon/URM-LLaMa-3.1-8B", 
        )

    @timeit
    def generate(self, prompt, redteam_query):
        return self.custom_generate(prompt, redteam_query)
    
    def custom_generate(self, prompt, redteam_query):
        logging.info(f"text_prompts: {prompt.text_prompts}")
        output_token, scores =  self.tq_direct.generate(prompt, redteam_query, self.stopping_criteria, max_new_token=self.max_new_tokens, debug=False)
        logging.info(f'score RM: {scores}')
        
        # Decode the generated tokens
        logging.info(f'output_tokens: {output_token}')
        logging.info(f'output token shape: {output_token.shape}')

        output_text = self.model.llama_tokenizer.decode(output_token.view(-1), add_special_tokens=False)

        output_text = output_text.split('###Assistant:')[-1].strip()

        return output_text, output_token.cpu().numpy()
