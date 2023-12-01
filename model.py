
# Ref: https://github.com/voidism/DoLa/

import argparse
import time
import csv
import tqdm
import os
import json

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, AutoConfig
from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria

import argparse
import warnings
import pandas as pd
import numpy as np



class LLaMA_Analysis:
    def __init__(self, model_name, device, num_gpus, hidden_layers=80, max_gpu_memory=27):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory
        self.hidden_layers = hidden_layers
        self.model, self.tokenizer = self.load_model(model_name)

    def load_model(self, model_name):
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.bfloat16, "offload_folder": f"{model_name}/offload"}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")
        
        # config = AutoConfig.from_pretrained(model_name)
        # print(config)
        
        # config.num_hidden_layers =  self.hidden_layers
        # print(config)
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        if 'chatglm' in model_name:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
            low_cpu_mem_usage=True, config=config, **kwargs)
        else:
            config = AutoConfig.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name,
            low_cpu_mem_usage=True, config=config, **kwargs)

        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()
        
        return model, tokenizer

    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
        self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))

    
    def get_relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1):
        scores_normalized = scores.log_softmax(dim=-1) 
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores_normalized < probs_thresh

    def lm_score(self, input_text1, input_text2, pmi=False, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mature_layer=None, premature_layer=None, candidate_premature_layers=[], mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, post_softmax=True, **kwargs):
        with torch.no_grad():
            input_text = input_text1 + input_text2
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            # print(len(input_ids[0]))
            prefix_ids = self.tokenizer(input_text1, return_tensors="pt").input_ids.to(self.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]
            if  'baseline' in mode:
                
                if 'layer_wise' in mode:
                    
                    outputs = self.model(input_ids, output_hidden_states=True)
                    hidden_layers = outputs[-1]
                    log_probs = []
                    # print('len(hidden_layers):', len(hidden_layers))
                    for ind in range(1,len(hidden_layers)):
                        output = hidden_layers[ind].squeeze(0)
                        output = self.model.lm_head(output)
                        output = output.log_softmax(-1)  # logits to log probs

                    # skip tokens in the prompt -- we only care about the answer
                        output = output[prefix_ids.shape[-1] - 1: -1, :]
                        # get logprobs for each token in the answer
                        log_prob = output[range(output.shape[0]), continue_ids].sum().item()
                        log_probs.append(log_prob)
                    
                else:
                    outputs = self.model(input_ids)[0].squeeze(0)
                    outputs = outputs.log_softmax(-1)  # logits to log probs

                    # skip tokens in the prompt -- we only care about the answer
                    outputs = outputs[prefix_ids.shape[-1] - 1: -1, :]
                    # get logprobs for each token in the answer
                    log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()

        return log_probs,  None