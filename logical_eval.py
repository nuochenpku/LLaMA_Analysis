# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/metrics.py
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/utilities.py

import re
import os
import json
import random
import torch
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm, trange
import argparse
import pandas as pd
from transformers import AutoConfig
import ssl
import urllib.request
import zipfile

from model import LLaMA_Analysis

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 7
COT_FLAG = True
DEBUG = False
ANSWER_TRIGGER = "So the answer is"



def split_multi_answer(ans, sep=',', close=True):

    """Splits string of all reference answers into a list of formatted answers"""

    answers = ans.strip().split(sep)
    split_answers = []
    for a in answers[:-1]:
        a = a.strip()
        if len(a):
            if close:  # add a period after all answers
                # if a[-1] != '.':
                #     split_answers.append(a + '.')
                # else:
                split_answers.append(a)
            else:
                split_answers.append(a)

    return split_answers


def format_best(best_ans, close=True):

    """Formats best answer to match format of reference answers"""

    best = best_ans.strip()
    if close:
        if best[-1] != '.':
            best = best + '.'
    return best

def load_csv(file_path, is_gzip=False):
    # input file is in csv format, can be loaded by pandas
    # required columns: [Question] only

    open_func = open if not is_gzip else gzip.open
    list_data = []
    with open_func(file_path, 'r') as f:
        df = pd.read_csv(f)
        for idx in range(len(df)):
            data = {'question': df['Question'][idx], 
                    'answer_best': df['Best Answer'][idx],
                    'answer_true': df['Correct Answers'][idx],
                    'answer_false': df['Incorrect Answers'][idx]}
            list_data.append(data)

    return list_data


def load_tsv(file_path, is_gzip=False):
    # input file is in csv format, can be loaded by pandas
    # required columns: [Question] only

    open_func = open if not is_gzip else gzip.open
    list_data = []
    with open_func(file_path, 'r') as f:
        
        datas = f.readlines()
    for data in datas[1:]:
        split_data = data.split('\t')

        if len(split_data[2].split(',')) >1:
        
            data = {'question': split_data[0], 
                    'answer_true': split_data[1],
                    'answer_false': split_data[2]}
            list_data.append(data)

    return list_data


def download_url(url: str, folder='folder'):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition('/')[2]
    file = file if file[0] == '?' else file.split('?')[0]
    path = os.path.join(folder, file)
    if os.path.exists(path):
        print(f'File {file} exists, use existing file.')
        return path

    print(f'Downloading {url}')
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, 'wb') as f:
        f.write(data.read())

    return path

def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = answer
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer

def create_demo_text():
    context, question, answer = [], [], []
    
    context.append("In rheumatoid arthritis, the body' s immune system misfunctions by attacking healthy cells in the joints causing the release of a hormone that in turn causes pain and swelling. This hormone is normally activated only in reaction to injury or infection. A new arthritis medication will contain a protein that inhibits the functioning of the hormone that causes pain and swelling in the joints.")
    question.append("The statements above, if true, most strongly support which one of the following conclusions?")
    answer.append("A patient treated with the new medication for rheumatoid arthritis could sustain a joint injury without becoming aware of it.")

    context.append("Patient: Pharmacists maintain that doctors should not be permitted to sell the medicine that they prescribe because doctors would then be tempted to prescribe unnecessary medicines in order to earn extra income. But pharmacists have a financial interest in having a monopoly on the sale of prescription medicines, so their objection to the sale of medicines by doctors cannot be taken seriously.")
    question.append("The patient's argument proceeds by")
    answer.append("attempting to discredit a position by questioning the motives of the proponents of that position")
    

    context.append("Paula will visit the dentist tomorrow morning only if Bill goes golfing in the morning. Bill will not go golfing unless Damien agrees to go golfing too. However, Damien has decided not to go golfing. Ttherefore, Paula will not be visiting the dentist tomorrow morning.")
    question.append("The pattern of reasoning displayed above most closely parallels which of the following?")
    answer.append("Kevin will wash his car tomorrow only if Brittany has to go visit her grandmother. Unless Aunt Susan has to run errands, Brittany will not have to go visit her grandmother. Since Aunt Susan does not have to run errands, Kevin will not wash his car tomorrow.")
    # question.append("whaling is a subclass of <mask>.")
    # answer.append("hunting")

    # Concatenate demonstration examples ...
    demo_text = 'Please answer the logical question based on the passage.' + '\n\n'
    for i in range(len(question)):
        demo_text += "P: " + context[i] + "\nQ: " + question[i] + "\nA: " + answer[i] + "\n\n"
    return demo_text


def build_prompt(input_text):
    demo = create_demo_text()
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    return input_text_prompt

def build_prompt_with_answer(question, answer):
    demo = create_demo_text()
    input_text_prompt = demo + "Q: " + question + "\n" + "A: " + answer
    return input_text_prompt

def build_prompt_and_answer(context, input_text, answer):
    demo = create_demo_text()
    input_text_prompt = demo + "P: " + context +  "\nQ: " + input_text + "\n" + "A:"
    continue_text = " " + answer
    return input_text_prompt, continue_text


def Math_Cals(scores_true, scores_false, ref_best):
    """Given model scores for true / false reference answers, calculates MC scores"""
    scores = {}
    scores['max'] = max(scores_true)
    scores['diff'] = max(scores_true) - max(scores_false)
    scores['scores-true'] = scores_true
    scores['scores-false'] = scores_false

    # compute MC1: 1vFalse -- best correct answer vs all false answers
    max_false = max(scores_false)
    if scores_true[0] > max_false:
        scores['MC1'] = 1.0
    else:
        scores['MC1'] = 0.0
    return scores

def MC_calcs(scores_true, scores_false, ref_true=None, ref_best=None):

    """Given model scores for true / false reference answers, calculates MC scores"""
    scores = {}
    scores['max'] = max(scores_true)
    scores['diff'] = max(scores_true) - max(scores_false)
    scores['scores-true'] = scores_true
    scores['scores-false'] = scores_false

    # compute MC1: 1vFalse -- best correct answer vs all false answers
    max_false = max(scores_false)
    if scores_true[0] > max_false:
        scores['MC1'] = 1.0
    else:
        scores['MC1'] = 0.0

    # compute MC3: 1vFalse -- each correct answer vs all false answers
    max_false = max(scores_false)
    onevall = sum(np.array(scores_true) > max_false) / float(len(scores_true))
    scores['MC3'] = onevall

    # compute MC2: normalized probability mass for correct answers
    probs_true = np.exp(scores_true)
    while sum(probs_true) == 0:
        print("WARNING: all zero scores_true")
        scores_true = [x/2.0 for x in scores_true]
        probs_true = np.exp(scores_true)
    probs_false = np.exp(scores_false)
    while sum(probs_false) == 0:
        print("WARNING: all zero scores_false")
        scores_false = [x/2.0 for x in scores_false]
        probs_false = np.exp(scores_false)

    probs_true = probs_true / (sum(probs_true) + sum(probs_false))
    
    # check nan
    if np.isnan(sum(probs_true)):
        scores['MC2'] = 0.0
        print(f"WARNING: nan in probs_true: sum(probs_true)={sum(probs_true)}, sum(probs_false)={sum(probs_false)}")
    else:
        scores['MC2'] = sum(probs_true)

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=80)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="./tfqa")
    parser.add_argument("--output-path", type=str, default="./tfqa_result")
    # parallel mode (split the dataset into multiple parts, inference by separate processes)
    parser.add_argument("--early-exit-layers", type=str, default="-1")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--do-rating", action="store_true")
    parser.add_argument("--gpt3-config", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--relative_top", type=float, default=0.0)
    parser.add_argument("--relative_top_value", type=float, default=-1000.0)
    parser.add_argument("--hidden_layers", type=int, default=80)
    parser.add_argument("--layer_wise", action="store_true")
    parser.add_argument("--attention", action="store_true")
    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device

    # Get test file
    with open(args.data_path, 'r') as f:
        list_data_dict = json.load(f)
    
    
    

    if args.debug:
        list_data_dict = list_data_dict[:10]
    
    if args.parallel:
        chunk_size = len(list_data_dict) // args.total_shard
        list_data_dict = list_data_dict[args.shard_id * chunk_size: (args.shard_id + 1) * chunk_size]
    
    llm = LLaMA_Analysis(model_name, device, num_gpus, args.hidden_layers, args.max_gpu_memory)
    stop_word_list = ["Q:"]
    llm.set_stop_words(stop_word_list)
    early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]
    if len(early_exit_layers) == 1:
        print("MODE: naive decoding from the last layer", flush=True)
        if args.layer_wise:
            mode = 'baseline_layer_wise'
        elif args.attention:
            mode = 'baseline_attention'
        else:
            mode = "baseline"
        mature_layer = None
        premature_layer = None
        candidate_premature_layers = None
    answers = []
    
    result_dict = {'question': [], 'model_scores': [], 'total_mc1': 0.0, 'total_mc2': 0.0, 'total_mc3': 0.0}
    
    config = AutoConfig.from_pretrained(model_name)
        # print(config)
    # config.num_hidden_layers
    
    if args.layer_wise:
        all_results = {f'Layer_{i+1}_lm_head':{'total_mc1':0.0} for i in range(config.num_hidden_layers)}
    else:
        all_results = {f'Layer_{i+1}_attention':{'add': 0.0, 'sub': 0.0, 'mul': 0.0, 'div': 0.0, 'mix_ops_2': 0.0, 'mix_ops_3': 0.0, "min_ops_brackets":0.0} for i in range(config.num_hidden_layers)}
    # all_results = {f'Layer_{i+1}_lm_head':{'total_mc1': 0.0, 'total_mc2': 0.0, 'total_mc3': 0.0} for i in range(config.num_hidden_layers-70)}
    with torch.no_grad():
        for ind, sample in enumerate(tqdm(list_data_dict)):
            # reference answers
            # sample = json.loads(sample)
            ref_true = sample['answers'][sample['label']]
            # ref_best = format_best(sample['truth_answer'])
            # # ref_true = split_multi_answer(sample['answer_true'])
            # ref_false = split_multi_answer(sample['answer_false'])
            ref_false = sample['answers']
            ref_false.remove(ref_true)
            # print(ref_true)
            # print(ref_false)
            assert ref_true not in ref_false
            # print('answer choices')
            # print(ref_best)
            # print(ref_true)
            # print(ref_false)
            scores_true = []
            scores_false = []

            generate_kwargs = dict(max_new_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty, mode=mode, mature_layer=mature_layer, premature_layer=premature_layer, candidate_premature_layers=candidate_premature_layers, relative_top=args.relative_top, relative_top_value=args.relative_top_value, post_softmax=False)

            # for temp_ans in ref_true:
                # append the current answer choice to the prompt
            prompt, answer = build_prompt_and_answer(sample['context'], sample['question'], ref_true)
            # print(prompt, answer)
            log_probs, c_dist = llm.lm_score(prompt, answer, **generate_kwargs)
            scores_true.append(log_probs)
            # print(ref_false)
               

            for temp_ans in ref_false:
                # append the current answer choice to the prompt
                prompt, answer = build_prompt_and_answer(sample['context'], sample['question'], temp_ans)
                
                log_probs, c_dist = llm.lm_score(prompt, answer, **generate_kwargs)
                scores_false.append(log_probs)

            if args.layer_wise or args.attention:
                scores = []
                # print(len(log_probs))
                for ind_ in range(len(log_probs)):
                    
                    score_true, score_false = np.array(scores_true)[:,ind_], np.array(scores_false)[:,ind_]

                    score = Math_Cals(list(score_true), list(score_false), ref_true)
                    
                    # if np.isnan(score['MC1']) or np.isnan(score['MC2']) or np.isnan(score['MC3']):
                    #     print(ind_)
                    #     import ipdb; ipdb.set_trace()
                        
                        
                    if args.layer_wise:
                        key_ = f'Layer_{ind_+1}_lm_head'
                    else:
                        key_ = f'Layer_{ind_+1}_attention'

                    # update total scores
                    all_results[key_]['total_mc1'] += score['MC1']
                    # all_results[key_]['total_mc2'] += score['MC2']
                    # all_results[key_]['total_mc3'] += score['MC3']

    # Average the scores
    for key, value in all_results.items():
        all_results[key]['total_mc1'] /= len(list_data_dict)
        # all_results[key]['total_mc3'] /= len(list_data_dict)

    # Print the final scores, separated by ', '
        print(f'{key} MC1: \n{all_results[key]}')

    import pandas as pd    
    df = pd.DataFrame(all_results)
    if not args.attention:
        file_name = args.output_path + f'output-path_layer_wise.xlsx'
    else:
        file_name = args.output_path + f'output-path_layer_wise_atten.xlsx'
    
    df.to_excel(file_name, index=False)