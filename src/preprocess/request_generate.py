import json
import random
import pandas as pd
import copy
from typing import Dict, List

from preprocess import Prompt
from const import Const as C


def generate_example_data(data: pd.DataFrame, task: str, example_num: int) -> List[Dict[str, str]]:
    """Generate example data.

    Args:
        data (pd.DataFrame): The Dataframe of data.
        task (str): The task name.
        example_num (int): The num of example.

    Returns:
        List[Dict[str, str]]: The example data to be filled into the prompt template.
    """
    example_data = []
    # * take the first five data to make example
    upper = min(example_num, 5)
    defined_point_score_list = [[3], [1, 5], [1, 3, 5], [1, 2, 4, 5], [1, 2, 3, 4, 5]]
    for i, row in data[:upper].iterrows():
        if task == C.LIST_WISE:
            prompt_candidate = row.prompt_candidate_list[0]
            neg_index_list = list(range(len(row.itemid_candidate)))
            neg_index_list.remove(row.pos_target_index)
            random.shuffle(neg_index_list)
            answer_index_list = [row.pos_target_index] + neg_index_list
            answer = " ".join([chr(ord("A") + index) for index in answer_index_list])
        elif task == C.PAIR_WISE:
            pos_index = row.pair_pos_target_index[0]
            neg_index = 1 - pos_index
            if i % 2 == 0:
                candidate_item_A = row.pair_name_candidate[0][pos_index]
                candidate_item_B = row.pair_name_candidate[0][neg_index]
                prompt_candidate = f"(A) {candidate_item_A} (B) {candidate_item_B}" 
                answer = "A"
            else:
                candidate_item_A = row.pair_name_candidate[0][neg_index]
                candidate_item_B = row.pair_name_candidate[0][pos_index]
                prompt_candidate = f"(A) {candidate_item_A} (B) {candidate_item_B}" 
                answer = "B"
        elif task == C.POINT_WISE:
            pos_index = row.pos_target_index
            neg_index_list = list(range(len(row.itemid_candidate)))
            neg_index_list.remove(pos_index)
            random.shuffle(neg_index_list)
            neg_index = neg_index_list[0]

            answer = defined_point_score_list[upper - 1][i]
            if answer <= 3:
                prompt_candidate = row.prompt_candidate_list[neg_index]
            else:
                prompt_candidate = row.prompt_candidate_list[pos_index]

        fmt_dict = {"item_history": row.prompt_history, "candidate_item": prompt_candidate, "answer": answer}
        if task == C.PAIR_WISE:
            fmt_dict["candidate_item_A"] = candidate_item_A
            fmt_dict["candidate_item_B"] = candidate_item_B
        example_data.append(fmt_dict)

    return example_data

def generate_request(data: pd.DataFrame, model: str, domain: str, task: str, begin_index: int, end_index: int, candidate_num: int, no_instruction: bool, example_data: List[Dict[str, str]]) -> List[dict]:
    """Generate request based on model, domain and task.

    Args:
        data (pd.DataFrame): The Dataframe of data.
        model (str): The model name.
        domain (str): The domain name.
        task (str): The task name.
        begin_index (int): The begin index of data.
        end_index (int): The end index of data.
        candidate_num (int): The number of candidate for the given data.
        no_instruction (bool): Use instruction or not.
        example_data (List[Dict[str, str]]): The example data to be filled into the prompt template.

    Returns:
        List[dict]: The request list which will be sent to API.
    """
    request_data = copy.deepcopy(data[begin_index:end_index])
    if model in C.COMPLETION_MODEL_LIST:
        request_list = generate_davinci_request(data=request_data, model=model, domain=domain, task=task, candidate_num=candidate_num, no_instruction=no_instruction, example_data=example_data)
    elif model in C.CHAT_MODEL_LIST:
        request_list = generate_chatgpt_request(data=request_data, model=model, domain=domain, task=task, candidate_num=candidate_num, no_instruction=no_instruction, example_data=example_data)

    return request_list

def generate_davinci_request(data: pd.DataFrame, model: str, domain: str, task: str, candidate_num: int, no_instruction: bool, example_data: List[Dict[str, str]]) -> List[dict]:
    request_list = []
    for i, row in data.iterrows():
        for j, prompt_candidate in enumerate(row.prompt_candidate_list):
            # * general setting
            task_id = f"{i}-{j}"
            target_index = row.pos_target_index
            fmt_dict = {"item_history": row.prompt_history, "candidate_item": prompt_candidate}
            # * setting based on model and task
            if task == C.POINT_WISE:
                target = 1 if j == row.pos_target_index else 0
                if example_data:
                    max_tokens = 20
                    logprobs = None
                    logit_bias = {}
                else:
                    max_tokens = 1
                    logprobs = None
                    logit_bias = {str(16 + i): 50 for i in range(5)}
            elif task == C.PAIR_WISE:
                pos = [row.itemid_candidate.index(row.pair_itemid_candidate[j][0]), row.itemid_candidate.index(row.pair_itemid_candidate[j][1])]
                # * add data to fmt_data
                candidate_item_A, candidate_item_B = row.pair_name_candidate[j]
                fmt_dict["candidate_item_A"] = candidate_item_A
                fmt_dict["candidate_item_B"] = candidate_item_B
                target = row.pair_pos_target_index[j]
                if example_data:
                    max_tokens = 20
                    logprobs = None
                    logit_bias = {}
                else:
                    max_tokens = 1
                    logprobs = None
                    logit_bias = {"32": 50, "33": 50}
            elif task == C.LIST_WISE:
                target = row.pos_target_index
                if example_data:
                    max_tokens = 20
                    logprobs = None
                    logit_bias = {}
                else:
                    max_tokens = 1
                    logprobs = candidate_num
                    logit_bias = {str(32 + i): 50 for i in range(candidate_num)}

            # * create prompt
            prompt = Prompt.generate_prompt(
                fmt_dict=fmt_dict,
                task=task,
                domain=domain,
                no_instruction=no_instruction,
                example_data=example_data,
            )
            # * create request
            request = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
                "top_p": 1,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "stop": "\n",
                "logprobs": logprobs,
                "logit_bias": logit_bias,
            }
            # * append to request_list
            request_list.append({
                "task_id": task_id,
                "target": target,
                "target_index": target_index,
                "request": request,
            })
            if task == C.PAIR_WISE:
                request_list[-1]["pos"] = pos
        # * create request from neg pair like (neg, neg)
        if task == C.PAIR_WISE:
            for j, prompt_candidate in enumerate(row.new_prompt_candidate_list):
                task_id = f"{i}-{j + len(row.prompt_candidate_list)}"
                target_index = row.pos_target_index
                pos = [row.itemid_candidate.index(row.new_pair_itemid_candidate[j][0]), row.itemid_candidate.index(row.new_pair_itemid_candidate[j][1])]
                # * add data to fmt_data
                fmt_dict = {"item_history": row.prompt_history, "candidate_item": prompt_candidate}
                candidate_item_A, candidate_item_B = row.new_pair_name_candidate[j]
                fmt_dict["candidate_item_A"] = candidate_item_A
                fmt_dict["candidate_item_B"] = candidate_item_B
                target = -1
                if example_data:
                    max_tokens = 20
                    logprobs = None
                    logit_bias = {}
                else:
                    max_tokens = 1
                    logprobs = None
                    logit_bias = {"32": 50, "33": 50}
                # * create prompt
                prompt = Prompt.generate_prompt(
                    fmt_dict=fmt_dict,
                    task=task,
                    domain=domain,
                    no_instruction=no_instruction,
                    example_data=example_data,
                )
                # * create request
                request = {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "top_p": 1,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "stop": "\n",
                    "logprobs": logprobs,
                    "logit_bias": logit_bias,
                }
                # * append to request_list
                request_list.append({
                    "task_id": task_id,
                    "target": target,
                    "target_index": target_index,
                    "request": request,
                    "pos": pos,
                })

    return request_list

def generate_chatgpt_request(data: pd.DataFrame, model: str, domain: str, task: str, candidate_num: int, no_instruction: bool, example_data: List[Dict[str, str]]) -> List[dict]:
    request_list = []
    for i, row in data.iterrows():
        for j, prompt_candidate in enumerate(row.prompt_candidate_list):
            # * general setting
            task_id = f"{i}-{j}"
            target_index = row.pos_target_index
            fmt_dict = {"item_history": row.prompt_history, "candidate_item": prompt_candidate}
            # * setting based on model and task
            if task == C.POINT_WISE:
                target = 1 if j == row.pos_target_index else 0
                if example_data:
                    max_tokens = 20
                    logit_bias = {}
                else:
                    max_tokens = 1
                    logit_bias = {str(16 + i): 50 for i in range(5)}
            elif task == C.PAIR_WISE:
                pos = [row.itemid_candidate.index(row.new_pair_itemid_candidate[j][0]), row.itemid_candidate.index(row.new_pair_itemid_candidate[j][1])]
                # * add data to fmt_data
                pos = [row.itemid_candidate.index(row.pair_itemid_candidate[j][0]), row.itemid_candidate.index(row.pair_itemid_candidate[j][1])]
                candidate_item_A, candidate_item_B = row.pair_name_candidate[j]
                fmt_dict["candidate_item_A"] = candidate_item_A
                fmt_dict["candidate_item_B"] = candidate_item_B
                target = row.pair_pos_target_index[j]
                if example_data:
                    max_tokens = 20
                    logit_bias = {}
                else:
                    max_tokens = 1
                    logit_bias = {"32": 50, "33": 50}
            elif task == C.LIST_WISE:
                target = row.pos_target_index
                if example_data:
                    max_tokens = 20
                    logit_bias = {}
                else:
                    raise ValueError("ChatGPT API can't conduct 0 example list task")

            # * create prompt
            prompt = Prompt.generate_prompt(
                fmt_dict=fmt_dict,
                task=task,
                domain=domain,
                no_instruction=no_instruction,
                example_data=example_data,
            )
            # * create request
            request = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
                "top_p": 1,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "stop": "\n",
                "logit_bias": logit_bias,
            }
            # * append to request_list
            request_list.append({
                "task_id": task_id,
                "target": target,
                "target_index": target_index,
                "request": request,
            })
            if task == C.PAIR_WISE:
                request_list[-1]["pos"] = pos
        # * create request from neg pair like (neg, neg)
        if task == C.PAIR_WISE:
            for j, prompt_candidate in enumerate(row.new_prompt_candidate_list):
                task_id = f"{i}-{j + len(row.prompt_candidate_list)}"
                target_index = row.pos_target_index
                fmt_dict = {"item_history": row.prompt_history, "candidate_item": prompt_candidate}
                pos = [row.itemid_candidate.index(row.new_pair_itemid_candidate[j][0]), row.itemid_candidate.index(row.new_pair_itemid_candidate[j][1])]
                # * add data to fmt_data
                candidate_item_A, candidate_item_B = row.new_pair_name_candidate[j]
                fmt_dict["candidate_item_A"] = candidate_item_A
                fmt_dict["candidate_item_B"] = candidate_item_B
                target = -1
                if example_data:
                    max_tokens = 20
                    logit_bias = {}
                else:
                    max_tokens = 1
                    logit_bias = {"32": 50, "33": 50}
                # * create prompt
                prompt = Prompt.generate_prompt(
                    fmt_dict=fmt_dict,
                    task=task,
                    domain=domain,
                    no_instruction=no_instruction,
                    example_data=example_data,
                )
                # * create request
                request = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "top_p": 1,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "stop": "\n",
                    "logit_bias": logit_bias,
                }
                # * append to request_list
                request_list.append({
                    "task_id": task_id,
                    "target": target,
                    "target_index": target_index,
                    "request": request,
                    "pos": pos,
                })

    return request_list

def save_request_file(request_path: str, request_list: list):
    """Save the request list to file.

    Args:
        request_path (str): The path where request list is saved.
        request_list (list): The request list.
    """
    with open(request_path, "w") as f:
        for request in request_list:
            json_string = json.dumps(request)
            f.write(f"{json_string}\n")