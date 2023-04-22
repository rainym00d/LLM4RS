import json
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple

from const import Const as C


Y_pred = Dict[int, List[int]]

def get_y_true(data: pd.DataFrame, begin_index: int, end_index: int) -> Dict[int, List[int]]:
    """Get ground truth from data.

    Args:
        data (pd.DataFrame): The Dataframe of data.
        begin_index (int): The begin index of data.
        end_index (int): The end index of data.

    Returns:
        Dict[int, List[int]]: The ground truth from begin index to end index.
    """
    y_true = data[begin_index:end_index].pos_target_index.to_dict()
    if not isinstance(list(y_true.values())[0], list):
        y_true = {k: [v] for k, v in y_true.items()}

    return y_true

def process_result(response_path: str, exception_path: str, model: str, task: str, candidate_num: int) -> Dict[int, List[int]]:
    """Process result based on model and task, and then get predict label.

    Args:
        response_path (str): The path where response file is saved.
        exception_path (str): The path where exception file is saved.
        model (str): The model name.
        task (str): The task name.
        candidate_num (int): The number of candidate for the given data.

    Returns:
        Dict[int, List[int]]: The predict label.
    """
    if model in C.COMPLETION_MODEL_LIST:
        y_pred, exception_response = process_davinci_result(response_path=response_path, task=task, candidate_num=candidate_num)
    elif model in C.CHAT_MODEL_LIST:
        y_pred, exception_response = process_chatgpt_result(response_path=response_path, task=task, candidate_num=candidate_num)

    # * write error data to exception file
    print(f"exception response num: {len(exception_response)}")
    with open(exception_path, "w") as f:
        for response in exception_response:
            json_string = json.dumps(response)
            f.write(json_string + "\n")

    return y_pred

def process_davinci_result(response_path: str, task: str, candidate_num: int) -> Tuple[Dict[int, List[int]], list]:
    if task == C.LIST_WISE:
        y_pred, exception_response = process_davinci_list_result(response_path=response_path, candidate_num=candidate_num)
    elif task == C.PAIR_WISE:
        y_pred, exception_response = process_davinci_pair_result(response_path=response_path, candidate_num=candidate_num)
    elif task == C.POINT_WISE:
        y_pred, exception_response = process_davinci_point_result(response_path=response_path, candidate_num=candidate_num)

    return y_pred, exception_response

def process_davinci_list_result(response_path: str, candidate_num: int) -> Tuple[Y_pred, list]:
    result = {}
    exception_response = []
    with open(response_path, "r") as f:
        for line in f:
            content = json.loads(line)
            task_id = content["task_id"]
            group_id = int(task_id.split("-")[0])
            try:
                response = content["response"]
                # * process 0-shot
                if response["choices"][0]["logprobs"] is not None:
                    top_logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]
                    res = [ord(k) - ord("A")for k in top_logprobs.keys()]
                # * process few-shot
                else:
                    text = response["choices"][0]["text"].strip()
                    text = ''.join(c for c in text if c.isalpha())
                    res = [ord(c) - ord("A") for c in text.upper()]
                # * check result is valid or not
                if len(res) != candidate_num or 2 * sum(res) != candidate_num * (candidate_num - 1):
                    raise
                else:
                    result[group_id] = res
            except:
                exception_response.append(content)
    
    return result, exception_response

def process_davinci_pair_result(response_path: str, candidate_num: int) -> Tuple[Y_pred, list]:
    result = {}
    exception_response = []
    counter = defaultdict(lambda: [0] * candidate_num)
    target_index_dict = {}
    with open(response_path, "r") as f:
        for line in f:
            content = json.loads(line)
            task_id = content["task_id"]
            group_id = int(task_id.split("-")[0])
            pos = content["pos"]
            # target = content["target"]
            target_index_dict[group_id] = content["target_index"]
            try:
                response = content["response"]
                text = response["choices"][0]["text"].strip()
                text = ''.join(c for c in text if c.isalpha())
                option = ord(text) - ord("A")
                counter[group_id][pos[option]] += 1
            except:
                exception_response.append(content)
    # * gather all result
    for k, v in counter.items():
        large_count = 0
        eq_count = 0
        for i, score in enumerate(v):
            if v[target_index_dict[k]] < score:
                large_count += 1
            elif v[target_index_dict[k]] == score:
                eq_count += 1
        result[k] = [
            [-1 if j != large_count + i else target_index_dict[k] for j in range(candidate_num)]
            for i in range(eq_count)
        ]

    return result, exception_response

def process_davinci_point_result(response_path: str, candidate_num: int) -> Tuple[Y_pred, list]:
    result = {}
    exception_response = []
    counter = defaultdict(lambda: [0] * candidate_num)
    target_index_dict = {}
    with open(response_path, "r") as f:
        for line in f:
            content = json.loads(line)
            task_id = content["task_id"]
            group_id = int(task_id.split("-")[0])
            candidate_id = int(task_id.split("-")[1])
            target_index_dict[group_id] = content["target_index"]
            try:
                response = content["response"]
                text = response["choices"][0]["text"].strip()
                # if text == "3.5":
                #     score = 3.5
                # else:
                score = ''.join(c for c in text if c.isdigit())
                score = int(score)
                if score < 1 or score > 5:
                    raise
                counter[group_id][candidate_id] = score
            except:
                exception_response.append(content)
    # * gather all result
    for k, v in counter.items():
        large_count = 0
        eq_count = 0
        for i, score in enumerate(v):
            if v[target_index_dict[k]] < score:
                large_count += 1
            elif v[target_index_dict[k]] == score:
                eq_count += 1
        result[k] = [
            [-1 if j != large_count + i else target_index_dict[k] for j in range(candidate_num)]
            for i in range(eq_count)
        ]

    return result, exception_response

def process_chatgpt_result(response_path: str, task: str, candidate_num: int) -> Tuple[Y_pred, list]:
    if task == C.LIST_WISE:
        y_pred, exception_response = process_chatgpt_list_result(response_path=response_path, candidate_num=candidate_num)
    elif task == C.PAIR_WISE:
        y_pred, exception_response = process_chatgpt_pair_result(response_path=response_path, candidate_num=candidate_num)
    elif task == C.POINT_WISE:
        y_pred, exception_response = process_chatgpt_point_result(response_path=response_path, candidate_num=candidate_num)

    return y_pred, exception_response

def process_chatgpt_list_result(response_path: str, candidate_num: int) -> Tuple[Y_pred, list]:
    result = {}
    exception_response = []
    with open(response_path, "r") as f:
        for line in f:
            content = json.loads(line)
            task_id = content["task_id"]
            group_id = int(task_id.split("-")[0])
            try:
                response = content["response"]
                # * only process few-shot, since chatGPT can not get logprob
                text = response["choices"][0]["message"]["content"].strip()
                text = ''.join(c for c in text if c.isalpha())
                res = [ord(c) - ord("A") for c in text.upper()]
                # * check result is valid or not
                if len(res) != candidate_num or 2 * sum(res) != candidate_num * (candidate_num - 1):
                    raise
                else:
                    result[group_id] = res
            except:
                exception_response.append(content)
    
    return result, exception_response

def process_chatgpt_pair_result(response_path: str, candidate_num: int) -> Tuple[Y_pred, list]:
    result = {}
    exception_response = []
    counter = defaultdict(lambda: [0] * candidate_num)
    target_index_dict = {}
    with open(response_path, "r") as f:
        for line in f:
            content = json.loads(line)
            task_id = content["task_id"]
            group_id = int(task_id.split("-")[0])
            pos = content["pos"]
            target_index_dict[group_id] = content["target_index"]
            try:
                response = content["response"]
                text = response["choices"][0]["message"]["content"].strip()
                text = ''.join(c for c in text if c.isalpha())
                option = ord(text) - ord("A")
                counter[group_id][pos[option]] += 1
            except:
                exception_response.append(content)
    # * gather all result
    for k, v in counter.items():
        large_count = 0
        eq_count = 0
        for i, score in enumerate(v):
            if v[target_index_dict[k]] < score:
                large_count += 1
            elif v[target_index_dict[k]] == score:
                eq_count += 1
        result[k] = [
            [-1 if j != large_count + i else target_index_dict[k] for j in range(candidate_num)]
            for i in range(eq_count)
        ]

    return result, exception_response

def process_chatgpt_point_result(response_path: str, candidate_num: int) -> Tuple[Y_pred, list]:
    result = {}
    exception_response = []
    counter = defaultdict(lambda: [0] * candidate_num)
    target_index_dict = {}
    with open(response_path, "r") as f:
        for line in f:
            content = json.loads(line)
            task_id = content["task_id"]
            group_id = int(task_id.split("-")[0])
            candidate_id = int(task_id.split("-")[1])
            target_index_dict[group_id] = content["target_index"]
            try:
                response = content["response"]
                text = response["choices"][0]["message"]["content"].strip()
                score = ''.join(c for c in text if c.isdigit())
                score = int(score)
                if score < 1 or score > 5:
                    raise
                counter[group_id][candidate_id] = score
            except:
                exception_response.append(content)
    # * gather all result
    for k, v in counter.items():
        large_count = 0
        eq_count = 0
        for i, score in enumerate(v):
        #     if i == target_index_dict[k]:
        #         continue
            if v[target_index_dict[k]] < score:
                large_count += 1
            elif v[target_index_dict[k]] == score:
                eq_count += 1
        result[k] = [
            [-1 if j != large_count + i else target_index_dict[k] for j in range(candidate_num)]
            for i in range(eq_count)
        ]

    return result, exception_response