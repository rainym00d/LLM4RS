import json
from typing import Dict, List

from postprocess import Metric
from const import Const as C


def eval_result(task: str, result_path: str, y_true: Dict[int, List[int]], y_pred: Dict[int, List[int]], topk: List[int]) -> List[Dict[str, float]]:
    """Evaluate result and write result to file.

    Args:
        task (str): The task name.
        result_path (str): The path where result file is saved.
        y_true (Dict[int, List[int]]): _description_
        y_pred (Dict[int, List[int]]): _description_
        topk (List[int]): The list of number to control top-k evaluation.

    Returns:
        List[Dict[str, float]]: Evaluation result.
    """
    # * delete the key which is not in y_true
    del_keys = []
    for k in y_true.keys():
        if y_pred.get(k, None) is None:
            del_keys.append(k)
    for k in del_keys:
        y_true.pop(k, None)
    # * filter and rank topk list
    real_topk = []
    lower_bound = 1
    if task == C.PAIR_WISE or task == C.POINT_WISE:
        upper_bound = len(list(y_pred.values())[0][0])
    else:
        upper_bound = len(list(y_pred.values())[0])
    for k in topk:
        if not isinstance(k, int):
            continue
        if k < lower_bound or k > upper_bound:
            continue
        real_topk.append(k)
    real_topk = sorted(list(set(real_topk)))
    # * eval
    if task == C.POINT_WISE or task == C.PAIR_WISE:
        result_list = [
            {f"NDCG@{k}": 0 for k in real_topk},
            {f"MRR@{k}": 0 for k in real_topk},
            {f"Recall@{k}": 0 for k in real_topk},
            {f"Precision@{k}": 0 for k in real_topk},
        ]
        for k in y_true.keys():
            _y_true = {}
            _y_pred = {}
            for i, v in enumerate(y_pred[k]):
                _y_true[i] = y_true[k]
                _y_pred[i] = v
            ndcg = Metric.ndcg_score(_y_true, _y_pred, real_topk)
            mrr = Metric.mrr_score(_y_true, _y_pred, real_topk)
            recall = Metric.recall_score(_y_true, _y_pred, real_topk)
            precision = Metric.precision_score(_y_true, _y_pred, real_topk)
            for k in real_topk:
                result_list[0][f"NDCG@{k}"] += ndcg[f"NDCG@{k}"]
                result_list[1][f"MRR@{k}"] += mrr[f"MRR@{k}"]
                result_list[2][f"Recall@{k}"] += recall[f"Recall@{k}"]
                result_list[3][f"Precision@{k}"] += precision[f"Precision@{k}"]
        for k in real_topk:
            result_list[0][f"NDCG@{k}"] /= len(y_true)
            result_list[1][f"MRR@{k}"] /= len(y_true)
            result_list[2][f"Recall@{k}"] /= len(y_true)
            result_list[3][f"Precision@{k}"] /= len(y_true)
    else:
        result_list = []
        result_list.append(Metric.ndcg_score(y_true, y_pred, real_topk))
        result_list.append(Metric.mrr_score(y_true, y_pred, real_topk))
        result_list.append(Metric.recall_score(y_true, y_pred, real_topk))
        result_list.append(Metric.precision_score(y_true, y_pred, real_topk))
        result_list.append(Metric.map_score(y_true, y_pred, real_topk))
    # * round to 4 decimal places for results
    for result in result_list:
        for k, v in result.items():
            result[k] = round(v, 4)
    # * write result to file
    with open(result_path, "w") as f:
        for result in result_list:
            json_string = json.dumps(result)
            f.write(f"{json_string}\n")

    return result_list