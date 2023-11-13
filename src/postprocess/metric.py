import math
from typing import Dict, List


def ndcg_score(y_true: Dict[int, List[int]], y_pred: Dict[int, List[int]], topk: List[int]=[1]):
    assert len(y_true) == len(y_pred)
    if not isinstance(topk, (tuple, list)):
        raise ValueError("topk wrong, it should be tuple or list")

    ndcg_result = {}
    for k in topk:
        ndcg = 0
        for key, value in y_pred.items():
            dcg = 0
            idcg = 0
            for i in range(k):
                if value[i] in y_true[key]:
                    dcg += 1 / (math.log2(i + 2))
            for i in range(min(k,len(y_true[key]))):
                idcg += 1 / (math.log2(i + 2))
            ndcg += dcg / idcg
        ndcg_result[f"NDCG@{k}"] = ndcg / len(y_pred)
    
    return ndcg_result


def mrr_score(y_true: Dict[int, List[int]], y_pred: Dict[int, List[int]], topk: List[int]=[1]):
    assert len(y_true) == len(y_pred)
    if not isinstance(topk, (tuple, list)):
        raise ValueError("topk wrong, it should be tuple or list")

    mrr_result = {}
    for k in topk:
        mrr = 0
        for key, value in y_pred.items():
            for i in range(k):
                if value[i] in y_true[key]:
                    mrr += 1 / (i + 1)
                    break
        mrr_result[f"MRR@{k}"] = mrr / len(y_pred)

    return mrr_result

def recall_score(y_true: Dict[int, List[int]], y_pred: Dict[int, List[int]], topk: List[int]=[1]):
    assert len(y_true) == len(y_pred)
    if not isinstance(topk, (tuple, list)):
        raise ValueError("topk wrong, it should be tuple or list")

    recall_result = {}
    for k in topk:
        recall = 0
        for key, value in y_pred.items():
            count = 0
            for i in range(k):
                if value[i] in y_true[key]:
                    count += 1
            recall += count / len(y_true[key])
        recall_result[f"Recall@{k}"] = recall / len(y_pred)

    return recall_result

def precision_score(y_true: Dict[int, List[int]], y_pred: Dict[int, List[int]], topk: List[int]=[1]):
    assert len(y_true) == len(y_pred)
    if not isinstance(topk, (tuple, list)):
        raise ValueError("topk wrong, it should be tuple or list")
    
    precision_result = {}
    for k in topk:
        precision = 0
        for key, value in y_pred.items():
            rel_count = 0
            for i in range(k):
                if value[i] in y_true[key]:
                    rel_count += 1
            precision += rel_count / k
        precision_result[f"Precision@{k}"] = precision / len(y_pred)

    return precision_result

def map_score(y_true: Dict[int, List[int]], y_pred: Dict[int, List[int]], topk: List[int]=[1]):
    assert len(y_true) == len(y_pred)
    if not isinstance(topk, (tuple, list)):
        raise ValueError("topk wrong, it should be tuple or list")
        
    map_result = {}
    for k in topk:
        mean_average_precision = 0
        for key, value in y_pred.items():
            average_precision = 0
            rel_count = 0
            precision_sum = 0
            for i in range(k):
                if value[i] in y_true[key]:
                    rel_count += 1
                    precision_sum += rel_count / (i + 1)
            if rel_count != 0:
                average_precision = precision_sum / len(y_true[key])
            mean_average_precision += average_precision
        map_result[f"MAP@{k}"] = mean_average_precision / len(y_pred)

    return map_result
