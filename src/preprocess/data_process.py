import json
import pandas as pd
from typing import Dict, Tuple

from const import Const as C


def get_data(data_path: str) -> pd.DataFrame:
    """Get data from the data path.

    Args:
        data_path (str): The path where data is.

    Returns:
        pd.DataFrame: The Dataframe of loaded data.
    """
    data = pd.read_csv(data_path, delimiter="\t", usecols=C.DATA_COL_NAME_LIST)
    # * change data type from str to list
    for col_name in C.DATA_EVAL_COL_NAME_LIST:
        data[col_name] = data[col_name].apply(lambda x: eval(x))

    return data

def get_datamaps(datamaps_path: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Get data maps from data path. 

    Args:
        datamaps_path (str): The path where data is.

    Returns:
        Tuple[Dict[int, str], Dict[str, int]]: id2item mapping and item2id mapping.
    """
    with open(datamaps_path, "r") as f:
        datamaps = json.load(f)
    id2item_dict = {int(k): v for k, v in datamaps["id2item_dict"].items()}
    item2id_dict = {k: int(v) for k, v in datamaps["item2id_dict"].items()}

    return id2item_dict, item2id_dict

def process_data(data: pd.DataFrame, id2item_dict: Dict[int, str], task: str) -> pd.DataFrame:
    """Process data based on task, and it make following step easy.

    Args:
        data (pd.DataFrame): The Dataframe of data.
        id2item_dict (Dict[int, str]): id2item mapping.
        task (str): Task name.

    Returns:
        pd.DataFrame: The Dataframe of processed data.
    """
    # * history id to history name, history name to history str
    data["name_history"] = data["itemid_history"].apply(lambda x: [id2item_dict[i] for i in x])
    data["prompt_history"] = data["name_history"].apply(lambda x: ", ".join(x))

    if task == C.POINT_WISE:
        # * candidate id to candidate name
        data["name_candidate"] = data["itemid_candidate"].apply(lambda x: [id2item_dict[i] for i in x])
        # * gather candidate name to a list
        prompt_candidate_list = []
        for _, row in data.iterrows():
            prompt_candidate_list.append(row.name_candidate)
        data["prompt_candidate_list"] = prompt_candidate_list
    elif task == C.PAIR_WISE:
        # * pair candidate id to pair candidate name
        data["pair_name_candidate"] = data["pair_itemid_candidate"].apply(lambda x: [(id2item_dict[i[0]], id2item_dict[i[1]]) for i in x])
        # * create neg pair like (neg, neg)
        new_pair_itemid_candidate = []
        for _, row in data.iterrows():
            tmp_new_pair_itemid_candidate = []
            for i in range(len(row.itemid_candidate)):
                if i == row.pos_target_index:
                    continue
                for j in range(len(row.itemid_candidate)):
                    if j == row.pos_target_index or i >= j:
                        continue
                    tmp_new_pair_itemid_candidate.append([row.itemid_candidate[i], row.itemid_candidate[j]])
            new_pair_itemid_candidate.append(tmp_new_pair_itemid_candidate)
        data["new_pair_itemid_candidate"] = new_pair_itemid_candidate
        data["new_pair_name_candidate"] = data["new_pair_itemid_candidate"].apply(lambda x: [(id2item_dict[i[0]], id2item_dict[i[1]]) for i in x])
        # * gather candidate name to a list
        prompt_candidate_list = []
        new_prompt_candidate_list = []
        for _, row in data.iterrows():
            prompt_candidate_list.append([f"(A) {pos} (B) {neg}" for pos, neg in row.pair_name_candidate])
            new_prompt_candidate_list.append([f"(A) {pos} (B) {neg}" for pos, neg in row.new_pair_name_candidate])
        data["prompt_candidate_list"] = prompt_candidate_list
        data["new_prompt_candidate_list"] = new_prompt_candidate_list
    elif task == C.LIST_WISE:
        # * candidate id to candidate name
        data["name_candidate"] = data["itemid_candidate"].apply(lambda x: [id2item_dict[i] for i in x])
        # * gather candidate name to a list
        prompt_candidate_list = []
        for _, row in data.iterrows():
            choices = [f"({chr(ord('A') + i)}) {name}" for i, name in enumerate(row.name_candidate)]
            prompt_candidate_list.append([" ".join(choices)])
        data["prompt_candidate_list"] = prompt_candidate_list

    return data