from typing import Dict, List
from string import Template

from const import Const as C


def generate_prompt(fmt_dict: Dict[str, str], domain: str, task: str, no_instruction:bool, example_data: List[Dict[str, str]]) -> str:
    """Generrate prompt based on domain and task.

    Args:
        fmt_dict (Dict[str, str]): The data to be filled into the prompt template.
        domain (str): The domain name.
        task (str): Task name.
        no_instruction (bool): Use instruction or not.
        example_data (List[Dict[str, str]]): The example data to be filled into the prompt template.

    Returns:
        str: Prompt.
    """
    # * create instruction
    if no_instruction:
        instruction = ""
    else:
        instruction = instruction_dict[task].substitute(**domain_word_dict[domain])
    # * create example
    example_list = []
    if example_data:
        for example_fmt_data in example_data:
            example_list.append(example_dict[task].substitute(**{**example_fmt_data, **domain_word_dict[domain]}))
    example = "\n".join(example_list)
    # * create query
    query = query_dict[task].substitute(**{**fmt_dict, **domain_word_dict[domain]})
    # * prompt = instruction + example + query
    prompt =  "\n".join([instruction, example, query])

    return prompt

domain_word_dict = {
    "Movie": {
        "domain_word_1": "movie",
        "domain_word_2": "movies",
        "domain_word_3": "watched",
        "domain_word_4": "watching",
    },
    "Music": {
        "domain_word_1": "music",
        "domain_word_2": "music",
        "domain_word_3": "listened to",
        "domain_word_4": "listening",
    },
    "Book": {
        "domain_word_1": "book",
        "domain_word_2": "books",
        "domain_word_3": "read",
        "domain_word_4": "reading",
    },
    "News": {
        "domain_word_1": "news",
        "domain_word_2": "news",
        "domain_word_3": "read",
        "domain_word_4": "reading",
    },
    "agnostic": {
        "domain_word_1": "",
        "domain_word_2": "items",
        "domain_word_3": "interacted with",
        "domain_word_4": "interaction",
    },
}

example_dict = {
    C.LIST_WISE: Template(
        "Input: Here is the $domain_word_4 history of a user: $item_history. Based on this history, please rank the following candidate $domain_word_2: $candidate_item\nOutput: The answer index is $answer."
    ),
    C.PAIR_WISE: Template(
        "Input: Here is the $domain_word_4 history of a user: $item_history. Based on this history, would this user prefer $candidate_item_A or $candidate_item_B? Answer Choices: $candidate_item\nOutput: The answer index is $answer."
    ),
    C.POINT_WISE: Template(
        "Input: Here is the $domain_word_4 history of a user: $item_history. Based on this history, please predict the user's rating for the following item: $candidate_item (1 being lowest and 5 being highest)\nOutput: $answer."
    ),
}

query_dict = {
    C.LIST_WISE: Template(
        "Input: Here is the $domain_word_4 history of a user: $item_history. Based on this history, please rank the following candidate $domain_word_2: $candidate_item\nOutput: The answer index is"
    ),
    C.PAIR_WISE: Template(
        "Input: Here is the $domain_word_4 history of a user: $item_history. Based on this history, would this user prefer $candidate_item_A or $candidate_item_B? Answer Choices: $candidate_item\nOutput: The answer index is"
    ),
    C.POINT_WISE: Template(
        "Input: Here is the $domain_word_4 history of a user: $item_history. Based on this history, please predict the user's rating for the following item: $candidate_item (1 being lowest and 5 being highest)\nOutput:"
    ),
}

instruction_dict = {
    C.LIST_WISE: Template(
        "You are a $domain_word_1 recommender system now."
    ),
    C.PAIR_WISE: Template(
        "You are a $domain_word_1 recommender system now."
    ),
    C.POINT_WISE: Template(
        "You are a $domain_word_1 recommender system now."
    ),
}