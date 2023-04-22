import os
import argparse
from datetime import date

from dataclasses import dataclass
from const import Const as C


@dataclass
class Config:
    model: str  # * model name
    domain: str  # * domain name
    task: str  # * task name
    no_instruction: bool  # * use instruction or not
    example_num: int  # * example num used in prompt
    history_num: int  # * history num, default is 5
    candidate_num: int  # * candidate num, default is 5
    
    data_file: str  # * data file
    data_path: str  # * path of data file
    datamaps_path: str  # * path of item/id mapping file
    
    save_path: str  # * save path
    begin_index: int  # * begin index of data
    end_index: int  # * end index of data
    request_path: str  # * path of request file
    
    api_url: str  # * api url corresponding to the model
    response_path: str  # * path of response file
    api_key: str  # * openai api key
    max_requests_per_minute: int  # * max request num per min
    max_tokens_per_minute: int  # * max request token num per min
    token_encoding_name: str  # * tokenizer
    max_attempts: int  # * max attempts num per request
    proxy: str  # * proxy

    result_path: str  # * path of result file
    exception_path: str  # * path of exception file
    log_path: str  # * path of log file


def get_config(args: argparse.Namespace) -> Config:
    init_dict = {}

    init_dict["model"] = args.model
    init_dict["domain"] = args.domain
    init_dict["task"] = args.task
    init_dict["no_instruction"] = args.no_instruction
    init_dict["example_num"] = args.example_num
    init_dict["history_num"] = args.history_num
    init_dict["candidate_num"] = args.candidate_num

    init_dict["data_file"] = f"topk_candidate@{args.candidate_num}_history@{args.history_num}.csv"
    init_dict["data_path"] = os.path.join(C.DATA_ROOT_PATH, args.domain, "LLM", init_dict["data_file"])
    init_dict["datamaps_path"] = os.path.join(C.DATA_ROOT_PATH, args.domain, "LLM", "datamaps.json")

    if args.no_instruction:
        init_dict["save_path"] = os.path.join(C.RESULT_ROOT_PATH, args.domain, args.model, f"no-instruction_example@{args.example_num}_{args.task}_candidate@{args.candidate_num}_history@{args.history_num}")
    else:
        init_dict["save_path"] = os.path.join(C.RESULT_ROOT_PATH, args.domain, args.model, f"example@{args.example_num}_{args.task}_candidate@{args.candidate_num}_history@{args.history_num}")
    init_dict["begin_index"] = args.begin_index
    init_dict["end_index"] = args.end_index
    init_dict["request_path"] = os.path.join(init_dict["save_path"], "request.jsonl")
    
    init_dict["api_url"] = C.URL_DICT[args.model]
    init_dict["response_path"] = os.path.join(init_dict["save_path"], "response.jsonl")
    init_dict["api_key"] = args.api_key
    init_dict["max_requests_per_minute"] = args.max_requests_per_minute
    init_dict["max_tokens_per_minute"] = args.max_tokens_per_minute
    init_dict["token_encoding_name"] = C.TOKENIZER_DICT[args.model]
    init_dict["max_attempts"] = args.max_attempts
    init_dict["proxy"] = args.proxy

    init_dict["result_path"] = os.path.join(init_dict["save_path"], "result.jsonl")
    init_dict["exception_path"] = os.path.join(init_dict["save_path"], "exception.jsonl")
    init_dict["log_path"] = os.path.join(init_dict["save_path"], f"{date.today()}.log")

    return Config(**init_dict)