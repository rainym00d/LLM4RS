import os
import argparse
import asyncio
import logging

from const import Const as C
from config import Config, get_config
from preprocess import DataProcess, RequestGenerate
from api import process_api_requests_from_file
from postprocess import ResultProcess, Eval


def main(config: Config):
    # * 0. initialize logger, write config to log
    if not os.path.exists(config.save_path):  # * create a folder that holds all the files for this task
        os.makedirs(config.save_path)
    logging.basicConfig(
        filename=config.log_path,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )
    logging.info(config)
    # * 1. load orogin data and datamaps
    data = DataProcess.get_data(data_path=config.data_path)
    id2item_dict, _ = DataProcess.get_datamaps(datamaps_path=config.datamaps_path)
    # * 2. process data
    data = DataProcess.process_data(data, id2item_dict, task=config.task)
    # * 3. create example and request
    example_data = RequestGenerate.generate_example_data(data=data, task=config.task, example_num=config.example_num)
    request_list = RequestGenerate.generate_request(
        data=data,
        model=config.model,
        domain=config.domain,
        task=config.task,
        begin_index=config.begin_index,
        end_index=config.end_index,
        candidate_num=config.candidate_num,
        no_instruction=config.no_instruction,
        example_data=example_data,
    )
    RequestGenerate.save_request_file(request_path=config.request_path, request_list=request_list)
    # * 4. send request
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=config.request_path,
            save_filepath=config.response_path,
            request_url = config.api_url,
            api_key=config.api_key,
            max_requests_per_minute=config.max_requests_per_minute,
            max_tokens_per_minute=config.max_tokens_per_minute,
            token_encoding_name=config.token_encoding_name,
            max_attempts=config.max_attempts,
            proxy=config.proxy,
        )
    )
    # * 5. parse response
    y_true = ResultProcess.get_y_true(data=data, begin_index=config.begin_index, end_index=config.end_index)
    y_pred = ResultProcess.process_result(
        response_path=config.response_path, 
        exception_path=config.exception_path, 
        model=config.model, 
        task=config.task,
        candidate_num=config.candidate_num,
    )
    # * 6. evalution
    Eval.eval_result(
        task=config.task,
        result_path=config.result_path,
        y_true=y_true,
        y_pred=y_pred,
        topk=[1, 2, 3, 4, 5][:config.candidate_num],
    )

if __name__ == "__main__":
    # * parameter setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=C.MODEL_LIST, default=C.MODEL_DEFAULT)
    parser.add_argument("--domain", type=str, choices=C.DOMAIN_LIST, default=C.DOMAIN_DEFAULT)
    parser.add_argument("--task", type=str, choices=C.TASK_LIST, default=C.TASK_DEFAULT)
    parser.add_argument("--no_instruction", action='store_true', default=False)
    parser.add_argument("--example_num", type=int, default=1)
    parser.add_argument("--history_num", type=int, default=5)
    parser.add_argument("--candidate_num", type=int, default=5)
    parser.add_argument("--begin_index", type=int, default=5)
    parser.add_argument("--end_index", type=int, default=505)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--max_requests_per_minute", type=int, default=2000)
    parser.add_argument("--max_tokens_per_minute", type=int, default=10000)
    parser.add_argument("--max_attempts", type=int, default=10)
    parser.add_argument("--proxy", type=str, default=None)
    args, unknown = parser.parse_known_args()
    # * get config
    config = get_config(args)
    print(config)
    # * main function
    main(config)