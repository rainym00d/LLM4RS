# Overview

- 文件组织

    - preprocess：数据集预处理的文件夹

    - api：调用openai ai的api的文件夹

    - eval：评测的文件夹

    - data：中间过程文件存储的文件夹。

- 依次执行preprocess，api和eval，每一个任务都在data中创建一个文件夹，文件夹下有三个文件：origin.jsonl、response.jsonl和result，分别对应三个处理步骤产生的结果文件

# api

## completion任务

- 参数编写参考链接：https://platform.openai.com/docs/api-reference/completions

- 重点关注以下几个参数

    1. temperature

    2. top_p

    3. n

    4. logprobs

    5. presence_penalty

    6. frequency_penalty

# tokenizer查询表

``` python
MODEL_TO_ENCODING: dict[str, str] = {
    # text
    "text-davinci-003": "p50k_base",
    "text-davinci-002": "p50k_base",
    "text-davinci-001": "r50k_base",
    "text-curie-001": "r50k_base",
    "text-babbage-001": "r50k_base",
    "text-ada-001": "r50k_base",
    "davinci": "r50k_base",
    "curie": "r50k_base",
    "babbage": "r50k_base",
    "ada": "r50k_base",
    # code
    "code-davinci-002": "p50k_base",
    "code-davinci-001": "p50k_base",
    "code-cushman-002": "p50k_base",
    "code-cushman-001": "p50k_base",
    "davinci-codex": "p50k_base",
    "cushman-codex": "p50k_base",
    # edit
    "text-davinci-edit-001": "p50k_edit",
    "code-davinci-edit-001": "p50k_edit",
    # embeddings
    "text-embedding-ada-002": "cl100k_base",
    # old embeddings
    "text-similarity-davinci-001": "r50k_base",
    "text-similarity-curie-001": "r50k_base",
    "text-similarity-babbage-001": "r50k_base",
    "text-similarity-ada-001": "r50k_base",
    "text-search-davinci-doc-001": "r50k_base",
    "text-search-curie-doc-001": "r50k_base",
    "text-search-babbage-doc-001": "r50k_base",
    "text-search-ada-doc-001": "r50k_base",
    "code-search-babbage-code-001": "r50k_base",
    "code-search-ada-code-001": "r50k_base",
    # open source
    "gpt2": "gpt2",
}
```
