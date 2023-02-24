# Overview

- structure

    - ```preprocess```：数据集预处理的文件夹。

    - ```api```：调用openai ai的api的文件夹。

    - ```eval```：评测的文件夹。

    - ```data```：中间过程文件存储的文件夹。

- workflow

    1. 在```preprocess```文件夹中编写预处理数据集的代码，并设定好任务名称task_name，最后将处理好的数据集命名为```origin.jsonl```保存在```data/task_name```文件夹下。

    2. 在```api```文件夹下调用脚本，得到返回的数据，返回的数据会以```response.jsonl```的命名保存在```data/task_name```文件夹下。

    3. 调用```eval```文件夹下的评测脚本，结果数据会以```result.json```的命名保存在```data/task_name```文件夹下。

# api

## completion任务

- 参数编写参考链接：https://platform.openai.com/docs/api-reference/completions

- 重点关注以下几个参数

    1. temperature：默认值1，取值范围0~2。数值越大生成的随机性越强。该参数与top_n之间一般只会同时使用一个。

    2. top_p：默认值1，取值范围0~1。和temperature类似，也是一种采样方式。例子：0.1表示只会考虑头部的10%的token。该参数与temperature之间一般只会同时使用一个。

    3. n：默认值1。控制返回的结果数量。

    4. logprobs

    5. presence_penalty：默认值0，取值范围-2~2。生成token时，若token出现过，则对它进行惩罚。数值越大，则惩罚力度越大，模型越容易返回新的话题。

    6. frequency_penalty：默认值0，取值范围-2~2。生成token时，会根据token出现的频率对它进行惩罚。数值越大，则惩罚力度越大，模型约容易重复同样的句子。

- question

    1. resence_penalty和frequency_penalty参数看起来有点类似，区别在哪？

    2. logprobs参数的返回值的意思？

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
