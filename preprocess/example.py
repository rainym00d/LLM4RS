import os
import json

work_dir = f"{os.environ['HOME']}/LLM4RS"
os.chdir(work_dir)

task_name = "example"
if not os.path.exists(f"data/{task_name}"):
    os.makedirs(f"data/{task_name}")

filename = f"data/{task_name}/request.jsonl"
n_requests = 10

# * json中的true和false用True和False代替，null用None代替
jobs = [
    {
        "model": "text-davinci-003",
        "prompt": "Q:What is your name\nA:",
        "max_tokens": 100,
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop": "\n",
    }
    for x in range(n_requests)
]

# * 注意是jsonl格式
with open(filename, "w") as f:
    for job in jobs:
        json_string = json.dumps(job)
        f.write(json_string + "\n")
