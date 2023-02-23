import json
filename = "data/example_requests_to_parallel_process.jsonl"
n_requests = 10_000
jobs = [{"model": "text-embedding-ada-002", "input": str(x) + "\n"} for x in range(n_requests)]
with open(filename, "w") as f:
    for job in jobs:
        json_string = json.dumps(job)
        f.write(json_string + "\n")