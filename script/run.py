import time
import subprocess
from xpflow import Xp


class MyArgs(Xp):
    model = ["text-davinci-003"]
    domain = ["Movie"]
    task = ["list"]
    no_instruction = False
    example_num = [1]
    begin_index = 5
    end_index = 505
    api_key = "YOUR API-KEY"
    max_requests_per_minute = 2000
    max_tokens_per_minute = 80000
    max_attempts = 100
    proxy = None

if __name__ == "__main__":
    for i, args in enumerate(MyArgs()):
        cmd = (
            f"python src/main.py "
            f"--model {args.model} "
            f"--domain {args.domain} "
            f"--task {args.task} "
            f"--example_num {args.example_num} "
            f"--begin_index {args.begin_index} "
            f"--end_index {args.end_index} "
            f"--api_key {args.api_key} "
            f"--max_requests_per_minute {args.max_requests_per_minute} "
            f"--max_tokens_per_minute {args.max_tokens_per_minute} "
            f"--max_attempts {args.max_attempts} "
            f"--proxy {args.proxy} "
        )
        if args.no_instruction:
            cmd += "--no_instruction "
        start = time.time()
        print("-"*20)
        print(args)
        p = subprocess.Popen(cmd, shell=True)
        p.wait()
        end = time.time()
        print(f"Cmd: {i} finished! time cost(s): {end - start:.2f}")
        print("-"*20)