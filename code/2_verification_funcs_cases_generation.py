import random
import re
import os
import sys
import copy
import nltk
import numpy as np
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    RetryError
)
import logging
import signal
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import jsonlines

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from apis.res_generator import generate

"""Concat seed and augmented instructions for generation, then generation Eval funcs"""

seed_instructions = [each.strip() for each in open("./sample_data/seed_instruction.txt").readlines()]
augment_instructions_processed = [each.strip() for each in open("./sample_data/augment_instructions.txt").readlines()]


prompt_template = """You are an expert for writing evaluation functions in Python to evaluate whether a response strictly follows an instruction.
Here is the instruction: {instruction}
Please write a Python function named `evaluate` to evaluate whether an input string `response` follows this instruction. If it follows, simply return True, otherwise return False.
Please response with a single JSON includes the evaluation function in the key `func`, and a list of three test cases in the key `cases`, which includes an input in the key `input` and an expected output in the key `output` in (true, false).
Here is an example of output JSON format: {{"func": JSON_STR(use only \\n instead of \n), "cases": [{{"input": str, "output": str}}]}}."""


outputs = []
for instruction in seed_instructions + augment_instructions_processed:
    prompt = prompt_template.format(instruction=instruction)
    outputs.append({
        "prompt": prompt,
        "instruction": instruction
    })


with jsonlines.open("./output/eval_func_rft.jsonl", "w") as f:
    for each in outputs:
        f.write(each)
        
with jsonlines.open("./output/eval_func_rft_volc_deepseek.jsonl", "w") as f:
    for idx, output in enumerate(outputs):
        for i in range(5):
            output_final = {
                "custom_id": f'{idx}_seed_{i}', 
                "body": {
                    "messages": [{"role": "user", "content": output['prompt']}],
                    "max_tokens": 16000,
                    "seed": i,
                }
            }
            f.write(output_final)


'''
Please TODO:

please generate K verification functions for each sample by supervision model in eval_func_rft.jsonl

'''
# jsonlines_file = f'./output/eval_func_rft_deepseek.jsonl'
# print('start at:')
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# for output in tqdm(outputs):
#     for i in tqdm(range(5)):
#         reasoning, answser = generate(prompt=output['prompt'], seed=i)
#         with jsonlines.open(jsonlines_file, mode='a') as writer:
#             writer.write({
#                 'seed': i,
#                 'prompt': output['prompt'],
#                 'instruction': output['instruction'],
#                 'reasoning': reasoning,
#                 'answer': answser
#             })
            
# print('finish at:')
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))