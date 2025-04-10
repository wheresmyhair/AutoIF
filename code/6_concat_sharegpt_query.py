import jsonlines
import json
import random
import re
import os
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

from datasets import load_dataset

random.seed(42)




filter_results=[]

with jsonlines.open("./output/back_trans_filter.jsonl", "r") as f:
    for each in f:
        filter_results.append(each)


data = load_dataset(
    'wheresmyhair/ultrachat_autoif_promptonly', 
    cache_dir='/root/autodl-tmp/datasets', 
    split='train', 
    num_proc=8
)

inputs = []
for instruction in tqdm(filter_results):
    ins_queries = data.select(random.sample(range(len(data)), 135))
    for q in ins_queries:
        prompt = f"Please answer the query strictly following the instruction.\n[instruction] {instruction['instruction']}\n[Query] {q['query']}"
        item = copy.deepcopy(instruction)
        item['prompt'] = prompt
        item['hf_id'] = q['id']
        inputs.append(item)
        # import pdb
        # pdb.set_trace()

total_idx = 0
with jsonlines.open("./output/ultrachat_query.jsonl", "w") as f:
    for each in inputs:
        for i in range(3):
            output_final = {
                "custom_id": f'{total_idx}_queryid_{each["hf_id"]}_instid_{each["id"]}_seed_{i}', 
                "body": {
                    "messages": [{"role": "user", "content": each['prompt']}],
                    "max_tokens": 16000,
                    "seed": i,
                }
            }
            f.write(output_final)
            total_idx += 1


'''
Please TODO:

Please use supervision model perform RFT to generate k Responses for each query
'''