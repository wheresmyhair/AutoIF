import json

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
import tenacity

random.seed(0)


def extract_bracketed_content(text: str) -> str:
    pattern = r'\[.+?\]'
    matches = re.findall(pattern, text, re.DOTALL)
    return "".join(matches)


filter_results = []
count = 0 
filter_count =0

backtrans_input = []
backtrans_input_ids = {}
with jsonlines.open("./output/eval_func_backtrans_input.jsonl", "r") as f:
    for idx, each in enumerate(f):
        backtrans_input.append(each)
        backtrans_input_ids[each["custom_id"]] = idx
  
with jsonlines.open("./output/results_backtrans.jsonl", "r") as f:
    for each in f:
        custom_id = each["custom_id"]
        if custom_id in backtrans_input_ids:
            backtrans_input[backtrans_input_ids[custom_id]]["results"] = each['response']['body']['choices'][0]['message']['content']
        else:
            print("not find custom id", custom_id)

input_as_idx = {data['body']['messages'][0]['content']: idx for idx, data in enumerate(backtrans_input)}


with open('./output/cross_validation.jsonl', 'r') as file:
    for i,line in enumerate(tqdm(file)):
        line = json.loads(line)
        funcs = line["eval_func"][:3]
        # import pdb
        # pdb.set_trace()


        
      
        instruction = f"""You are an expert in converting the Python eval function code into the corresponding instruction text. I will provide the eval function code. Please strictly follow the code to convert it into the corresponding instruction text. Here's an example: \n\n[["def evaluate(response):\n    return 'e' not in response.lower()", 1.0], ["def evaluate(response):\n    words = response.split()\n    for word in words:\n        if 'e' in word.lower():\n            return False\n    return True", 1.0], ["def evaluate(response):\n    return all('e' not in word.lower() for word in response.split())", 1.0]] \n\n["Answer without using any words that contain the letter 'E'.","Answer with words that do not contain the letter 'E'.","Answer with words that do not contain the letter 'E'."] Please convert the following eval function into instructions stored in a list: \n\n{funcs}"""

        back_instruction = backtrans_input[input_as_idx[instruction]]["results"]

        try:
            back_instruction = json.loads(back_instruction)
        except:
            try:
                back_instruction = json.loads(extract_bracketed_content(back_instruction))
            except:
                print("error in json.loads", back_instruction)
                filter_count+=1
                continue

            
        line["back_instruction"] = back_instruction
        
        filter_results.append(line)
            
        count+=1



print("filter_count",filter_count)

with jsonlines.open("./output/back_trans.jsonl", "w") as f:
    for each in filter_results:
        f.write(each)

'''the example of output format is in ./sample_data/back_trans.jsonl'''

