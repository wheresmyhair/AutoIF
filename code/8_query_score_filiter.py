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

metadata = list(jsonlines.open("./output/query_need_quality_score.jsonl"))
# apply ds result back
matadata_id_idx_mapping = {}
for idx, meta in enumerate(metadata):
    matadata_id_idx_mapping[meta['id']] = idx
ds_res = list(jsonlines.open("./output/score_res.jsonl"))    

for res in ds_res:
    metadata[matadata_id_idx_mapping[res['custom_id']]]['gen'] = res['response']['body']['choices'][0]['message']['content']
    
filter_results = []
print(len(metadata))
for result in tqdm(metadata):
    score = re.findall(r'Score: (\d+?)$', result['gen'])
    score = [int(s) for s in score]
    final_score = np.mean(score) if len(score) > 0 else 0
    if final_score > 8: # quality score
        filter_results.append(result)
print(len(filter_results))



with jsonlines.open("./output/query_score_filter.jsonl", "w") as f:
    for each in filter_results:
        f.write(each)
