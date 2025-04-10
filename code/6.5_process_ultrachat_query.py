import json
import re

import jsonlines


def parse_instruction_id_from_custom_id(id: str):
    pattern = r'instid_(\d+)_'
    match = re.search(pattern, id)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Invalid custom ID format: {id}")


# Matching response
generated = []
with jsonlines.open("./output/ultrachat_query.jsonl", "r") as f:
    for each in f:
        generated.append(each)
custom_id_mapping = {}
for idx, each in enumerate(generated):
    custom_id_mapping[each['custom_id']] = idx

with jsonlines.open('./output/ultrachat_query_res.jsonl', 'r') as f:
    for each in f:
        try:
            idx = custom_id_mapping[each['custom_id']]
            generated[idx]['res'] = each['response']['body']['choices'][0]['message']['content']
        except KeyError:
            print(f"KeyError: {each['custom_id']}")
            continue

# apply back to the meta
data = []
with jsonlines.open("./output/back_trans_filter.jsonl", "r") as f:
    for each in f:
        each['queries'] = {}
        data.append(each)
instruction_id_mapping = {}
for idx, each in enumerate(data):
    instruction_id_mapping[each['id']] = idx
        

for generated_datapoint in generated:
    instid = int(parse_instruction_id_from_custom_id(generated_datapoint['custom_id']))
    inst_idx = instruction_id_mapping[instid]
    prompt = generated_datapoint['body']['messages'][0]['content']
    if prompt not in data[inst_idx]['queries'].keys():
        data[inst_idx]['queries'][prompt] = []
    data[inst_idx]['queries'][prompt].append(generated_datapoint['res'])

with jsonlines.open("./output/6.5_output.jsonl", "w") as f:
    for each in data:
        f.write(each)