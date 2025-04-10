import json

from datasets import Dataset

data = []


with open('./output/query_score_filter.jsonl', 'r', encoding='utf-8') as file:
    
    for dat in file:
        d = json.loads(dat)
        data.append(d)
        
with open('./output_seed_0/query_score_filter.jsonl', 'r', encoding='utf-8') as file:
    
    for dat in file:
        d = json.loads(dat)
        data.append(d)


processed_data = []
for item in data:
    
    item['query'] = item['query'][0].upper() + item['query'][1:]
    item['instruction'] = item['instruction'][0].upper()+ item['instruction'][1:]
    if "?" in item['query']:
        inputs = item['query']+" "+item['instruction']+"."

    elif "." in item['query']:
        inputs = item['query']+" "+item['instruction']+"."
    else:
        inputs=item['query']+". "+item['instruction']+"."

    response = item['response'].strip('\n')
    messages = [
        {"role": "user", "content": inputs},
        {"role": "assistant", "content": response}
    ]


    processed_data.append({"messages": messages})

print(len(processed_data))

hf_data = Dataset.from_list(processed_data)
hf_data.push_to_hub('wheresmyhair/ultrachat-autoif', split='train')