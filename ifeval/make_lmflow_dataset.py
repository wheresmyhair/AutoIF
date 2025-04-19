import json

import jsonlines

data = list(jsonlines.open('/root/autodl-tmp/projs/AutoIF/ifeval/samples_ifeval_2025-04-10T05-58-34.550330.jsonl', 'r'))

data_out = {"type": "conversation", "instances": []}

for data_point in data:
    data_out['instances'].append(
        {
            "messages": [
                {"role": "user", "content": data_point['doc']['prompt']},
                {"role": "assistant", "content": data_point['resps'][0][0].strip()}
            ]
        }
    )
    
json.dump(data_out, open('/root/autodl-tmp/projs/AutoIF/ifeval/ifeval_train.json', 'w'), indent=4)