import json
import re

import jsonlines


data = list(jsonlines.open('/root/autodl-tmp/projs/AutoIF/output/6.5_output.jsonl', 'r')) + list(jsonlines.open('/root/autodl-tmp/projs/AutoIF/output_seed_0/6.5_output.jsonl', 'r'))

final_output = {"type": "conversation", "instances": []}

for datapoint in data:
    instruction = datapoint["instruction"]
    instruction = instruction[0].upper()+ instruction[1:]
    
    for qgroup in datapoint['queries'].items():
        full_query = qgroup[0]
        all_responses = qgroup[1]
        actual_query = re.findall(r'\[Query\](.*)$', full_query, re.DOTALL)[0].strip()
        actual_query = actual_query[0].upper() + actual_query[1:]
        
        if actual_query.endswith("?"):
            inputs = actual_query+" "+instruction+"."
        elif actual_query.endswith("."):
            inputs = actual_query+" "+instruction+"."
        else:
            inputs=actual_query+". "+instruction+"."

        for response in all_responses:
            messages = [
                {"role": "user", "content": inputs},
                {"role": "assistant", "content": response.strip('\n')}
            ]
            final_output["instances"].append({"messages": messages})
            
json.dump(final_output, open('/root/autodl-tmp/projs/AutoIF/output/lmflow_unverified_data.json', 'w'), indent=4)