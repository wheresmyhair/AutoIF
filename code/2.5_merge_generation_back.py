import jsonlines
from tqdm import tqdm

original = []
with jsonlines.open('/root/autodl-tmp/projs/AutoIF/output/eval_func_rft_volc_deepseek.jsonl', 'r') as reader:
    for obj in reader:
        original.append(obj)

original_idx_mapping = {}
for i, obj in enumerate(original):
    original_idx_mapping[obj['custom_id']] = i

output_deepseek = []

with jsonlines.open('/root/autodl-tmp/projs/AutoIF/output/eval_func_volc_deepseek_results.jsonl', 'r') as reader:
    for obj in tqdm(reader, total=len(original)):
        input_data = original[original_idx_mapping[obj['custom_id']]]
        if obj['custom_id'] != input_data['custom_id']:
            raise ValueError(f"Custom ID mismatch: {obj['custom_id']} != {input_data['custom_id']}")
        out_deepseek = {
            "instruction_id": int(input_data['custom_id'].split('_')[0]),
            "seed": int(input_data['custom_id'].split('_')[2]),
            "prompt": input_data['body']['messages'],
            "responses": [obj['response']['body']['choices'][0]['message']],
        }
        output_deepseek.append(out_deepseek)
        
merged_output = []

all_instruction_ids = list(set([data['instruction_id'] for data in output_deepseek]))
all_instruction_ids.sort()

for instruction_id in all_instruction_ids:
    output = {
        "instruction_id": instruction_id,
        "prompt": [],
        "responses": [None]*5,
    }
    for data in output_deepseek:
        if data['instruction_id'] == instruction_id:
            output['prompt'] = data['prompt']
            output['responses'][data['seed']] = data['responses']
    merged_output.append(output)
    
with jsonlines.open('/root/autodl-tmp/projs/AutoIF/output/eval_func_volc_deepseek_merged.jsonl', 'w') as writer:
    for obj in tqdm(merged_output, total=len(merged_output)):
        writer.write(obj)