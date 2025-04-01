import jsonlines

generated_data = []
with jsonlines.open("./sample_data/generated_instructions.jsonl", "r") as f:
    for obj in f:
        generated_data.append(obj)

content_processed = []
for data_point in generated_data:
    inst_raw = data_point['answer'].split('- ')
    inst_pp = [inst.strip() for inst in inst_raw if inst]
    content_processed.extend(inst_pp)
    
content_processed = list(set(content_processed))
seed_instructions = [each.strip() for each in open("./sample_data/seed_instruction.txt").readlines()]
content_processed = [inst for inst in content_processed if inst not in seed_instructions]
content_processed = [inst for inst in content_processed if len(inst) > 0]
content_processed.sort()
with open("./sample_data/augment_instructions.txt", "w") as f:
    for line in content_processed:
        f.write(line + "\n")
