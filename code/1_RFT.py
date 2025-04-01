import jsonlines
import os
import random
import sys

from tqdm import tqdm

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from apis.res_generator import generate

random.seed(0)

# cd ./AutoIF/code

seed_instructions = [each.strip() for each in open("./sample_data/seed_instruction.txt").readlines()]

augment_instruction_prompt = """You are an expert for writing instructions. Please provide 50 different instructions that meet the following requirements:
- Instructions are about the format but not style of a response
- Whether instructions are followed can be easily evaluate by a Python function
Here are some examples of instructions we need:
{seed_instructions}
Do not generate instructions about writing style, using metaphor, or translation. Here are some examples of instructions we do not need:
- Incorporate a famous historical quote seamlessly into your answer
- Translate your answer into Pig Latin
- Use only words that are also a type of food
- Respond with a metaphor in every sentence
- Write the response as if you are a character from a Shakespearean play
Please generate one instruction per line in your response and start each line with '- '.
"""


augment_instructions = augment_instruction_prompt.format(seed_instructions='\n'.join(seed_instructions))

print(augment_instructions)



'''
Tips:

augment_instructions is instructions with rewriting prompt

Please use supervision model rewrite each instruction in augment_instructions for K times, save into a augment_instructions.txt file like seed_instruction.txt


'''


jsonlines_file = f'./sample_data/generated_instructions.jsonl'

for i in tqdm(range(5)):
    reasoning, answser = generate(prompt=augment_instructions, seed=i)
    with jsonlines.open(jsonlines_file, mode='a') as writer:
        writer.write({
            'seed': i,
            'reasoning': reasoning,
            'answer': answser
        })