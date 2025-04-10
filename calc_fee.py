import argparse

import jsonlines


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the fee for a given prompt and completion.')
    parser.add_argument('--file', type=str, required=True, help='Path to the JSON file containing the response data.')
    args = parser.parse_args()
    
    all_data = list(jsonlines.open(args.file))
    num_prompt_tokens = 0
    num_completion_tokens = 0
    
    for data in all_data:
        num_prompt_tokens += data['response']['body']['usage']['prompt_tokens']
        num_completion_tokens += data['response']['body']['usage']['completion_tokens']
            
    print(f'Number of prompt tokens: {num_prompt_tokens}')
    print(f'Number of completion tokens: {num_completion_tokens}')
    fee_prompt = num_prompt_tokens / 1000 * 0.001
    fee_completion = num_completion_tokens / 1000 * 0.004
    print(f'Fee for prompt: {fee_prompt:.6f}')
    print(f'Fee for completion: {fee_completion:.6f}')
    print(f'Total fee: {fee_prompt + fee_completion:.6f}')
    print(f'Number of requests: {len(all_data)}')
    
    