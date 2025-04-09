from datasets import load_dataset


data = load_dataset('stingning/ultrachat', cache_dir='/root/autodl-tmp/datasets', split='train', num_proc=8)
data = data.map(lambda x: {"id": x['id'], "query": x['data'][0]}, num_proc=8, remove_columns=['data']) 
data = data.filter(lambda x: len(x['query']) > 20 and len(x['query']) < 300, num_proc=8)
data = data.shuffle(seed=42)

data.push_to_hub('wheresmyhair/ultrachat_autoif_promptonly')