import json 
from tqdm import tqdm 
from transformers import AutoModelForMaskedLM, AutoTokenizer 
import argparse
import torch
import numpy as np

random_seed = 0 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help='Path to sql2text model')
parser.add_argument('--input_queries_aug', type=str, help='Path to augmented queries', required=True)
parser.add_argument('--input_queries_train', type=str, help='Path to train queries',)
parser.add_argument('--output_file', type=str, help='Path to output json', required=True)
parser.add_argument('--num_beams', type=int, help='Number of beams', default=5)
parser.add_argument('--num_sequences', type=int, help='Number of sequences to generate', default=5)
args = parser.parse_args()

device = 'cuda'

model = AutoModelForMaskedLM.from_pretrained(args.model_path).to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(args.model_path)

def collate_fn(batch):
    inputs = [' '.join(item['sql_encoding'][1:-1]).replace('Ġ','') for item in batch]
    ret = tokenizer(inputs, return_tensors='pt', padding=True, max_length=512).to(device)
    return ret
    
num_beams = args.num_beams 
num_return_sequences = args.num_sequences 
input_aug_file = args.input_queries_aug
input_train_file = args.input_queries_train 
output_file = args.output_file

data = json.load(open(input_aug_file))
data = data + json.load(open(input_train_file))

save_data = []

batch_size = 32
for i in tqdm(range(0, len(data), batch_size), total=len(data)//batch_size):
    items = data[i: min(i+batch_size, len(data))]
#for item in tqdm(data, total=len(data)):
    batch = collate_fn(items)
    outputs = model.generate(**batch,
                            do_sample=True,
                            max_length=128,
                            top_k=100,
                            top_p=0.95, 
                            num_return_sequences=num_return_sequences, 
                            return_dict_in_generate=True, 
                            output_scores=True)
    samples = tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)

    for j in range(0, len(samples), num_return_sequences):
        questions = set(samples[j:j+num_return_sequences])
        for q in questions:
            new_item = items[j//num_return_sequences].copy()
            new_item['question'] = q
            save_data.append(new_item)

with open(output_file, 'w') as f:
    json.dump(save_data, f, indent=4, sort_keys=True)
