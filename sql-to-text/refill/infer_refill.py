import json 
from tqdm import tqdm 
from transformers import AutoModelForMaskedLM, AutoTokenizer 
import argparse
import torch
import numpy as np
from math import ceil
from collections import OrderedDict
from utils import read_spider_json, read_templates, get_sql_to_templates

from bleu import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

random_seed = 0 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help='Path to sql2text model')
parser.add_argument('--input_json', type=str, help='Path to augmented queries', required=True)
parser.add_argument('--nbrs_json', type=str, help='Path to json containing retrieved nbrs')
parser.add_argument('--templates_json', type=str, help='Path to json containing question templates')
parser.add_argument('--tables_json', type=str, help='Path to tables', required=True)
parser.add_argument('--output_json', type=str, help='Path to output json', required=True)
parser.add_argument('--num_beams', type=int, help='Number of beams', default=5)
parser.add_argument('--num_sequences', type=int, help='Number of sequences to generate', default=10)
parser.add_argument('--num_nbrs', type=int, help='Number of retrieved nbrs to use', default=5)
args = parser.parse_args()

device = 'cuda'

model = AutoModelForMaskedLM.from_pretrained(args.model_path).to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

def collate_fn(item):
    query = item[0]
    templates = [el['masked_template'] for el in item[1][0]['templates']]
    inputs = [query] + [query + f' <temp> {item} </temp>' for item in templates]
    ret = tokenizer(inputs, return_tensors='pt', padding=True, max_length=512).to(device)
    return ret
    
num_beams = args.num_beams 
num_return_sequences = args.num_sequences 
input_json = args.input_json
output_json = args.output_json
tables_json = args.tables_json
templates_json = args.templates_json
nbrs_json = args.nbrs_json
num_nbrs = args.num_nbrs

templates = read_templates(json.load(open(templates_json))) # ques to template
sql_to_templates = get_sql_to_templates(json.load(open(nbrs_json)), templates, num_nbrs=num_nbrs) # sql to templates


data = read_spider_json(input_json, tables_json, add_tables=False, sql_to_templates=sql_to_templates)
save_data = []

print('\nstarting inference\n')
for item in tqdm(data):
    batch = collate_fn(item)
    
    outputs = model.generate(
                    **batch, 
                    max_length=128, 
                    num_beams=5,  
                    num_return_sequences=1, 
                    early_stopping=True,
                    return_dict_in_generate=True,
                    output_scores=True
                )

    samples = tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)
    samples_scores = outputs['sequences_scores']
    
    questions = samples
    ques_scores = samples_scores.cpu()
    argmax_ques = np.argmax(ques_scores)
    best_ques = questions[argmax_ques]

    templates = [''] + item[1][0]['templates']

    result = OrderedDict()
    result['db_id'] = item[1][0]['db_id']
    result['query'] = item[1][0]['query']
    result['proc_query'] = item[0]
    result['references'] = list(set([el['question'] for el in item[1]]))
    result['hypotheses'] = questions
    result['templates'] = templates
    result['best_hyp'] = best_ques
    save_data.append(result)

all_references = []
best_bleu_hypotheses = []
best_hypotheses = []
for item in save_data:
    all_references.append([word_tokenize(el) for el in item['references']])
    best_hypotheses.append(word_tokenize(item['best_hyp']))
    best_hyp_bleu = sentence_bleu(all_references[-1], 
                                  best_hypotheses[-1], 
                                  smoothing_function=SmoothingFunction().method3) 
    bleus = [sentence_bleu(all_references[-1], 
                           word_tokenize(el), 
                           smoothing_function=SmoothingFunction().method3) 
            for el in item['hypotheses']]
    best_bleu_idx = np.argmax(bleus)
    best_bleu = bleus[best_bleu_idx]
    hyp_with_best_bleu = item['hypotheses'][best_bleu_idx]
    best_bleu_hypotheses.append(word_tokenize(hyp_with_best_bleu))
    item['best_hyp_bleu'] = best_hyp_bleu
    item['best_oracle_bleu'] = best_bleu
    item['hyp_with_best_oracle_bleu'] = hyp_with_best_bleu
    
corpus_bleu_best_hyp = corpus_bleu(all_references, best_hypotheses, smoothing_function=SmoothingFunction().method3)
corpus_best_bleu = corpus_bleu(all_references, best_bleu_hypotheses, smoothing_function=SmoothingFunction().method3)

[item.update({'corpus_bleu': corpus_bleu_best_hyp}) for item in save_data]
[item.update({'corpus_best_bleu_oracle': corpus_best_bleu}) for item in save_data]

with open(output_json, 'w') as f:
    json.dump(save_data, f, indent=4)
