from transformers import AutoModelForMaskedLM, AutoTokenizer
import argparse
import json 
import os 
from pathlib import Path
from tqdm import tqdm 
import random
import torch
from process_sql import tokenize, get_sql, get_schema_from_data

from tensor2struct.utils import registry, pcfg
grammar_config = {
    "name": "spiderv2",
    "include_literals": True,
    "end_with_from": True,
    "infer_from_conditions": True
}
grammar = registry.construct("grammar", grammar_config)

# Path to spider files
train_input = '../../data/spider/train_spider.json'
dev_input = '../../data/spider/dev.json'
tables = '../../data/spider/tables.json'
database = '../../data/spider/database/'
IGNORE_DEV_INDICES = [175, 926, 927]

train_data_config = {
        "name": "spider",
        "paths": [train_input],
        "tables_paths": [tables],
        "db_path": database
}

random_seed = 1618 # 1000x golden ratio
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# This is just a dummy object used for accessing schema related information
train_data = registry.construct("dataset", train_data_config)

with open(tables, 'r') as f:
    all_db_data = json.load(f)
    all_db_data = {x['db_id']: x for x in all_db_data}

def l2s_encoding(sql_text, db_id):
    # Convert text SQL to the dict version like spider
    db_data = all_db_data[db_id]
    db_data['columns'] = [x[1] for x in db_data['column_names']]
    db_data = [db_data]
    sql = get_sql(get_schema_from_data(db_data), sql_text)

    # Parse the dict SQL using L2S and unparse it again
    # https://github.com/berlino/tensor2struct-public/blob/e7492d41af342263d7d0ffe55720abb6b7a4d683/tensor2struct/datasets/spider.py#L136-L137
    astree = grammar.parse(sql)
    proc_query = pcfg.unparse(grammar.ast_wrapper, train_data.schemas[db_id], astree)
    return proc_query

preproc_dict = {
    'none': lambda x:x['query'],
    'l2s': lambda x: l2s_encoding(x['query'], x['db_id'])[1],
}

def is_value(x):
    if x in ['t', 'f', 'true', 'false'] or (x[0] == x[-1] == '"'):
        return True 
    try:
        x = float(x)
        return True 
    except Exception as e:
        try:
            x = float(x[1:-1])
            return True 
        except Exception as e:
            return False

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', required=True, help='Path to HF model. Should also contain saved tokenizer')
parser.add_argument('--root-dir-input', required=True, help='Path to groups root dir')
parser.add_argument('--root-dir-output', required=True, help='Path to output root dir')
parser.add_argument('--num-sequences', default=5, help='Number of sequences per query')
parser.add_argument('--preproc', default='none', choices=['none', 'l2s'], help='Preprocessing for SQL')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for inference')
parser.add_argument('--dev-frac', default=0.3, type=float, help='Fraction of train as dev')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForMaskedLM.from_pretrained(args.model_path)

root_dir = Path(args.root_dir_input)
output = Path(args.root_dir_output)
if not os.path.exists(output):
    os.makedirs(output, exist_ok=True)

preproc = preproc_dict[args.preproc]
batch_size = args.batch_size

for group in range(1, 5):
    with open(root_dir / f'group_{group}/perturbed_train_queries.json') as f:
        train = json.load(f)

    queries = []
    for x in train:
        x['query'] = x['query'].replace('intersect', 'INTERSECT')
        queries.append(preproc(x))

    all_samples = []
    for i in tqdm(range(0, len(queries), batch_size)):
        items = queries[i: min(i+batch_size, len(queries))]
        batch = tokenizer(items, return_tensors='pt', padding=True, max_length=512)
        outputs = model.generate(**batch, do_sample=True,
            max_length=128,
            top_k=100,
            top_p=0.95,
            num_return_sequences=args.num_sequences,
            return_dict_in_generate=True,
            output_scores=True
        )
        samples = tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)
        all_samples.extend(samples)

    save_samples = []
    for i, q in enumerate(queries):
        for j in range(args.num_sequences):
            query_toks = tokenize(train[i]['query'])
            query_toks_no_value = ['value' if is_value(tok) else tok for tok in query_toks]
            save_samples.append({
                'db_id': train[i]['db_id'],
                'query': train[i]['query'],
                'query_toks': query_toks,
                'query_toks_no_value': query_toks_no_value,
                'question': all_samples[i*args.num_sequences+j],
            })

    if not os.path.exists(output / f'group_{group}'):
        os.makedirs(output / f'group_{group}', exist_ok=True)

    with open(output / f'group_{group}/finetune_train.json', 'w') as f:
        json.dump(save_samples, f, indent=4, sort_keys=True)

    if args.dev_frac < 0 or args.dev_frac >= 1:
        dev = save_samples 
    else:
        num_dev = int(args.dev_frac * len(save_samples)) + 1
        num_dev = min(num_dev, len(save_samples))
        dev = random.sample(save_samples, num_dev)

    with open(output / f'group_{group}/finetune_dev.json', 'w') as f:
        json.dump(dev, f, indent=4, sort_keys=True)
