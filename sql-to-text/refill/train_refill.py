import os
import json
import random
import numpy as np 
import torch 
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    BartForConditionalGeneration,
    BartConfig,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import load_metric
from bleu import corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from utils import SPECIAL_TOKENS

from edit_distance import SequenceMatcher
from pprint import pprint

MODEL_NAME = 'facebook/bart-base'
PROJECT_NAME = 'SqlToText'
OUTPUT_DIR = 'models/sql-to-text/refill'
TRAIN_PATH = 'data/sql-to-text/refill/jsons/train_spider_template_improved_blec_th_70.json'
TRAIN_NBRS = 'data/sql-to-text/refill/jsons/train_train_nbrs.json'
DEV_PATH = 'data/sql-to-text/refill/jsons/dev_template_improved_blec_th_70.json'
DEV_NBRS = 'data/sql-to-text/refill/jsons/val_train_nbrs.json'
TABLES = 'data/spider/tables.json'

n_epochs = 100
batch_size = 64
lr = 3e-5
weight_decay = 1e-2

tables = TABLES
tables = json.load(open(tables))
db_tables = {}
for table in tables:
    db_tables[table['db_id']] = table 
    db_tables[table['db_id']]['columns'] = [x[1] for x in table['column_names']]


run_name = 'template-improved-blec-th-70-aug-simp'
os.environ['WANDB_PROJECT'] = PROJECT_NAME

training_args = TrainingArguments(
    output_dir=f'{OUTPUT_DIR}/{run_name}',
    overwrite_output_dir=True,
    num_train_epochs=n_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_total_limit=1,
    prediction_loss_only=False,
    do_train=True,
    do_eval=True,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    weight_decay=weight_decay,
    learning_rate=lr,
    warmup_steps=100,
    lr_scheduler_type='linear',
    metric_for_best_model='bleu',
    load_best_model_at_end=True,
    fp16=True,
    eval_accumulation_steps=8,
    run_name=run_name,
    report_to='wandb',
)

class Spider(Dataset):
    def __init__(self, data_path, nbrs_path, ques_to_temp, training):
        super().__init__()
        self.training = training
        self.data = json.load(open(data_path))
        self.ques_to_temp = ques_to_temp
        nbrs = json.load(open(nbrs_path))
        self.nbrs = {}
        for q in nbrs:
            sanitized_q = self._sanitize_query(q)
            self.nbrs[sanitized_q] = nbrs[q]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx].copy()
        #if self.training:
        if True:
            toss = np.random.uniform()
            if toss < 0.30: # no template
                data_item['bart_input'] = data_item['proc_query']
            
            elif toss < 0.60: # same example template
                data_item['bart_input'] = data_item['proc_query'] \
                                        + f" <temp> {data_item['template']} </temp>"
            
            elif toss < 0.90: # template with low wed
                item_query = data_item['query']
                item_ques = data_item['question']
                item_temp = data_item['template']
                item_nbrs = self.nbrs.get(item_query,[])
                nbr_questions = [el['question'] for el in item_nbrs]
                nbr_templates = [self.ques_to_temp[el] for el in nbr_questions]
                nbr_templates_dist = [SequenceMatcher(
                                        item_temp.lower().replace('<mask>','').split(), 
                                        el.lower().replace('<mask>','').split()).distance()
                                    for el in nbr_templates
                                    ]
                nbr_templates_dist = [el if el!=0 else 100 for el in nbr_templates_dist]
                if len(item_nbrs) > 0:
                    nearest_dist = min(nbr_templates_dist)
                    nearest_idx = [i for i,el in enumerate(nbr_templates_dist) if el==nearest_dist]
                    nearest_idx = np.random.choice(nearest_idx)
                    nearest_template = nbr_templates[nearest_idx]
                    data_item['bart_input'] = data_item['proc_query'] \
                                            + f" <temp> {nearest_template} </temp>"
                    
                else:
                    data_item['bart_input'] = data_item['proc_query']

            else: # mix masks in same example template
                assert 0.9 < toss < 1
                template = data_item['template'].split()
                num_ins = np.random.randint(2, 5)
                for _ in range(num_ins):
                    ins_idx = np.random.randint(len(template))
                    template.insert(ins_idx, '<mask>')
                template = ' '.join(template)
                data_item['bart_input'] = data_item['proc_query'] \
                                        + f" <temp> {template} </temp>"


        else:
            data_item['bart_input'] = data_item['proc_query'] \
                                    + f" <temp> {data_item['template']} </temp>"
        return data_item

    def _sanitize_query(self, query):
        query = query.replace(")", " ) ")
        query = query.replace("(", " ( ")
        query = ' '.join(query.split())
        query = query.replace('> =', '>=')
        query = query.replace('< =', '<=')
        query = query.replace('! =', '!=')

        query = query.replace('"', "'")
        if query.endswith(";"):
            query = query[:-1]
        for i in [1, 2, 3, 4, 5]:
            query = query.replace(f"t{i}", f"T{i}")
        for agg in ["count", "min", "max", "sum", "avg"]:
            query = query.replace(f"{agg} (", f"{agg}(")
        for agg in ["COUNT", "MIN", "MAX", "SUM", "AVG"]:
            query = query.replace(f"{agg} (", f"{agg}(")
        for agg in ["Count", "Min", "Max", "Sum", "Avg"]:
            query = query.replace(f"{agg} (", f"{agg}(")
        return query


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)

def compute_bleu(outputs):
    logits = outputs.predictions 
    pred_ids = np.argmax(logits[0], axis=-1)
    pred_ids[pred_ids==-100]=1
    label_ids = outputs.label_ids
    label_ids[label_ids==-100]=1
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    pred_toks = [word_tokenize(item) for item in pred_str]
    label_toks = [[word_tokenize(item)] for item in label_str]
    pseudo_label_toks = [word_tokenize(item) for item in label_str]
    bleu = corpus_bleu(label_toks, pred_toks, smoothing_function=SmoothingFunction().method3)
    print()
    randidx = np.random.randint(0,len(pred_str))
    print(f'pred: {pred_str[randidx]}')
    print(f'gold: {label_str[randidx]}')
    print()
    return {'bleu': bleu}

def get_table_names(item):
    table_names = db_tables[item['db_id']]['table_names'].copy()
    random.shuffle(table_names)
    return table_names

def collate_fn(batch):
    labels = [item['question'] for item in batch]
    inputs = [item['bart_input'] for item in batch]
    ret = tokenizer(inputs, return_tensors='pt', padding=True, max_length=512)
    ret['labels'] = tokenizer(labels, return_tensors='pt', padding=True, max_length=512)['input_ids']
    return ret

def get_question_to_template(data):
    ques_to_temp = {}
    for item in data:
        ques_to_temp[item['question']] = item['template']
    return ques_to_temp


def main():
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    train_ques_to_temp = get_question_to_template(json.load(open(TRAIN_PATH)))

    train = Spider(TRAIN_PATH, TRAIN_NBRS, train_ques_to_temp, training=True)
    dev = Spider(DEV_PATH, DEV_NBRS, train_ques_to_temp, training=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=dev,
        data_collator=collate_fn,
        compute_metrics=compute_bleu,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)]
    )

    result = trainer.train()
    trainer.save_model()
    trainer.log_metrics('train', result.metrics)
    trainer.save_metrics('train', result.metrics)
    trainer.save_state()
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == '__main__':
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['WANDB_PROJECT']='sql2text'
    main()
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
