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

MODEL_NAME = 'facebook/bart-base'
PROJECT_NAME = 'SqlToText'
OUT_DIR = '../../models/sql-to-text/snowball/'
TRAIN_PATH = '../../data/sql-to-text/snowball/jsons/train_spider_snowball.json'
DEV_PATH = '../../data/sql-to-text/snowball/jsons/dev_snowball.json'

n_epochs = 100
batch_size = 64
lr = 3e-5
weight_decay = 1e-2

run_name = 'run-snowball-sql-to-text'
os.environ['WANDB_PROJECT'] = PROJECT_NAME

training_args = TrainingArguments(
    output_dir=f'{OUT_DIR}/{run_name}',
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
    report_to='wandb',
    run_name=run_name,
)

class Spider(Dataset):
    def __init__(self, path):
        super().__init__()
        with open(path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def compute_bleu(outputs):
    logits = outputs.predictions 
    pred_ids = np.argmax(logits[0], axis=-1)
    pred_ids[pred_ids == -100] = 1
    label_ids = outputs.label_ids
    label_ids[label_ids == -100] = 1
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

def collate_fn(batch):
    labels = [item['question'] for item in batch]
    inputs = [item['proc_query'] for item in batch]
    ret = tokenizer(inputs, return_tensors='pt', padding=True, max_length=512)
    ret['labels'] = tokenizer(labels, return_tensors='pt', padding=True, max_length=512)['input_ids']
    return ret

def main():
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    train = Spider(TRAIN_PATH)
    dev = Spider(DEV_PATH)

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

if __name__ == '__main__':
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['WANDB_PROJECT']='sql2text'
    main()
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
