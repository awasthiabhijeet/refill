import os
import json
import random
import numpy as np 
import torch
import pickle 
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from datasets import load_metric
accuracy_metic = load_metric("accuracy")
f1_metric = load_metric("f1")
from sklearn.metrics import roc_curve, auc, roc_auc_score
from filter_utils import get_augmented_data, get_db_to_queries, SPECIAL_TOKENS

MODEL_NAME = 'roberta-base'
PROJECT_NAME = 'BinaryClassifier'
ALL_DB_VALUES_PATH = 'data/spider/spider_db_values.pkl'    
TRAIN_DATA_PATH = 'data/spider/train_spider.json'
DEV_DATA_PATH = 'data/filter/dev.json'
OUTPUT_DIR = 'models/sql-to-text/refill'

n_epochs = 100
batch_size = 16
lr = 1e-5
weight_decay = 1e-2
run_name = 'filter'

os.environ['WANDB_PROJECT'] = PROJECT_NAME
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
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
    evaluation_strategy='steps',
    eval_steps=500,
    save_strategy='steps',
    save_steps=500,
    weight_decay=weight_decay,
    learning_rate=lr,
    warmup_steps=3000,
    lr_scheduler_type='linear',
    metric_for_best_model='f1',
    load_best_model_at_end=True,
    fp16=True,
    report_to='wandb',
)

class FilterTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        if 'num_augs' in kwargs:
            self.num_augs = kwargs.pop('num_augs')

        super().__init__(*args, **kwargs)
        self.bce_loss = BCEWithLogitsLoss()
        self.xent_loss = CrossEntropyLoss()
        

    def compute_loss(self, model, inputs, return_outputs=False):
        assert 'labels' in inputs, 'No labels were passed for computing loss'
        labels = inputs.pop('labels')
        out = model(**inputs)
        bce_loss = self.bce_loss(out['logits'], labels)

        total_batch_size = labels.shape[0]
        batch_size = total_batch_size//self.num_augs
        labels = labels.reshape(-1, self.num_augs) # [b x C]
        logits = out['logits'].reshape(-1, self.num_augs) # [b x C]
        assert torch.all(labels[:,0] == 1) and torch.all(labels[:,1:]==0)
        xent_loss = -torch.nn.functional.log_softmax(logits, dim=-1)
        xent_loss = xent_loss[:,0] # [b]
        xent_loss = xent_loss.mean()
        loss = bce_loss + xent_loss
        out['loss'] = loss
        return (loss, out) if return_outputs else loss

class BCSpider(Dataset):
    def __init__(self, path, all_db_values, training):
        super().__init__()
        self.data = json.load(open(path))
        self.data_len = len(self.data)
        self.all_db_values = all_db_values
        self.training = training
        if self.training:
            self.db_to_queries = get_db_to_queries(self.data)
        else:
            self.num_augs = len(self.data[0])
        # self.db_to_queries only required during training

    def __len__(self):
        if self.training:
            return self.data_len
        else:
            return self.data_len * self.num_augs

    def __getitem__(self, idx):
        if self.training:
            original_data = self.data[idx]
            # contains original data, and augmented data in a list of dicts
            augmented_data = get_augmented_data(original_data, self.all_db_values, self.db_to_queries)
            self.num_augs = len(augmented_data)
        else:
            ex_idx = idx // self.num_augs
            aug_idx = idx % self.num_augs
            augmented_data = self.data[ex_idx][aug_idx]
        
        #assert len(augmented_data) == 8
        return augmented_data


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)

def collate_fn(batch):
    if isinstance(batch[0], list): # training
        labels = [item['label'] for sublist in batch for item in sublist]
        questions = [item['question'] for sublist in batch for item in sublist]
        if 'proc_query' in batch[0][0]:
            queries = [item['proc_query'] for sublist in batch for item in sublist]
        else:
            queries = [item['query'] for sublist in batch for item in sublist]
    else: # eval
        labels = [item['label'] for item in batch]
        questions = [item['question'] for item in batch]
        if 'proc_query' in batch[0]:
            queries = [item['proc_query'] for item in batch]
        else:
            queries = [item['query'] for item in batch]

    ret = tokenizer(text=questions, text_pair=queries, return_tensors='pt', padding=True)
    ret['labels'] = torch.Tensor(labels).view(-1, 1)
    return ret

def compute_metrics(outputs):
    #preds = outputs.predictions[0] if isinstance(outputs.predictions, tuple) else outputs.predictions
    preds = outputs.predictions
    preds = preds.reshape(-1)
    probs = torch.sigmoid(torch.tensor(preds))
    pred_labels = (probs > 0.5).float()
    labels = outputs.label_ids.reshape(-1)

    logits = outputs.predictions
    logits = torch.tensor(logits.reshape(-1,8))
    xent_labels = torch.tensor(labels.reshape(-1,8))
    assert torch.all(xent_labels[:,0] == 1) and torch.all(xent_labels[:,1:]==0)
    xent_loss = -torch.nn.functional.log_softmax(logits, dim=-1)
    xent_loss = xent_loss[:,0]
    xent_loss = xent_loss.mean()

    metrics = {}
    metrics.update({'xent': xent_loss})
    metrics.update({'roc': roc_auc_score(y_true=labels, y_score=preds, average='micro')})
    metrics.update(accuracy_metic.compute(predictions=pred_labels, references=labels)),
    metrics.update(f1_metric.compute(predictions=pred_labels, references=labels))
    return metrics

def main():
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    assert model.config.num_labels == 1
    model.resize_token_embeddings(len(tokenizer))

    all_db_values = pickle.load(open(ALL_DB_VALUES_PATH,'rb'))
    train_set = BCSpider(TRAIN_DATA_PATH, all_db_values, training=True)
    dev_set = BCSpider(DEV_DATA_PATH, all_db_values, training=False)

    trainer = FilterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
        num_augs=8
    )

    result = trainer.train()
    trainer.save_model()
    trainer.log_metrics('train', result.metrics)
    trainer.save_metrics('train', result.metrics)
    trainer.save_state()
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == '__main__':
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    main()
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
