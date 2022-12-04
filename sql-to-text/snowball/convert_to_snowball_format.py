import os
import json
from sql_formatter.formatting import translate_sql
from tqdm import tqdm
import traceback

train_input = '../../data/spider/train_spider.json'
dev_input = '../../data/spider/dev.json'

output_root = '../../data/sql-to-text/snowball/jsons/'

if not os.path.exists(output_root):
    os.makedirs(output_root)

train_output = output_root + 'train_spider_snowball.json'
dev_output = output_root + 'dev_snowball.json'

train_data = json.load(open(train_input))
dev_data = json.load(open(dev_input))

proc_train_data = []
for item in tqdm(train_data, total=len(train_data)):
    try:
        proc_query = translate_sql(item['query'])[1]
        item["proc_query"] = proc_query
        proc_train_data.append(item)
    except Exception as e:
        print(f'Exception on query : {item["query"]}')
        print(traceback.format_exc())
        continue

proc_dev_data = []
for item in tqdm(dev_data, total=len(dev_data)):
    try:
        proc_query = translate_sql(item['query'])[1]
        item["proc_query"] = proc_query
        proc_dev_data.append(item)
    except Exception as e:
        print(f'Exception on query : {item["query"]}')
        print(traceback.format_exc())
        continue

json.dump(proc_train_data, open(train_output,'w'), indent=4)
json.dump(proc_dev_data, open(dev_output,'w'), indent=4)
