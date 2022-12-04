import os
import json
from tqdm import tqdm
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

output_root = '../../data/sql-to-text/l2s/jsons/'

if not os.path.exists(output_root):
    os.makedirs(output_root)

train_output = output_root + 'train_spider_l2s.json'
dev_output = output_root + 'dev_l2s.json'

train_data_config = {
        "name": "spider",
        "paths": [train_input],
        "tables_paths": [tables],
        "db_path": database
}
dev_data_config = {
        "name": "spider",
        "paths": [dev_input],
        "tables_paths": [tables],
        "db_path": database
}

train_data = registry.construct("dataset", train_data_config)
dev_data = registry.construct("dataset", dev_data_config)

train_json_data = json.load(open(train_input))
dev_json_data = json.load(open(dev_input))

proc_train_data = []
for i,item in tqdm(enumerate(train_data.examples), total=len(train_data.examples)):
	astree = grammar.parse(item.code)
	proc_query = pcfg.unparse(grammar.ast_wrapper, item.schema, astree)
	json_item = train_json_data[i]
	json_item["proc_query"] = proc_query
	proc_train_data.append(json_item)

proc_dev_data = []
for i,item in tqdm(enumerate(dev_data.examples), total=len(dev_data.examples)):
	astree = grammar.parse(item.code)
	proc_query = pcfg.unparse(grammar.ast_wrapper, item.schema, astree)
	json_item = dev_json_data[i]
	json_item["proc_query"] = proc_query
	proc_dev_data.append(json_item)

json.dump(proc_train_data, open(train_output,'w'), indent=4)
json.dump(proc_dev_data, open(dev_output,'w'), indent=4)
