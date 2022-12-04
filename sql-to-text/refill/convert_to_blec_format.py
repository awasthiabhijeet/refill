import json
from sql_formatter.formatting import translate_sql
from tqdm import tqdm

train_input = '/mnt/infonas/data/awasthi/semantic_parsing/smbop/dataset/train_spider.json'
dev_input = '/mnt/infonas/data/awasthi/semantic_parsing/smbop/dataset/dev.json'

train_output = 'jsons/train_spider_blec.json'
dev_output = 'jsons/dev_blec.json'

train_data = json.load(open(train_input))
dev_data = json.load(open(dev_input))

proc_train_data = []
for item in tqdm(train_data, total=len(train_data)):
	try:
		proc_query = translate_sql(item['query'])[1]
		item["proc_query"] = proc_query
		proc_train_data.append(item)
	except:
		continue

proc_dev_data = []
for item in tqdm(dev_data, total=len(dev_data)):
	try:
		proc_query = translate_sql(item['query'])[1]
		item["proc_query"] = proc_query
		proc_dev_data.append(item)
	except:
		continue

json.dump(proc_train_data, open(train_output,'w'), indent=4)
json.dump(proc_dev_data, open(dev_output,'w'), indent=4)