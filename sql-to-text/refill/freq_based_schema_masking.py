import json
from nltk import word_tokenize
from tqdm import tqdm
from collections import OrderedDict
from nltk.stem import PorterStemmer
ps = PorterStemmer()

COUNT_THRESHOLD = 70
TRAIN_JSON = 'data/spider/train_spider.json'
DEV_JSON = 'data/spider/dev.json'
TRAIN_OUTPUT = f'data/sql-to-text/refill/jsons/train_th_{COUNT_THRESHOLD}.json'
DEV_OUTPUT = f'data/sql-to-text/refill/jsons/dev_th_{COUNT_THRESHOLD}.json'


def replace_data(data, words_to_keep):
	for item in tqdm(data):
		template_words = item['question'].split()
		new_template = []
		for word in template_words:
			if ps.stem(word.lower()) not in words_to_keep and len(word)>3:
				word = '<mask>'

			if len(new_template)>0 and new_template[-1] == '<mask>' and word == '<mask>':
				continue
			else:
				new_template.append(word)
		new_template = ' '.join(new_template)
		item['template'] = new_template
	return data

train_data = json.load(open(TRAIN_JSON))
dev_data = json.load(open(DEV_JSON))

db_to_words = {}
all_words = {}
word_to_schema = {}

for item in tqdm(train_data):
	question = item['question'].lower()
	db_id = item['db_id']
	words = set(word_tokenize(question))
	for item in words:
		item = ps.stem(item)
		if item in word_to_schema:
			if db_id not in word_to_schema[item]:
				word_to_schema[item].append(db_id)
		else:
			word_to_schema[item] = [db_id]


word_to_schema_count = dict(sorted([(item, len(word_to_schema[item])) 
									for item in word_to_schema], 
							key=lambda elem: elem[1]))
words_to_keep = [word for word in word_to_schema_count 
				if word_to_schema_count[word]>=COUNT_THRESHOLD]

new_train_data = replace_data(train_data, words_to_keep)
new_dev_data = replace_data(dev_data, words_to_keep)

json.dump(new_train_data, open(TRAIN_OUTPUT,'w'), indent=4)
json.dump(new_dev_data, open(DEV_OUTPUT,'w'), indent=4)