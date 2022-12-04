import numpy as np
import torch
import os,sys
import json
import random
from process_sql import tokenize
from math import ceil
from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

MODEL_PATH = 'models/sql-to-text/refill/filter'

DEV_FRAC = 0.15
SEED = 0
random.seed(SEED)

@torch.no_grad()
def get_query_question_score(model, tokenizer, query, question):
	inputs = tokenizer(text=question, text_pair=query, return_tensors='pt', padding=True)
	output = model(**inputs)
	score = output['logits'].reshape(-1)
	score = score.cpu().numpy()
	return score



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

def anonymize_query_toks(query_toks):
	return ['value' if is_value(x.lower().strip()) else x.lower() for x in query_toks]

def get_query_toks(query):
	return tokenize(query)


def read_json_data(input_path, 
				num_hypotheses, 
				use_template_instances=True, 
				use_no_template_text=False,
				model=None,
				tokenizer=None,
				threshold=5.0):
	print('reading and sorting data')
	data = json.load(open(input_path))

	all_main_examples = []
	all_template_instances = []
	sorted_data = []
	for item in data:
		sorted_data.append(item.copy())
		query = item["query"]
		query_toks = get_query_toks(query)
		query_toks_no_value = anonymize_query_toks(query_toks)
		db_id = item["db_id"]
		no_template_text = item["hypotheses"][0]
		hypotheses = item["hypotheses"][1:]
		templates = item["templates"][1:]

		bc_score_no_template = get_query_question_score(model, tokenizer, query, no_template_text).tolist()[0]

		if len(hypotheses) > 0:
			no_template = False		
			bc_scores = get_query_question_score(model, tokenizer, [query]*len(hypotheses), hypotheses)
			sortex_idx = np.argsort(-bc_scores)
			hypotheses = [hypotheses[idx] for idx in sortex_idx]
			templates = [templates[idx] for idx in sortex_idx]

			bc_scores = sorted(bc_scores.tolist(), reverse=True) 
			sorted_data[-1]['hypotheses'] = [(no_template_text, bc_score_no_template)] \
										  + list(zip(hypotheses,bc_scores))
			sorted_data[-1]['templates'] = [''] + templates

			hypotheses = [hypotheses[idx] for idx in range(len(hypotheses)) if bc_scores[idx]>threshold]
			templates = [templates[idx] for idx in range(len(templates)) if bc_scores[idx]>threshold]
			#sorted_data[-1]['bc_scores'] = bc_scores
		else:
			no_template = True
			sorted_data[-1]['hypotheses'] = [(no_template_text, bc_score_no_template)]
			sorted_data[-1]['templates'] = ['']
			
		
		main_examples = []
		questions_seen = []
		retained_templates = []
		for hyp,tmp in zip(hypotheses, templates):
			if len(questions_seen) == num_hypotheses:
				break
			if hyp not in questions_seen:
				main_examples.append(
								{
								'db_id': db_id,
								'query': query,
								'question': hyp,
								'query_toks': query_toks,
								'query_toks_no_value': query_toks_no_value,
								})
				questions_seen.append(hyp)
				retained_templates.append(tmp)
			else:
				continue

		#if (use_no_template_text or (no_template and bc_score_no_template>threshold)) \
		#	and (no_template_text not in questions_seen):
		if (bc_score_no_template>threshold) and (no_template_text not in questions_seen):
			# print(f'yo: {no_template}, {bc_score_no_template}')
			no_temp_ex = {'db_id': db_id, 'query': query, 'question': no_template_text, 
						'query_toks': query_toks, 'query_toks_no_value': query_toks_no_value}
			main_examples.append(no_temp_ex)

		all_main_examples.extend(main_examples)

		template_instances = []
		for el in retained_templates:
			q = el['query']
			q_toks = get_query_toks(q)
			el['query_toks'] = q_toks
			el['query_toks_no_value'] = anonymize_query_toks(q_toks)
			template_instances.append(el)

		all_template_instances.extend(template_instances)

	sorted_file_path = input_path.replace('.json', '_sorted_bc_th.json')
	json.dump(sorted_data, open(sorted_file_path,'w'), indent=4)
	return all_main_examples, all_template_instances

def dump_data(input_path, model, tokenizer):
	main_examples, template_examples = read_json_data(input_path, 
										num_hypotheses=5, 
										model=model, 
										tokenizer=tokenizer)

	print('dumping data...')
	random.shuffle(main_examples)
	random.shuffle(template_examples)

	num_main_examples = len(main_examples)
	num_dev_examples = ceil(DEV_FRAC * len(main_examples))
	main_examples_train = main_examples[0:-num_dev_examples]
	main_examples_dev = main_examples[-num_dev_examples:]
	train_with_template = main_examples_train + template_examples

	input_file_name = os.path.basename(input_path) 
	output_dir = os.path.dirname(input_path)
	train_main_path = os.path.join(output_dir, input_file_name.replace('.json', '_train_main_bc_th.json'))
	dev_main_path = os.path.join(output_dir, input_file_name.replace('.json', '_dev_main_bc_th.json'))
	train_with_template_path = os.path.join(output_dir, input_file_name.replace('.json', '_train_w_temp_bc_th.json'))

	json.dump(main_examples_train, open(train_main_path,'w'), indent=4)
	json.dump(main_examples_dev, open(dev_main_path,'w'), indent=4)
	json.dump(train_with_template, open(train_with_template_path,'w'), indent=4)


if __name__ == '__main__':
	model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
	model.eval()
	tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
	dump_data(sys.argv[1], model, tokenizer)





