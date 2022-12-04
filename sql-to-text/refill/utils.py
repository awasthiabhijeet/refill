import json
from collections import OrderedDict
from tqdm import tqdm
from convert_to_improved_blec_format import process_item

SPECIAL_TOKENS = [
				'<col>', '</col>', 
                '<tab>', '</tab>',
                '<val>', '</val>',
                '<sep>', '</sep>',
                '<temp>', '</temp>',
          		] 

def sanitize_query(query):
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

def read_spider_json(json_path, tables_path, add_tables=False, sql_to_templates=None):
	data = json.load(open(json_path))
	tables = json.load(open(tables_path))
	db_tables = {}
	for table in tables:
	    db_tables[table['db_id']] = table 
	    db_tables[table['db_id']]['columns'] = [x[1] for x in table['column_names']]

	sql_to_text = OrderedDict() 
	for item in data:
		if 'proc_query' not in item:
			item = process_item(item)
			if item is None:
				continue
		if 'question' not in item:
			item['question'] = ''
		sql = item['query']
		query = item['proc_query']
		if sql_to_templates is not None:
			templates = sql_to_templates.get(sanitize_query(sql), None)
		else:
			templates = None
		if add_tables:
			table_names = db_tables[item['db_id']]['table_names']
			query = query + ' <sep> All tables are ' + ' , '.join(table_names)
		if query in sql_to_text:
			#print(sql, sql_to_text[query][-1]['query'])
			#assert sql_to_text[query][-1]['query'] == sql
			sql_to_text[query].append({
										'question': item['question'], 
										'query': sql,
										'db_id': item['db_id'],
									})
		else:
			sql_to_text[query] = [{'question': item['question'], 
								   'query': sql,
								   'db_id': item['db_id'],
								   'templates': []
								 }]

		if templates is not None:
			existing_templates = sql_to_text[query][0]['templates']
			if len(existing_templates) > 0:
				continue
			else:
				sql_to_text[query][0]['templates'] = templates
	return list(sql_to_text.items())

def read_templates(data):
	ques_to_temp = {}
	print('reading templates to construct ques to template dict..')
	for item in tqdm(data):
		ques = item['question']
		template = item['template']
		if ques in ques_to_temp:
			if template.count('<mask>') > ques_to_temp[ques].count('<mask>'):
				ques_to_temp[ques] = template
		else:
			ques_to_temp[ques] = template
	return ques_to_temp

def get_sql_to_templates(retrieved_nbrs, templates, num_nbrs=5):
	sql_to_templates = {}
	print('reading sql to retrieved_nbrs to construct sql to templates...')
	for sql in tqdm(retrieved_nbrs):
		nbrs = retrieved_nbrs[sql]
		nbr_ques = [{
					'question': item['question'], 
					'masked_template': templates[item['question']], 
					'query': item['query'], 
					'ted': item['ted'],
					'db_id': item['db_id']
					} for item in nbrs]
		unqiue_templates = []
		result = []
		for item in nbr_ques:
			if item['masked_template'] in unqiue_templates:
				continue
			else:
				unqiue_templates.append(item['masked_template'])
				result.append(item)
				if len(result) == num_nbrs:
					break

		sql_to_templates[sanitize_query(sql)] = result
	return sql_to_templates





	
