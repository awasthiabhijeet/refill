import json
from sql_formatter.formatting import translate_sql
from tree_utils.ra_preproc import ast_to_ra
from tree_utils.node_util import get_literals
from tree_utils.moz_sql_parser import parse
from tqdm import tqdm
from joblib import Parallel, delayed


train_input = f'data/sql-to-text/refill/jsons/train_th_70.json'
dev_input = f'data/sql-to-text/refill/jsons/dev_th_70.json'
tables = 'data/spider/tables.json'
train_output = 'data/sql-to-text/refill/jsons/train_spider_template_improved_blec_th_70.json'
dev_output = 'data/sql-to-text/refill/jsons/dev_template_improved_blec_th_70.json'
SPACE='<space>'


train_data = json.load(open(train_input))
dev_data = json.load(open(dev_input))
tables = json.load(open(tables))

db_tables = {}
for item in tables:
	table_names_original = item['table_names_original']
	table_names = item['table_names']
	assert len(table_names) == len(table_names_original)
	column_names_original = [c[1] for c in item['column_names_original'] if c[1]!='*']
	column_names = [c[1] for c in item['column_names'] if c[1]!='*']
	assert len(column_names) == len(column_names_original)
	table_names_mapping = dict(zip(table_names_original, table_names))
	column_names_mapping = dict(zip(column_names_original, column_names))
	db_tables[item['db_id']] = {
								'table_names_mapping': table_names_mapping, 
								'column_names_mapping': column_names_mapping
							   }

def get_query_literals(query):
	try:
		tree_dict = parse(query)
		tree_obj = ast_to_ra(tree_dict["query"])
		literals = get_literals(tree_obj)
		#print(literals)
	except Exception as e:
		print(f'could not parse: {query}')
		print(e)
		literals = []
	return literals

def sanitize(query):
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

def replace_names(query, db_id, literals):
	table_names_mapping = db_tables[db_id]['table_names_mapping']
	column_names_mapping = db_tables[db_id]['column_names_mapping']
	mapping_keys = list(table_names_mapping.keys()) + list(column_names_mapping.keys())
	mapping_keys = sorted(mapping_keys, key=len)
	for item in mapping_keys:
		if item in column_names_mapping:
			query = query.replace(f' {item.lower()} ', f' <col>{SPACE}{column_names_mapping[item]}{SPACE}</col> ')
		elif item in table_names_mapping:
			query = query.replace(f' {item.lower()} ', f' <tab>{SPACE}{table_names_mapping[item]}{SPACE}</tab> ')
		else:
			raise ValueError
	if len(literals) > 0:
		for lit in literals:
			query = query.replace(f" {lit.lower()} ", f' <val>{SPACE}{lit}{SPACE}</val> ')
			query = query.replace(f" '{lit.lower()}' ", f' <val>{SPACE}{lit}{SPACE}</val> ')
			query = query.replace(f' "{lit.lower()}" ', f' <val>{SPACE}{lit}{SPACE}</val> ')

	query = query.replace(SPACE, ' ')
	return query

def process_item(item):
	try:
		q = item['query']
		q = q.replace(' intersect ', ' INTERSECT ')
		q = q.replace(' union ', ' UNION ')
		q = q.replace(' except ', ' EXCEPT ')

		proc_query = translate_sql(sanitize(q))[1]
		literals = get_query_literals(sanitize(item['query']))
		proc_query = replace_names(proc_query, item['db_id'], literals)
		item["proc_query"] = proc_query
		item["query"] = sanitize(item["query"])
		return item
	except Exception as e:
		print(e)
		return None

if __name__ == '__main__':
	proc_train_data = Parallel(n_jobs=-1)(delayed(process_item)(item) for item in tqdm(train_data))
	proc_train_data = [item for item in proc_train_data if item is not None]

	proc_dev_data = Parallel(n_jobs=-1)(delayed(process_item)(item) for item in tqdm(dev_data))
	proc_dev_data = [item for item in proc_dev_data if item is not None]

	json.dump(proc_train_data, open(train_output,'w'), indent=4)
	json.dump(proc_dev_data, open(dev_output,'w'), indent=4)