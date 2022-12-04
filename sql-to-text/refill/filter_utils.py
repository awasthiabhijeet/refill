import numpy as np
import random
import json
import pickle
import re

from math import ceil, floor
from joblib import Parallel, delayed
from tqdm import tqdm
from tree_utils.moz_sql_parser import parse

from convert_to_improved_blec_format import process_query

SEED = 0

random.seed(SEED)
np.random.seed(SEED)

SPECIAL_TOKENS = [
                '<col>', '</col>', 
                '<tab>', '</tab>',
                '<val>', '</val>',
                '<sep>', '</sep>',
                '<temp>', '</temp>',
                ]

SQL_TOK_REPLACEMENTS = {
    "intersect": ["union", "except"],
    "union": ["intersect", "except"],
    "except": ["intersect", "union"],
    #"between": ["not between"], # ast_to_ra unable to parse not between
    "not between": ["between"],
    "in": ["not in"],
    "not in": ["in"],
    "like": ["not like"],
    "not like": ["like"],
    "is": ["not is"],
    "not is": ["is"],
    "exists": ["not exists"],
    "not exists": ["exists"],
    "=": ["!="],
    "!=": ["="],
    ">": ["<"],
    "<": [">"],
    ">=": ["<="],
    "<=": [">="],
    "and": ["or"],
    "or": ["and"],
    "desc": ["asc"],
    "asc": ["desc"],
}

SQL_AGG_REPLACEMENTS = {
    "max": ["min", "count", "sum", "avg"],
    "min": ["max", "count", "sum", "avg"],
    "count": ["max", "min", "sum", "avg"],
    "sum": ["max", "min", "count", "avg"],
    "avg": ["max", "min", "count", "sum"],
}

OTHER_SQL_TOKENS = [
    "select",
    "from",
    "where",
    "group",
    "order",
    "limit",
    "join", "on", "as",
    "none", "-", "+", "*", "/",
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


def is_number(x):
    """
    Takes a word and checks if Number (Integer or Float).
    """
    try:
        # only integers and float converts safely
        num = float(x)
        return True
    except:  # not convertable to float
        return False

def replace_numerical_value(value):
    num = float(value)
    num_frac = num / np.random.randint(2,11)
    toss = np.random.uniform()
    if toss < 0.5:
        new_num = ceil(num - num_frac)
    else:
        new_num = ceil(num + num_frac)
    new_num = max(0, new_num)
    return str(new_num)

def replace_column_value(col_name, all_values, cur_value, db_id):
    col_name = col_name.split('.')[-1]
    if cur_value[0] == "'":
        potential_value = cur_value
        for table in all_values[db_id]['unique_db_values']:
            for col in all_values[db_id]['unique_db_values'][table]:
                values_to_choose = all_values[db_id]['unique_db_values'][table][col]
                values_to_choose = list(filter(None, values_to_choose))
                if col.lower() == col_name.lower() and len(values_to_choose) > 0:
                    new_value = np.random.choice(values_to_choose)
                    if isinstance(new_value, str):
                        new_value = re.sub(r'[^A-Za-z0-9 ]+', '', new_value)
                        new_value = new_value.replace("'","")
                        new_value = f"'{new_value}'"
                    else:
                        new_value = str(new_value)
                        new_value = f"'{new_value}'"
                    if cur_value[1:-1] in all_values[db_id]['unique_db_values'][table][col]:
                        return new_value
                    else:
                        potential_value = new_value
        return potential_value
    elif is_number(cur_value):
        return replace_numerical_value(cur_value)
    else:
        return cur_value

def replace_values_in_query(query, db_id, all_values, p=0.5):
    query_toks = query.split()
    new_toks = []
    value_possible = False
    comparator = False
    value_end = True
    constructed_value = []
    toss = np.random.uniform()

    for tok in query_toks:
        if value_possible:
            if tok[0] == "'":
                value_end = False
            if tok[-1] == "'":
                value_end = True
            
            constructed_value.append(tok)

            if not value_end:
                continue
            else:
                original_value = ' '.join(constructed_value)
                constructed_value = []
                if toss < p:
                    tok = replace_column_value(col_name, all_values, original_value, db_id)
                else:
                    tok = original_value        
                value_possible = False

        if comparator:
            if toss < p and is_number(tok):
                tok = replace_numerical_value(tok)
            comparator = False

        if tok in ['=', '!=']:
            col_name = new_toks[-1]
            value_possible = True
        if tok in ['<', '<=', '>', '>=']:
            comparator = True

        new_toks.append(tok)
    
    new_query = sanitize_query(' '.join(new_toks))
    new_query = new_query.replace(' intersect ', ' INTERSECT ')
    new_query = new_query.replace(' union ', ' UNION ')
    new_query = new_query.replace(' except ', ' EXCEPT ')
    
    original_query = sanitize_query(' '.join(query_toks))
    # if new_query == "SELECT count(*) , city FROM employees WHERE title = 'IT Staff' GROUP BY city":
    #    breakpoint()
    return new_query, new_query.lower() !=  original_query.lower()


def replace_sql_tokens_in_query(query, p=0.5):
    query_toks = query.split()
    new_toks = []
    for i,tok in enumerate(query_toks):
        toss = np.random.uniform()
        new_tok = tok.lower()
        
        # if = is for join , let's not perturb that (joins!)
        if new_tok == '=' and not (query_toks[i+1][0] == "'" or query_toks[i+1][0].isdigit()):
            new_toks.append(new_tok)
        # covers not x : drops not with prob p.
        elif i>0 and query_toks[i-1].lower() == 'not':
            if toss < p: # remove not
                new_toks = new_toks[:-1]
            else: # retain not
                assert new_toks[-1].lower() == 'not'
                new_toks[-1] = 'not'
            #assert new_tok in SQL_TOK_REPLACEMENTS # commented out because: not may also appear in a value
            new_toks.append(new_tok)
        elif new_tok in SQL_TOK_REPLACEMENTS:
            if new_tok == 'and' and 'between' in query.lower():
                new_tok = new_tok
            elif toss < p:
                replacements = SQL_TOK_REPLACEMENTS[new_tok]
                new_tok = np.random.choice(replacements)
            #if new_tok == "or" and query_toks[i-2].lower() == 'between': # between a or b is wrong!
            #    new_tok = query_toks[i].lower()
            new_toks.append(new_tok)
        elif new_tok in SQL_AGG_REPLACEMENTS.keys():
            for key in SQL_AGG_REPLACEMENTS:
                if new_tok == key and toss < p:
                    new_tok = new_tok.replace(key, np.random.choice(SQL_AGG_REPLACEMENTS[key]))
            new_toks.append(new_tok)
        elif new_tok == 'limit':
            if new_toks[-1] not in ['asc', 'desc'] and toss < p:
                new_toks.append('desc')
            new_toks.append(new_tok)
        # elif new_tok in OTHER_SQL_TOKENS:
        #     new_toks.append(new_tok)
        else:
            new_toks.append(tok)
    
    new_query = sanitize_query(' '.join(new_toks))
    new_query = new_query.replace(' intersect ', ' INTERSECT ')
    new_query = new_query.replace(' union ', ' UNION ')
    new_query = new_query.replace(' except ', ' EXCEPT ')

    original_query = sanitize_query(' '.join(query_toks))
    # if new_query.lower() !=  sanitize_query(' '.join(query_toks)).lower():
    #     breakpoint()
    return new_query, new_query.lower() !=  original_query.lower()
    
def shuffle_text(text, span_fraction=0.3):
    original_text = text
    text = text.split()
    text_len = len(text)
    span_len = max(3, floor(span_fraction * text_len))
    span_len = min(text_len-1, span_len)
    span_start = np.random.randint(0, text_len-span_len)
    span_end = span_start + span_len # non inclusive
    span = text[span_start:span_end]
    random.shuffle(span)
    new_text = text[0:span_start] + span + text[span_end:]
    new_text = ' '.join(new_text)
    # if new_text == original_text:
    #     print()
    #     print(span)
    #     print(new_text, original_text, '  ## span ##  ')
    #     print()
    # else:
    #     pass
    #     #print('yo', '  ## span ##  ' )
    # return new_text
    if new_text != original_text:
        return new_text
    else:
        return shuffle_text(original_text, span_fraction)

def drop_text_tokens(text, drop_fraction=0.3):
    tokens = text.split()
    new_tokens = [tok for tok in tokens if np.random.uniform() > drop_fraction]
    new_text = ' '.join(new_tokens)
    # if new_text == text:
    #     print(new_text, text, '  ## drop ##  ')
    #     breakpoint()
    # else:
    #     pass
    #     #print('yo', '  ## drop ##  ')
    # return new_text
    if new_text != text:
        return new_text
    else:
        return drop_text_tokens(text, drop_fraction)

def replace_sql_query_from_same_db(query, db_to_queries, db_id):
    new_query = query
    while new_query.lower() == query.lower():
        new_query = np.random.choice(db_to_queries[db_id])
    assert new_query.lower() != query.lower()
    return new_query

def get_text_from_refill(query, query_to_refill_text):
    return np.random.choice(query_to_refill_text[query])


def get_num_of_successfully_parsed_queries(queries):
    count = 0
    for query in tqdm(queries):
        try:
            parsed_query = parse(query)
            count +=1
        except Exception as e:
            print(e)
            print(f'unable to parse: {query}')
            continue
    return count

def get_augmented_data(data, all_db_values, db_to_queries):
    examples = []
    query = sanitize_query(data["query"])
    question = data["question"]
    db_id = data["db_id"]

    # 1: Original Query
    examples.append({
        "query": query,
        "question": question,
        "label": 1,
        "type": "original_query",
        "db_id": db_id,
        })
    
    # 2: val_replaced_query
    val_replaced_query, is_val_replaced = replace_values_in_query(query, db_id, all_db_values, p=1.0)
    if is_val_replaced:
        examples.append({
            "query": val_replaced_query,
            "question": question,
            "label": 0,
            "type": "val_replaced_query",
            "db_id": db_id,
            })

    # 3: tok_replaced_query
    tok_replaced_query, is_tok_replaced = replace_sql_tokens_in_query(query, 0.5)
    if is_tok_replaced:
        examples.append({
            "query": tok_replaced_query,
            "question": question,
            "label": 0,
            "type": "tok_replaced_query",
            "db_id": db_id,
            })

    # 4: val_tok_replaced_query
    val_tok_replaced_query, is_val_tok_replaced = replace_sql_tokens_in_query(val_replaced_query, 0.5)
    if is_val_tok_replaced:
        examples.append({
            "query": val_tok_replaced_query,
            "question": question,
            "label": 0,
            "type": "val_tok_replaced_query",
            "db_id": db_id,
            })

    # 5: SAME_DB_QUERY
    same_db_query = replace_sql_query_from_same_db(query, db_to_queries, db_id)
    examples.append({
        "query": same_db_query,
        "question": question,
        "label": 0,
        "type": "same_db_query",
        "db_id": db_id,
        })

    # 6: Shuffled Text:
    shuffled_question = shuffle_text(question, span_fraction=0.3)
    assert shuffled_question != question
    examples.append({
        "query": query,
        "question": shuffled_question,
        "label": 0,
        "type": "shuffled_text",
        "db_id": db_id,
        })

    # 7: Drop Text
    dropped_tok_question = drop_text_tokens(question, drop_fraction=0.3)
    assert dropped_tok_question != question
    examples.append({
        "query": query,
        "question": dropped_tok_question,
        "label": 0,
        "type": "dropped_toks",
        "db_id": db_id,
        })

    # 8: ToDO ReFILL Text

    while len(examples) != 8:
        same_db_query = replace_sql_query_from_same_db(query, db_to_queries, db_id)
        examples.append({
            "query": same_db_query,
            "question": question,
            "label": 0,
            "type": "same_db_query",
            "db_id": db_id,
            })

    assert len(examples) == 8

    if 'proc_query' in data:
        '''
        for ex_idx,ex in enumerate(examples):
            proc_query = process_query(ex['query'], ex['db_id'])
            #if ex_idx == 0 and proc_query != data['proc_query']:
            #    breakpoint()
            ex['proc_query'] = proc_query
        '''

        # parallel
        proc_queries = Parallel(n_jobs=-1)(delayed(process_query)(ex['query'], ex['db_id']) 
                                        for ex in examples)
        for ex_idx,ex in enumerate(examples):
            examples[ex_idx]['proc_query'] = proc_queries[ex_idx]
        

    return examples

def get_db_to_queries(data):
    db_to_queries = {}
    for item in data:
        db_id = item['db_id']
        query = sanitize_query(item['query'])
        if db_id in db_to_queries:
            db_to_queries[db_id].append(query)
        else:
            db_to_queries[db_id] = [query]
    return db_to_queries

if __name__ == '__main__':
    #INPUT_JSON = '/mnt/infonas/data/awasthi/semantic_parsing/smbop/dataset/dev.json'
    INPUT_JSON = '/mnt/infonas/data/awasthi/semantic_parsing/copy_ms_rnd_1/abhijeet/sqltotext_variants/jsons/dev_improved_blec.json'
    ALL_DB_VALUES_PATH = 'replace_db_values/spider_db_values.pkl'
    OUPUT_GOLD_QUERIES = 'scratch/gold.sql'
    OUTPUT_VAL_REPLACED_QUERIES = 'scratch/val_replaced.sql'
    OUTPUT_TOK_REPLACED_QUERIES = 'scratch/tok_replaced.sql'
    OUTPUT_AUG_DATA = 'data/dev_blec_improved.json'
    
    
    data = json.load(open(INPUT_JSON))#[900:]
    all_db_values = pickle.load(open(ALL_DB_VALUES_PATH,'rb'))
    queries = [sanitize_query(item['query']) for item in data]
    db_ids = [item['db_id'] for item in data]
    questions = [item['question'] for item in data]
    db_to_queries = get_db_to_queries(data)

    
    augmented_data = [get_augmented_data(item, all_db_values, db_to_queries) 
                        for item in tqdm(data)]

    # augmented_data = Parallel(n_jobs=-1)(delayed(get_augmented_data)(item, all_db_values, db_to_queries) 
    #                                         for item in tqdm(data))
    # json.dump(augmented_data, open(OUTPUT_AUG_DATA,'w'), indent=4)
    


    
    '''
    num_parsed_queries = get_num_of_successfully_parsed_queries(queries)


    value_replaced_queries, is_value_replaced_queries = \
    zip(*[replace_values_in_query(query, db_id, all_db_values, p=1.0)
                              for query,db_id in tqdm(zip(queries, db_ids), total=len(queries))])
    value_replaced_queries, is_value_replaced_queries = list(value_replaced_queries), list(is_value_replaced_queries)

    tok_replaced_queries, is_tok_replaced_queries = \
        zip(*[replace_sql_tokens_in_query(query,p=1) for query in tqdm(queries)])
    tok_replaced_queries, is_tok_replaced_queries = list(tok_replaced_queries), list(is_tok_replaced_queries)

    num_parsed_val_rep_queries = get_num_of_successfully_parsed_queries(value_replaced_queries)
    num_parsed_tok_rep_queries = get_num_of_successfully_parsed_queries(tok_replaced_queries)
    '''

    # with open(OUPUT_GOLD_QUERIES,'w') as f:
    #     for i, (sql, db_id) in enumerate(zip(queries, db_ids)):
    #         if is_value_replaced_queries[i]:
    #             f.write(sql + '\t' + db_id + '\n')

    # with open(OUTPUT_VAL_REPLACED_QUERIES,'w') as f:
    #     for i, (sql, db_id) in enumerate(zip(value_replaced_queries, db_ids)):
    #         if is_value_replaced_queries[i]:
    #             if sql == queries[i]:
    #                 breakpoint()
    #                 #pass
    #             f.write(sql + '\t' + db_id + '\n')

    # with open(OUPUT_GOLD_QUERIES,'w') as f:
    #     for i, (sql, db_id) in enumerate(zip(queries, db_ids)):
    #         if is_tok_replaced_queries[i]:
    #             f.write(sql + '\t' + db_id + '\n')

    # with open(OUTPUT_TOK_REPLACED_QUERIES,'w') as f:
    #     for i, (sql, db_id) in enumerate(zip(tok_replaced_queries, db_ids)):
    #         if is_tok_replaced_queries[i]:
    #             if sql == queries[i]:
    #                 pass
    #             f.write(sql + '\t' + db_id + '\n')

    x = "Hi I'm Abhijeet Awasthi from Indore Madhya Pradesh. I study at IIT Bombay Mumbai. I am currently working from home"





'''
CLAUSE_KEYWORDS = (
    "select",
    "from",
    "where",
    "group",
    "order",
    "limit",
    "intersect",
    "union",
    "except",
)
JOIN_KEYWORDS = ("join", "on", "as")

WHERE_OPS = (
    "not",
    "between",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "in",
    "like",
    "is",
    "exists",
)
UNIT_OPS = ("none", "-", "+", "*", "/")
AGG_OPS = ("none", "max", "min", "count", "sum", "avg")
TABLE_TYPE = {
    "sql": "sql",
    "table_unit": "table_unit",
}

COND_OPS = ("and", "or")
SQL_OPS = ("intersect", "union", "except")
ORDER_OPS = ("desc", "asc")
mapped_entities = []
'''