import json
import converter
from preprocess_nl2sql import BERT_MODEL, bad_query_replace, bad_question_replace, ValueAlignmentException, QueryBuildError, SQL_PRIMITIVES
from nltk.stem.porter import PorterStemmer


import preprocess_sql2nl as preprocess
from transformers import BartForConditionalGeneration, BartTokenizer
import sys

from eval_scripts.process_sql import tokenize

TABLES_PATH = '../../data/spider/tables.json'
DATABASE_PATH = '../../data/spider/database/'
BART_PATH = 'facebook/bart-base'

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

class SQLEncoder:
    def __init__(self):
        pass

    @classmethod
    def align_values(cls, no_value, yes_value):
        if yes_value[-1] == ';':
            yes_value.pop()
        yes_value = '___'.join(yes_value)
        for f, t in bad_query_replace:
            yes_value = yes_value.replace(f, t)
        yes_value = yes_value.split('___')

        def find_match(no_value, i, yes_value):
            before = None if i == 0 else no_value[i-1].lower()
            after = None if i+1 == len(no_value) else no_value[i+1].lower()
            candidates = []

            for j in range(len(yes_value)):
                mybefore = None if j == 0 else yes_value[j-1].lower()
                if mybefore == before:
                    for k in range(j, len(yes_value)):
                        yk = yes_value[k].lower()
                        if yk in SQL_PRIMITIVES and yk not in {'in'}:
                            break
                        # if '_' in yk and 'mk_man' not in yk and 'pu_man' not in yk and 'xp_' not in yk or yk in {'t1', 't2', 't3', 't4'}:
                        #     break
                        myafter = None if k+1 == len(yes_value) else yes_value[k+1].lower()
                        if myafter == after:
                            candidates.append((j, k+1))
                            break
            if len(candidates) == 0:
                print(no_value)
                print(no_value[i])
                print(yes_value)
                print('import pdb; pdb.set_trace()')
                return None, None
            candidates.sort(key=lambda x: x[1] - x[0])
            return candidates[0]

        values = []
        num_slots = 0
        for i, t in enumerate(no_value):
            t = t.lower()
            if 'value' in t and t not in {'attribute_value', 'market_value', 'value_points', 'market_value_in_billion', 'market_value_billion', 'product_characteristic_value', 'total_value_purchased'}:
                start, end = find_match(no_value, i, yes_value)
                if start is None and end is None:
                    print('Start and End are both None')
                    raise ValueAlignmentException
                values.append(yes_value[start:end])
                num_slots += 1
        if num_slots != len(values):
            raise Exception('Found {} values for {} slots'.format(len(values),  num_slots))
        return values

    @classmethod
    def build_contexts(cls, query_norm_toks, g_values, db, bert, max_lim=512):
        columns = []
        for table_id, (to, t) in enumerate(zip(db['table_names_original'] + ['NULL'], db['table_names'] + ['NULL'])):
            # insert a NULL table at the end
            columns += [{'oname': '*', 'name': '*', 'type': 'all', 'key': '{}.*'.format(to).replace('NULL.', '').lower(), 'table_name': t.lower()}]
            keys = set(db['primary_keys'])
            for a, b in db['foreign_keys']:
                keys.add(a)
                keys.add(b)
            for i, ((tid, co), (_, c), ct) in enumerate(zip(db['column_names_original'], db['column_names'], db['column_types'])):
                ct = ct if i not in keys else 'key'
                if tid == table_id:
                    columns.append({
                        'oname': co, 'name': c, 'type': ct,
                        'key': '{}.{}'.format(to, co).lower(),
                        'table_name': t.lower(),
                    })

        key2col = {col['key']: col for col in columns}

        question_context = [bert.cls_token]
        for t in query_norm_toks:
            if t in key2col:
                col = key2col[t]
                question_context.extend(bert.tokenize('[ {} {} : {} ]'.format(col['type'], col['table_name'], col['name'])))
            else:
                question_context.extend(bert.tokenize(t))
        question_context.append(bert.sep_token)
        for v in g_values:
            question_context.extend(bert.tokenize(' '.join(v)))
            question_context.append(';')
        if question_context[-1] == ';':
            question_context[-1] = bert.sep_token

        if len(question_context) > max_lim:
            raise Exception('question context of {} > {} is too long!'.format(len(question_context), max_lim))
        return question_context, columns

    @classmethod
    def get_sql_encoding(cls, ex, bert, conv):
        if 'column_mapped' in ex and 'values' in ex:
            question_context, columns = preprocess.SQLDataset.build_contexts(ex['column_mapped'], 
                                                                            ex['values'], 
                                                                            conv.database_schemas[ex['db_id']],
                                                                            bert)
            return question_context

        db_id = ex['db_id']

        invalid = False
        if 'query_toks' not in ex.keys():
            ex['query_toks'] = tokenize(ex['query'])
        if 'query_toks_no_value' not in ex.keys():
            ex['query_toks_no_value'] = ['value' if is_value(tok) else tok for tok in ex['query_toks']]
        try:
            # normalize query
            query_norm = conv.convert_tokens(ex['query_toks'], ex['query_toks_no_value'], db_id)
        except Exception as e:
            print('preprocessing error')
            print(ex['query'])
            print(e)
            raise e
            return None

        if query_norm is None:
            return None
        query_norm_toks = query_norm.split()

        query_recov = g_values = None
        try:
            query_recov = conv.recover(query_norm, db_id)
            em, g_sql, r_sql = conv.match(ex['query'], query_recov, db_id)
            if not em:
                invalid = True
            g_values = cls.align_values(ex['query_toks_no_value'], ex['query_toks'])
        except ValueAlignmentException as e:
            print(ex['query'])
            print(repr(e))
            invalid = True
        except QueryBuildError as e:
            print(ex['query'])
            print(repr(e))
            invalid = True
        except Exception as e:
            print(e)
            invalid = True
            raise

        try:
            question_context, columns = cls.build_contexts(query_norm_toks, 
                                                        g_values, 
                                                        conv.database_schemas[db_id], 
                                                        bert)
        except Exception as e:
            print(e)
            return None

        return question_context


if __name__ == '__main__':

    input_json = sys.argv[1]
    output_json = input_json.replace('.json', '_sql_enc.json')

    input_data = json.load(open(input_json))
    tokenizer = BartTokenizer.from_pretrained(BART_PATH)
    conv = converter.Converter(TABLES_PATH, DATABASE_PATH)
    sql_encoder = SQLEncoder()

    output_data = []
    for item in input_data:
        sql_encoding = sql_encoder.get_sql_encoding(item, tokenizer, conv)
        item['sql_encoding'] = sql_encoding
        if sql_encoding is not None:
            output_data.append(item)

    json.dump(output_data,open(output_json,'w'), indent=4)
