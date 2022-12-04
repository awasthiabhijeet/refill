import json 
import pickle 
from tqdm import tqdm 
from apted import APTED, Config
import numpy as np 
import os,sys 
from anytree import PreOrderIter
import itertools 
import multiprocessing
from joblib import Parallel, delayed
from collections import OrderedDict
import edit_distance
from nltk import word_tokenize

import smbop.utils.ra_preproc as ra_preproc
from smbop.utils import moz_sql_parser as msp 
import smbop.dataset_readers.disamb_sql as disamb_sql

import argparse


class CustomConfig(Config):
    def __init__(self):
        super().__init__()
        self.agg_grp=['max','min','avg','count','sum']
        self.order_grp=['Orderby_desc','Orderby_asc']
        self.boolean_grp=['Or','And']
        self.set_grp=['union','intersect','except']
        self.leaf_grp=['Val_list','Value','literal']#,'Table']
        self.sim_grp=['like','in','nin']
        self.comp_grp=['gt','lte','eq','lt','gte','neq']

    def rename(self, node1, node2):
        if (node1.name in self.agg_grp and node2.name in self.agg_grp) or \
        (node1.name in self.order_grp and node2.name in self.order_grp) or \
        (node1.name in self.boolean_grp and node2.name in self.boolean_grp) or \
        (node1.name in self.set_grp and node2.name in self.set_grp) or \
        (node1.name in self.leaf_grp and node2.name in self.leaf_grp) or \
        (node1.name in self.sim_grp and node2.name in self.sim_grp) or \
        (node1.name in self.comp_grp and node2.name in self.comp_grp):
            return 0.5 if node1.name != node2.name else 0
        else:
            return 1 if node1.name != node2.name else 0
    def children(self, node):
        return [x for x in node.children]

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

def parse_item(item):
    try:
        query = sanitize_query(item["query"])
        item["query"] = query
        tree_dict = msp.parse(query)
        tree_obj = ra_preproc.ast_to_ra(tree_dict['query'])
        size = len(list(PreOrderIter(tree_obj)))
        return [tree_obj, size, tree_dict, item]
    except:
        print(f'couldnot parse query: {item["query"]}')
        return None

def parse_data(data):
    result = []
    result = Parallel(n_jobs=-1)(delayed(parse_item)(item) for item in tqdm(
                        data))
    result = [item for item in result if item is not None]
    return result

parser = argparse.ArgumentParser()
parser.add_argument('--ted_threshold_low', type=float, default=0.0, help='lower limit of tree edit distance')
parser.add_argument('--ted_threshold_high', type=float, default=0.1, help='upper limit of tree edit distance')
parser.add_argument('--source_json', type=str, help='Path to existing training dataset i.e. spider', required=True)
parser.add_argument('--target_json', type=str, help='Path to SQL logs json from target DB', required=True)
parser.add_argument('--output_json', type=str, help='Path to output json containing retrieved examples', required=True)
args = parser.parse_args()

TED_THRESHOLD_LOW = args.ted_threshold_low
TED_THRESHOLD_HIGH = args.ted_threshold_high

SOURCE_JSON = args.source_json
TARGET_JSON = args.target_json
OUTPUT_DUMP = args.output_json
SOURCE_AND_TARGET_SAME = SOURCE_JSON == TARGET_JSON

source_data = json.load(open(SOURCE_JSON))#[0:100]
source_data = parse_data(source_data)
if SOURCE_AND_TARGET_SAME:
    target_data = source_data
else:
    target_data = json.load(open(TARGET_JSON))#[0:100]
    target_data = parse_data(target_data)

src_len, tgt_len = len(source_data), len(target_data)
src_sizes, tgt_sizes = np.zeros(src_len), np.zeros(tgt_len)
dist_matrix = np.ones([src_len, tgt_len]) * 1e9

config = CustomConfig()

idxs = [(i,j) for i in range(src_len) for j in range(tgt_len)]

def process_i_j(idx):
    i, j = idx 
    tree_obj_i, size_i, _, _ = source_data[i]
    if tree_obj_i is None:
        src_sizes[i] = 1e-6 
        return 
    else:
        src_sizes[i] = size_i
    tree_obj_j, size_j, _, _ = target_data[j]
    if tree_obj_j is None:
        tgt_sizes[j] = 1e-6
        return
    else:
        tgt_sizes[j] = size_j
    dist = APTED(tree_obj_i, tree_obj_j, config).compute_edit_distance()
    assert dist >= 0
    if i == j and SOURCE_AND_TARGET_SAME:
        assert dist == 0
    normalized = dist / max(size_j, size_i)
    return normalized

with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
    res = list(tqdm(p.imap(process_i_j, idxs), total=len(idxs)))

for res_i, idx in enumerate(idxs):
    i, j = idx
    normalized = res[res_i]
    dist_matrix[i, j] = normalized

dist_matrix = dist_matrix.T

sqls_to_instances = OrderedDict()

for i in range(tgt_len):
    sql = target_data[i][-1]["query"]
    if sql in sqls_to_instances and len(sqls_to_instances[sql]) > 0:
        print('sql in sqls_to_instances and len(sqls_to_instances[sql]) > 0')
        continue
    distances = dist_matrix[i]
    sim_instances = []
    for j in np.argsort(distances):
        if distances[j] > TED_THRESHOLD_HIGH:
            break
        elif distances[j] < TED_THRESHOLD_LOW:
            continue
        elif j == i and SOURCE_AND_TARGET_SAME:
            assert distances[j] == 0
            continue
        else:
            sim_instance = source_data[j][-1].copy()
            sim_instance['ted'] = distances[j]
            sim_instance.pop('query_toks')
            sim_instance.pop('query_toks_no_value')
            sim_instance.pop('question_toks')
            sim_instance.pop('sql')
            sim_instances.append(sim_instance)
    sqls_to_instances[sql] = sim_instances

json.dump(sqls_to_instances, open(OUTPUT_DUMP,'w'), indent=4)