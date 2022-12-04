#!/bin/bash
set -e
set -x

SNOWBALL_IMPL_DIR='../../sql-to-text/snowball/'
SNOWBALL_CKPT='../../models/sql-to-text/run-XXX/checkpoint-XXXX/'
INPUT_GROUPS_DIR='../../data/spider_groups/'
OUTPUT_GROUPS_DIR='../../data/sql-to-text/snowball/smbop_jsons/'

mkdir -p $OUTPUT_GROUPS_DIR

cd $SNOWBALL_IMPL_DIR

for frac in 0.3 0.5 0.7
do
    echo "Generating for with frac=$frac"
    python create_finetune_json.py \
        --model-path $SNOWBALL_CKPT \
        --root-dir-input $INPUT_GROUPS_DIR/$frac/ \
        --root-dir-output $OUTPUT_GROUPS_DIR/$frac/ \
        --preproc snowball 
done
