#!/bin/bash
set -e
set -x

CURR_DIR="$(pwd)"

GAZP_ROOT='../../sql-to-text/gazp/'
GAZP_CKPT='../../models/sql-to-text/run-XXX/checkpoint-XXXX/'
INPUT_GROUPS_DIR='../../data/spider_groups/'
OUTPUT_GROUPS_DIR='../../data/sql-to-text/gazp/smbop_jsons/'

SMBOP_ROOT='../../text-to-sql/'
SPIDER_DATABASE='../../data/spider/database/'
SPIDER_TABLES='../../data/spider/tables.json'
SMBOP_CKPT='../../models/text-to-sql/model.tar.gz'

FILTER_THRESHOLD=10

# Generate using SQL-to-Text
cd $GAZP_ROOT
for frac in 0.3 0.5 0.7
do
    for group in group_1 group_2 group_3 group_4
    do
        mkdir -p $OUTPUT_GROUPS_DIR/$frac/$group
        python get_query_encodings.py $INPUT_GROUPS_DIR/$frac/$group/perturbed_train_queries.json
        mv $INPUT_GROUPS_DIR/$frac/$group/perturbed_train_queries_sql_enc.json $OUTPUT_GROUPS_DIR/$frac/$group/
        # aug.json is just a dummy empty file
        python generate_sql_to_text.py \
            --model_path $GAZP_CKPT \
            --input_queries_train $OUTPUT_GROUPS_DIR/$frac/$group/perturbed_train_queries_sql_enc.json \
            --input_queries_aug aug.json \
            --output_file $OUTPUT_GROUPS_DIR/$frac/$group/${group}_train_gazp_output.json
    done
done

cd $CURR_DIR

# Filter using Text-to-SQL
for frac in 0.3 0.5 0.7
do
    for group in group_1 group_2 group_3 group_4
    do
        jq -r '.[] | "\(.query)\t\(.db_id)"' "$OUTPUT_GROUPS_DIR/$frac/$group/${group}_train_gazp_output.json" > "$OUTPUT_GROUPS_DIR/$frac/$group/${group}_text2sql-gold.sql"
        rm -rf cache
        mkdir -p cache 
        python "$SMBOP_ROOT/eval.py" --archive_path $SMBOP_CKPT --dev_path "$OUTPUT_GROUPS_DIR/$frac/$group/${group}_train_gazp_output.json" --table_path $SPIDER_TABLES --dataset_path $SPIDER_DATABASE --output "$OUTPUT_GROUPS_DIR/$frac/$group/${group}_text2sql-eval.sql"
        python "$SMBOP_ROOT/smbop/eval_final/evaluation.py" --gold "$OUTPUT_GROUPS_DIR/$frac/$group/${group}_text2sql-gold.sql" --pred "$OUTPUT_GROUPS_DIR/$frac/$group/${group}_text2sql-eval.sql" --etype all --db $SPIDER_DATABASE --table $SPIDER_TABLES
        mv entries_list.json "$OUTPUT_GROUPS_DIR/$frac/$group/${group}_entries_list.json"
    done
done

# Generate finetuning splits
python split.py --threshold $FILTER_THRESHOLD --root_dir $OUTPUT_GROUPS_DIR
