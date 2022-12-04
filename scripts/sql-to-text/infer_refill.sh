GROUPS_DIR="data/spider_groups"
REFILL_DATA_DIR="data/sql-to-text/refill/jsons/spider_groups"
export PYTHONPATH=$PWD:$PYTHONPATH

for frac in 0.3 0.5 0.7
do
  for group_id in group_1 group_2 group_3 group_4
  do
  	mkdir -p $REFILL_DATA_DIR/$frac/$group_id
  	python3 sql-to-text/refill/compute_neighbours.py \
  	  --source_json=data/spider/train_spider.json \
  	  --target_json=$GROUPS_DIR/$frac/$group_id/perturbed_train_queries.json	 \
  	  --output_json=$REFILL_DATA_DIR/$frac/$group_id/perturbed_train_queries_with_retrieved_examples.json
  done
done	

for frac in 0.3 0.5 0.7
do
  for group_id in group_1 group_2 group_3 group_4
  do
  	python3 -u sql-to-text/refill/infer_refill.py \
		--model_path models/sql-to-text/refill/template-improved-blec-th-70-aug-simp \
		--input_json $GROUPS_DIR/$frac/$group_id/perturbed_train_queries.json \
		--nbrs_json $REFILL_DATA_DIR/$frac/$group_id/perturbed_train_queries_with_retrieved_examples.json \
		--templates_json data/sql-to-text/refill/jsons/train_th_70.json \
		--output_json $REFILL_DATA_DIR/$frac/$group_id/filled_templates_th_70.json \
		--tables_json data/spider/tables.json \
		--num_nbrs 30
  done
done

for frac in 0.3 0.5 0.7
do
  for group_id in group_1 group_2 group_3 group_4
  do
  	python3 -u sql-to-text/refill/filter_and_create_splits.py $REFILL_DATA_DIR/$frac/$group_id/filled_templates_th_70.json
  done
done	