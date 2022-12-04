# Apply masking to train and dev examples
python3 sql-to-text/refill/freq_based_schema_masking.py

# Convert SQLs into pseudo-English representations
python3 sql-to-text/refill/augment_with_pseudo_eng_reps.py

# Pre-compute train-train neighbours
python3 sql-to-text/refill/compute_neighbours.py \
	--source_json=data/spider/train_spider.json \
	--target_json=data/spider/train_spider.json	 \
	--output_json=data/sql-to-text/refill/jsons/train_train_nbrs.json

# Pre-compute val-train neighbours
python3 sql-to-text/refill/compute_neighbours.py \
	--source_json=data/spider/train_spider.json \
	--target_json=data/spider/dev.json	 \
	--output_json=data/sql-to-text/refill/jsons/val_train_nbrs.json

