# Using [SnowBall framework](https://aclanthology.org/2021.findings-acl.388/) for SQL-to-Text


* [`bleu.py`](bleu.py) -- [util,train] BLEU calculation code obtained from [NLTK](https://www.nltk.org/). This is used to checkpoint the best SQL-to-Text model by [`train_bart_snowball.py`](train_bart_snowball.py)
* [`convert_to_snowball_format.py`](convert_to_snowball_format.py) -- [util,train] Reads spider JSONs, converts each SQL to SnowBall encoding and dumps the encoding back to disk. The converted JSONs are used to train the SQL-to-Text model by [`train_bart_snowball.py`](train_bart_snowball.py). Main implementation of the encoding is present in [`sql_formatter`](sql_formatter/) directory
* [`train_bart_snowball.py`](train_bart_snowball.py) -- [train] Trains a [BART](https://huggingface.co/docs/transformers/main/en/model_doc/bart#bart) model on the SnowBall encoded spider (obtained using [`convert_to_snowball_format.py`](convert_to_snowball_format.py)). Uses the BLEU score (obtained using [`bleu.py`](bleu.py)) to keep the best checkpoint
* [`create_finetune_json.py`](create_finetune_json.py) -- [infer] Reads every workload, converts them to SnowBall encoding, uses the pretrained model to infer the corresponding text and finally splits the (SQL,generated text) pairs appropriately for SmBop finetuning
* [`process_sql.py`](process_sql.py) -- [util,infer] SQL processing utilities. Obtained from [spider](https://github.com/taoyds/spider/blob/master/process_sql.py)'s preprocessing utilities
