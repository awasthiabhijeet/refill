# Using [L2S framework](https://aclanthology.org/2021.naacl-main.220/) for SQL-to-Text

Scripts below require [tensor2struct](https://github.com/berlino/tensor2struct-public) to be installed first

* [`bleu.py`](bleu.py) -- [util,train] BLEU calculation code obtained from [NLTK](https://www.nltk.org/). This is used to checkpoint the best SQL-to-Text model by [`train_bart_l2s.py`](train_bart_l2s.py)
* [`convert_to_l2s_format.py`](convert_to_l2s_format.py) -- [util,train] Reads spider JSONs, converts each SQL to L2S encoding and dumps the encoding back to disk. The converted JSONs are used to train the SQL-to-Text model by [`train_bart_l2s.py`](train_bart_l2s.py). The L2S encoding is essentially parsing the SQL using the tensor2struct and unparsing it again using PCFG to get a consistent SQL representation
* [`train_bart_l2s.py`](train_bart_l2s.py) -- [train] Trains a [BART](https://huggingface.co/docs/transformers/main/en/model_doc/bart#bart) model on the L2S encoded spider (obtained using [`convert_to_l2s_format.py`](convert_to_l2s_format.py)). Uses the BLEU score (obtained using [`bleu.py`](bleu.py)) to keep the best checkpoint
* [`create_finetune_json.py`](create_finetune_json.py) -- [infer] Reads every workload, converts them to L2S encoding, uses the pretrained model to infer the corresponding text and finally splits the (SQL,generated text) pairs appropriately for SmBop finetuning
* [`process_sql.py`](process_sql.py) -- [util,infer] SQL processing utilities. Obtained from [spider](https://github.com/taoyds/spider/blob/master/process_sql.py)'s preprocessing utilities
