# Using [GAZP framework](https://aclanthology.org/2020.emnlp-main.558/) for SQL-to-Text

* [`split.py`](split.py) -- [util, infer] Utility that goes through the scores from Text-to-SQL parser produces for filtering the generated text. After the filtering is complete, this script will also prepare train-dev splits for further SmBop finetuning
* [`utils/process_sql.py`](utils/process_sql.py) -- [util] SQL preprocessing code adapted from [spider](https://github.com/taoyds/spider/blob/master/process_sql.py)'s repo

Rest of the files in this directory are borrowed from the official implementation of [GAZP](https://github.com/vzhong/gazp)
