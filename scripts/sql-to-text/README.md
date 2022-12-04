# SQL-to-Text Scripts

Note : Please run each script from this directory only since it contains relative paths

Here, each `<X>` represents an SQL-to-Text method

* `gazp` -- represents ["Grounded Adaptation for Zero-shot Executable Semantic Parsing" (Zhong et al. EMNLP 2020)](https://aclanthology.org/2020.emnlp-main.558/)
* `l2s` -- represents ["Learning to Synthesize Data for Semantic Parsing" (Wang et al. NAACL 2021)](https://aclanthology.org/2021.naacl-main.220/)
* `snowball` -- represents ["Logic-Consistency Text Generation from Semantic Parses" (Shu et al. ACL 2021)](https://aclanthology.org/2021.findings-acl.388/)

Each `train_<X>.sh` script does the following :

* Preprocesses the spider data appropriately and stores the preprocessed data in [`../../data/sql-to-text/<X>/`](../../data/sql-to-text/) directory
* Trains the appropriate model using the preprocessed data and saves the intermediate and best checkpoints in `../models/sql-to-text/<X>/` directory

Each `infer_<X>.sh` script does the following :

* Reads spider workloads from [`../../data/spider_groups/`](../../data/spider_groups/)
* Generates text corresponding to each SQL using appropriate SQL-to-Text method (uses model checkpoints from the location where `train_<X>.sh` saves them)
* Splits the (SQL, generated-text) pairs into train-dev splits for further SmBop finetuning

Implementation of each method can be found in [`../../sql-to-text/`](../../sql-to-text/) directory
