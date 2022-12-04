# data

This directory will host all the data used for experiments. [`spider_groups`](./spider_groups/) directory contains the splits we used for our adaptation experiments on spider dataset.

The organization of this folder is as follows :

```
spider_groups
├── <frac>
│   ├── group_<X>
│   │   ├── disjoint_test.json
│   │   ├── disjoint_test_queries.json
│   │   ├── original_train_queries.json
│   │   ├── perturbed_train_queries.json
│   │   ├── test.json
│   │   ├── test_queries.json
│   │   ├── test.sql
│   │   └── train.sql
... ... 
```

Each group contains a schema from spider's dev set. e.g. `group_1` consists of `singer`, `orchestra` and `concerts`. 

* `test.json` -- Collection of all examples in the group from spider's dev set 
* `test_queries.json` -- `test.json` but with question related information removed
* `original_train_queries.json` -- Subset of `<frac> * len(test_queries.json)` queries from `test_queries.json`
* `perturbed_train_queries.json` -- `original_train_queries.json` queries but with db values replaced with other values from same db. These queries simulate our workload in each case
* `disjoint_test.json` -- Examples in `test.json` whose queries are not in `original_train_queries.json`. `test.json \ original_train_queries.json`
* `disjoint_test_queries.json` -- `disjoint_test.json` but with question related information removed
* `test.sql` -- A text file with `<query><tab><db_id>` on every line corresponding to `test.json`. Here, `<tab>` represents the tab (`\t`) character. Directly used for EM/EX evaluation
* `train.sql` -- A text file with `<query><tab><db_id>` on every line corresponding to `perturbed_train_queries.json`. Here `<tab>` represents the tab (`\t`) character

