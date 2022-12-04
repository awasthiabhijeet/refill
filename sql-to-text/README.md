# SQL-to-Text

Each directory contains the implementation of a specific SQL-to-Text model

* [`gazp`](./gazp/) -- contains code for ["Grounded Adaptation for Zero-shot Executable Semantic Parsing" (Zhong et al. EMNLP 2020)](https://aclanthology.org/2020.emnlp-main.558/)
* [`l2s`](./l2s/) -- contains code for ["Learning to Synthesize Data for Semantic Parsing" (Wang et al. NAACL 2021)](https://aclanthology.org/2021.naacl-main.220/)
* [`snowball`](./snowball/) -- contains code for ["Logic-Consistency Text Generation from Semantic Parses" (Shu et al. ACL 2021)](https://aclanthology.org/2021.findings-acl.388/)

Each implementation contains code that :

* trains the particular SQL-to-Text model
* uses the trained SQL-to-Text model to generate data for new workloads

Relevant bash scripts that use these implementations for training and inferring respective models can be found in [`../scripts/sql-to-text/`](../scripts/sql-to-text/) directory
