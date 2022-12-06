# Diverse Parallel Data Synthesis for Cross-Database Adaptation of Text-to-SQL Parsers (EMNLP 2022) 

**[WIP]** This repository provides an implementation of experiments in our [EMNLP-22 paper](https://arxiv.org/abs/2210.16613	)
```
@article{awasthi2022diverse,
  title={Diverse Parallel Data Synthesis for Cross-Database Adaptation of Text-to-SQL Parsers},
  author={Awasthi, Abhijeet and Sathe, Ashutosh and Sarawagi, Sunita},
  journal={arXiv preprint arXiv:2210.16613},
  year={2022}
}
```  

# Requirements
This code was developed with python 3.8.8. <br/>
Create a new virtual environment and install the dependencies by running `pip install -r requirements.txt`

# Datasets
 - [Spider dataset](https://github.com/taoyds/spider): useful files copied in `data/spider`
 - ReFill generated datasets: `data/sql-to-text/refill/jsons/spider_groups`

# ReFill Pipeline

  * Preprocessing: Apply Masking, Convert SQLs into Pseudo-English form, Pre-compute SQL-neighbours for train and val set
    ```
    bash scripts/data/refill_postprocess.sh
    ```
  * Train BART model for ReFill
    ```
    bash scripts/sql-to-text/train_refill.sh
    ```
  * Train Filtering model to filter out inconsistent SQL-Text pairs
    ```
    bash scripts/sql-to-text/train_filter.sh
    ```
  * ReFill Inference: Find SQL-neighbours of the given workload, Apply Masking and ReFilling followed by filtering
    ```
    bash scripts/sql-to-text/infer_refill.sh
    ```

# L2S Pipeline

This pipeline makes use of relative paths. It is recommended to change directory to [`scripts/sql-to-text/`](scripts/sql-to-text) first before running any script

  * Preprocessing + Training: Convert SQLs into L2S encoding and train a Seq2Seq model
    ```
    bash train_l2s.sh
    ```
  * L2S Inference: Use the trained SQL-to-Text Seq2Seq model to generate text for the given workload
    ```
    bash infer_l2s.sh
    ```

# GAZP Pipeline

This pipeline makes use of relative paths. It is recommended to change directory to [`scripts/sql-to-text/`](scripts/sql-to-text) first before running any script

  * Preprocessing + Training: Convert SQLs into GAZP encoding and train a Seq2Seq model
    ```
    bash train_gazp.sh
    ```
  * GAZP Inference + Filtering: Use the trained SQL-to-Text Seq2Seq model to generate text for the given workload and use a forward Text-to-SQL parser for cycle consistency based filtering
    ```
    bash infer_gazp.sh
    ```

# SnowBall Pipeline

This pipeline makes use of relative paths. It is recommended to change directory to [`scripts/sql-to-text/`](scripts/sql-to-text) first before running any script

  * Preprocessing + Training: Convert SQLs into SnowBall encoding and train a Seq2Seq model
    ```
    bash train_snowball.sh
    ```
  * SnowBall Inference: Use the trained SQL-to-Text Seq2Seq model to generate text for the given workload
    ```
    bash infer_snowball.sh
    ```
