#!/bin/bash
set -e
set -x

SNOWBALL_IMPL_DIR='../../sql-to-text/snowball/'

cd $SNOWBALL_IMPL_DIR
python convert_to_snowball_format.py
python train_bart_snowball.py
