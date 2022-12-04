#!/bin/bash
set -e
set -x

L2S_IMPL_DIR='../../sql-to-text/l2s/'

cd $L2S_IMPL_DIR
python convert_to_l2s_format.py
python train_bart_l2s.py
