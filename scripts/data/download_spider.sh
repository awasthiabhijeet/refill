#!/bin/bash
set -e

SPIDER_URL='https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ'

gdown $SPIDER_URL
unzip spider.zip
mv spider ../../data/
