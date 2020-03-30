#!/usr/bin/env bash
set -e

# bert-serving-start \
#   -model_dir=/app/base-models/uncased_L-12_H-768_A-12/ \
#   -tuned_model_dir=/app/trained-models/engadget_L-12_H-768_A-12/ \
#   -ckpt_name=model.ckpt-43000 \
#   -num_worker=1 \
#   -http_port 9000 \
#   -cpu \
#   -pooling_strategy=REDUCE_MEAN_MAX \
#   -max_batch_size 16 \
#   -verbose

bert-serving-start \
    -model_dir models/english_L-12_H-768_A-12/ \
    -num_worker=1 \
    -model_dir=models/uncased_L-12_H-768_A-12
