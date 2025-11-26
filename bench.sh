#!/bin/bash

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
CUDA_VISIBLE_DEVICES=1 python /projects/delight/ivan/HEST/src/hest/bench/benchmark.py \
    --config /projects/delight/ivan/HEST/bench_config/bench_3x3_graph_prad_config.yaml