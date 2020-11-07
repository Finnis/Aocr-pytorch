#! /usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 \
python -m torch.distributed.launch \
--nproc_per_node=1 \
--nnodes=1 \
--node_rank=0 \
train.py -c config.yaml

pkill -f torch/bin/python  # Kill previous died processes