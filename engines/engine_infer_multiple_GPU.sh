#!/usr/bin/env bash

PYTHON=/home/yuan/.conda/envs/py36/bin/python

$PYTHON -m torch.distributed.launch --nproc_per_node=3 engines/engine_infer_Classification_multiple_GPU.py