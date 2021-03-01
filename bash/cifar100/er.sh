#!/usr/bin/env bash
cd ../..
source cl-torch/bin/activate
python -u multiple_run.py --general config_CVPR/general.yml --data config_CVPR/data/cifar100/cifar100_nc.yml --agent config_CVPR/agent/er/er_1k.yml
python -u multiple_run.py --general config_CVPR/general.yml --data config_CVPR/data/cifar100/cifar100_nc.yml --agent config_CVPR/agent/er/er_2k.yml
python -u multiple_run.py --general config_CVPR/general.yml --data config_CVPR/data/cifar100/cifar100_nc.yml --agent config_CVPR/agent/er/er_5k.yml
