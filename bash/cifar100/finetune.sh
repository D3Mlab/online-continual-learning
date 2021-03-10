#!/usr/bin/env bash
cd ../..
source online-cl/bin/activate
python -u multiple_run.py --general config_CVPR/general_finetune.yml --data config_CVPR/data/cifar100/cifar100_nc.yml --agent config_CVPR/agent/er/finetune.yml
