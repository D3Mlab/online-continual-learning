#!/usr/bin/env bash
cd ../..
source online-cl/bin/activate
python -u multiple_run.py --general config_CVPR/general.yml --data config_CVPR/data/cifar100/cifar100_nc.yml --agent config_CVPR/agent/lwf/lwf_ncm.yml