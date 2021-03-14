#!/usr/bin/env bash
cd ../..
source online-cl/bin/activate
python -u multiple_run.py --general config_CVPR/general.yml --data config_CVPR/data/cifar10/cifar10_nc.yml --agent config_CVPR/agent/ewc/ewc_ncm.yml
