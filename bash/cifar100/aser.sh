#!/usr/bin/env bash
cd ../..
source online-cl/bin/activate
python -u multiple_run.py --general config_CVPR/general.yml --data config_CVPR/data/cifar100/cifar100_nc.yml --agent config_CVPR/agent/aser/aser_1k_cifar100.yml
python -u multiple_run.py --general config_CVPR/general.yml --data config_CVPR/data/cifar100/cifar100_nc.yml --agent config_CVPR/agent/aser/aser_2k_cifar100.yml
python -u multiple_run.py --general config_CVPR/general.yml --data config_CVPR/data/cifar100/cifar100_nc.yml --agent config_CVPR/agent/aser/aser_5k_cifar100.yml
