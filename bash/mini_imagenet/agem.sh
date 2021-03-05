#!/usr/bin/env bash
cd ../..
source online-cl/bin/activate
python -u multiple_run.py --general config_CVPR/general.yml --data config_CVPR/data/mini_imagenet/mini_imagenet_nc.yml --agent config_CVPR/agent/agem/agem_1k.yml
python -u multiple_run.py --general config_CVPR/general.yml --data config_CVPR/data/mini_imagenet/mini_imagenet_nc.yml --agent config_CVPR/agent/agem/agem_2k.yml
python -u multiple_run.py --general config_CVPR/general.yml --data config_CVPR/data/mini_imagenet/mini_imagenet_nc.yml --agent config_CVPR/agent/agem/agem_5k.yml
