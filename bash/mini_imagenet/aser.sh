#!/usr/bin/env bash
cd ../..
source online-cl/bin/activate
python -u multiple_run.py --general config_CVPR/general.yml --data config_CVPR/data/mini_imagenet/mini_imagenet_nc.yml --agent config_CVPR/agent/aser/aser_1k_mini.yml
python -u multiple_run.py --general config_CVPR/general.yml --data config_CVPR/data/mini_imagenet/mini_imagenet_nc.yml --agent config_CVPR/agent/aser/aser_2k_mini.yml
python -u multiple_run.py --general config_CVPR/general.yml --data config_CVPR/data/mini_imagenet/mini_imagenet_nc.yml --agent config_CVPR/agent/aser/aser_5k_mini.yml
