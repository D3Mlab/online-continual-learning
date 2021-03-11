#!/usr/bin/env bash
cd ../..
source online-cl/bin/activate
python -u multiple_run.py --general config_CVPR/general.yml --data config_CVPR/data/mini_imagenet/mini_imagenet_nc.yml --agent config_CVPR/agent/gss/gss_1k_ncm.yml
python -u multiple_run.py --general config_CVPR/general.yml --data config_CVPR/data/mini_imagenet/mini_imagenet_nc.yml --agent config_CVPR/agent/gss/gss_2k_ncm.yml
python -u multiple_run.py --general config_CVPR/general.yml --data config_CVPR/data/mini_imagenet/mini_imagenet_nc.yml --agent config_CVPR/agent/gss/gss_5k_ncm.yml
