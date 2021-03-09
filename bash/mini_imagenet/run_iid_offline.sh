#!/usr/bin/env bash
sbatch --nodes=1 --time=27:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner iid_offline.sh
