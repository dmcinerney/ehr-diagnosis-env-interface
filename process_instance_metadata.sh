#!/bin/bash
#SBATCH --partition=frink
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
source activate /work/frink/mcinerney.de/envs/ehrenvint
python process_instance_metadata.py
