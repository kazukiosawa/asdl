#!/bin/bash
#YBATCH -r a100_1
#SBATCH -N 1
#SBATCH -o /home/lfsm/code/exp_result/logs/job.%j.out
#SBATCH --time=72:00:00
#SBATCH -J train_vitb16_with_kfac
#SBATCH --error /home/lfsm/code/exp_result/logs/%j.err
source activate
conda deactivate
conda activate torch
python3 /home/lfsm/code/sun_asdfghjkl/examples/train_vitb16_with_kfac.py --data-path /mnt/nfs/datasets/ILSVRC2012 --batch-size 256 --log-path /home/lfsm/code/exp_result/vitb16
