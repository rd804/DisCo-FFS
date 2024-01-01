#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --job-name=cross_validation
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G   # memory per cpu-core


source ~/.bashrc
conda activate disco-ffs

split=$1
iter=$2
exp_name=$3
data=$4

cd /scratch/rd804/DisCo-FFS/


python scripts/training_variance.py --split=${split} --iter=${iter} --exp_name=${exp_name} --${data}
