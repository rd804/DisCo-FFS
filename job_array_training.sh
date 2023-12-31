#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --job-name=cross_validation
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G   # memory per cpu-core
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=1G


source ~/.bashrc
conda activate disco-ffs


variable=$2
split=$1
name=$3
data=$4

cd /scratch/rd804/training_variance_checks/


python training_variance.py $split $variable ${name} -$data
