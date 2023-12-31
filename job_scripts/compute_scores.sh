#!/bin/bash --posix

MYHETDIR=/het/p2/ranit/
seed=$1 
iter=$2
exp_name=$3
dataset=$4
cd ${MYHETDIR}DisCo-FFS

source ~/.bashrc
conda activate /het/p2/ranit/.conda/disco-ffs


python scripts/compute_scores.py --parallel_index=${seed} --iter=${iter} --exp_name=${exp_name} --${dataset} --parallel_step=21
