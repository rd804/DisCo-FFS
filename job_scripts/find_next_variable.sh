#!/bin/bash --posix

MYHETDIR=/het/p1/ranit/
seed=$1 
M=$2
I=$3
dataset=$4
cd ${MYHETDIR}/tops/training_method_ypred_cut_04_06_changing_threshold/




python compute_scores.py ${seed} $M $I -${dataset}
