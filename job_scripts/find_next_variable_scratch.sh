#!/bin/bash --posix


MYDIR=/users/h2/ranit/Wtagging/
MYHETDIR=/het/p1/ranit/
seed=$1 
M=$2
I=$3

cd ${MYHETDIR}/tops/training_method_ypred_cut_04_06_changing_threshold/




python find_next_variable.py ${seed} $M $I -s
