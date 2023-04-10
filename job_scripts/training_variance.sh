#!/bin/bash

source ~/.bashrc

name=$2
I=$1

data=$3

mkdir -p ./performance
mkdir -p ./performance/r30_variance_${name}

#for I in {1..10..1}
#do
arr=1	

#mkdir -p ./r30_variance_${name}/r30_$I
mkdir -p ./model/model_${name}

while ((${#arr[@]}))
do
	arr=()
	for j in {0..9..1}
	do
		if [[ ! -f /scratch/rd804/training_variance_checks/performance/r30_variance_${name}/r30_$I/r30_${j}.npy ]]
		
		then
			arr+=("$j")
		fi

	done
	
	echo ${arr[@]}
	for j in ${arr[@]}
	do
		sbatch -W --output=./logs_training/output/find_var.$I.$j.${name}.out --error=./logs_training/error/find_var.$I.$j.${name}.err --export=I=$I,j=$j,name=${name} job_array_training.sh $j $I ${name} ${data} &
	done
	wait

done
echo $I	

#done

