#!/bin/bash

source ~/.bashrc

iter=$1
exp_name=$2
data=$3

#mkdir -p ./performance
#mkdir -p ./performance/r30_variance_${name}

#for I in {1..10..1}
#do
arr=1	

#mkdir -p ./r30_variance_${name}/r30_$I
#mkdir -p ./model/model_${name}

while ((${#arr[@]}))
do
	arr=()
	for split in {0..9..1}
	do
		if [[ ! -f /scratch/rd804/DisCo-FFS/results/${exp_name}/r30_variance/r30_${iter}/r30_${split}.npy ]]
		
		then
			arr+=("${split}")
		fi

	done
	
	echo ${arr[@]}
	for j in ${arr[@]}
	do
		sbatch -W --output=./logs/output/find_var.iter_${iter}.split_${split}.name_${exp_name}.out --error=./logs/error/find_var.iter_${iter}.split_${split}.name_${exp_name}.err --export=split=${split},iter=${iter},exp_name=${exp_name} job_array_training.sh ${split} ${iter} ${exp_name} ${data} &
	done
	wait

done
echo $I	

#done

