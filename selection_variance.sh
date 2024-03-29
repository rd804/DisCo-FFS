source ~/.bashrc

# This script is the full pipeline for DisCo-FFS.
conda activate /het/p2/ranit/.conda/disco-ffs

#Top tagging
dataset=tops
pascal_dir=/home/rd804/DisCo-FFS
amarel_dir=/scratch/rd804/DisCo-FFS


exp_name="tops"
temp=results/${exp_name}/


end="done"
try=0
iter=0

while [ ${try} != ${end} ]

do
	while [ ${iter} -ne 21 ]

	do
		TIMESTAMP=`date "+%Y-%m-%d %H:%M:%S"`
		cat >> logs/time_stamps_${exp_name}.txt <<EOF
$TIMESTAMP : Finding feature ${iter} of experiment ${exp_name}
EOF
		# create training data
		python -u scripts/create_training_data.py --tops --iter=${iter} --exp_name=${exp_name}>logs/output/create_training_data.${iter}.${exp_name}.out 2>logs/error/create_training_data.${iter}.${exp_name}.err
		
		# transfer training data to pascal and amarel for training
		ssh rd804@pascal mkdir -p ${pascal_dir}/results/${exp_name}
		ssh rd804@pascal mkdir -p ${pascal_dir}/results/${exp_name}/features		
		scp -r ${temp}features/* rd804@pascal:${pascal_dir}/results/${exp_name}/features/	
		ssh rd804@amarel.rutgers.edu mkdir -p ${amarel_dir}/results/${exp_name}
		ssh rd804@amarel.rutgers.edu mkdir -p ${amarel_dir}/results/${exp_name}/features
		scp -r ${temp}features/* rd804@amarel.rutgers.edu:${amarel_dir}/results/${exp_name}/features/	

		# train classifier on amarel for training variance
	 	ssh rd804@amarel.rutgers.edu "bash -s" <./job_scripts/remote_commands.sh ${iter} ${exp_name} ${dataset}

	 # obtain classifer out	
	 	ssh rd804@pascal /home/rd804/.conda/envs/disco-ffs/bin/python -u ${pascal_dir}/scripts/classifier_training.py --tops --iter=${iter} --exp_name=${exp_name}>logs/output/classifier_training.${iter}.${exp_name}.out 2>logs/error/classifier_training.${iter}.${exp_name}.err
	 # transfer classifier output set
	 	mkdir -p ${temp}/ypred
	 	scp -r rd804@pascal:${pascal_dir}/results/${exp_name}/ypred/* ${temp}ypred/		
	 	mkdir -p ${temp}/discor
	 	mkdir -p ${temp}/discor/iteration_${iter}
		arr=1
	 # calculate score
		while ((${#arr[@]}))
		do
			echo ${arr[@]}
	 		condor_submit iter=${iter} exp_name=${exp_name} dataset=${dataset} job_scripts/compute_scores.jdl.base
	 # wait for condor to finish job
	 		condor_wait /het/p2/ranit/DisCo-FFS/logs/logs/findvar_7k_${iter}.log
	 		rm /het/p2/ranit/DisCo-FFS/logs/logs/findvar_7k_${iter}.log
			
			arr=()
			for split in {0..349..1}
			do
				if [[ ! -f /het/p2/ranit/DisCo-FFS/results/${exp_name}/discor/iteration_${iter}/dis_cor_${split}.txt ]]
				
				then
					arr+=("${split}")
				fi

			done
	

		done	
	 # find highest score
	 	python -u scripts/sort_scores_add_new_feature.py --iter=${iter} --exp_name=${exp_name} --tops>./logs/output/sort.${iter}.${exp_name}.out 2>./logs/error/sort.${iter}.${exp_name}.err	

	
	 	iter=$((${iter}+1))

	# transfer efps to amarel for training variance			
	echo "done with feature ${iter}"
	done
	iter=0	 
echo "Done with try ${try} !!!"

try=${end}

done

echo "Done with all tries !!"
