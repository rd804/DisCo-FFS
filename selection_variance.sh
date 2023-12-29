source ~/.bashrc

# This script is the full pipeline for DisCo-FFS.
conda activate /het/p2/ranit/.conda/disco-ffs
#Quark Gluon tagging
#temp=/het/p1/ranit/qg/disco_ffs/temp/
#pascal_dir=/home/rd804/qg/disco_ffs/temp/

#Top tagging
dataset=tops
#temp=./temp/
pascal_dir=/home/rd804/DisCo-FFS


#exp_name="m_pt_mw_efp_bip_test"
exp_name="test"
temp=results/${exp_name}/


end="done"
try=0
iter=0

while [ ${try} != ${end} ]

do
	while [ ${iter} -ne 1 ]

	do
		TIMESTAMP=`date "+%Y-%m-%d %H:%M:%S"`
		cat >> logs/time_stamps_${exp_name}.txt <<EOF
$TIMESTAMP : Finding feature ${iter} of experiment ${exp_name}
EOF
		python -u scripts/create_training_data.py --tops --iter=${iter} --exp_name=${exp_name}>logs/output/create_training_data.${iter}.${exp_name}.out 2>logs/error/create_training_data.${iter}.${exp_name}.err		
	 	scp -r ${temp} rd804@pascal:${pascal_dir}/results/	
	# 	scp -r ${temp}features/*labels* rd804@pascal:${pascal_dir}features/	
	# 	scp -r ${temp}features/*labels* rd804@amarel.rutgers.edu:/scratch/rd804/training_variance_checks/temp/features/
	# 	scp -r ${temp}features/*_${iter}_*$I* rd804@amarel.rutgers.edu:/scratch/rd804/training_variance_checks/temp/features/
	# 	ssh rd804@amarel.rutgers.edu "bash -s" <./remote_commands.sh ${iter} $I tops

	# # obtain classifer out	
	 	ssh rd804@pascal /home/rd804/.conda/envs/disco-ffs/bin/python -u ${pascal_dir}/scripts/classifier_training.py --tops --iter=${iter} --exp_name=${exp_name}>logs/output/classifier_training.${iter}.${exp_name}.out 2>logs/error/classifier_training.${iter}.${exp_name}.err
	# # transfer classifier output set
	# 	scp -r rd804@pascal:${pascal_dir}/ypred_batch ./temp/		
	# 	mkdir -p ./temp/discor_$I
	# 	mkdir -p ./temp/discor_$I/iteration_${iter}
	# # calculate score
	# 	condor_submit M=${iter} I=$I dataset=${dataset} find_next_variable.jdl.base
	# #wait for condor to finish job
	# 	condor_wait /het/p1/ranit/tops/training_method_ypred_cut_04_06_changing_threshold/logs_selection_variance/logs/findvar_7k_$I.log
	# 	rm ./logs_selection_variance/logs/findvar_7k_$I.log	
	# # find highest score
	# 	python -u sort_scores_add_new_feature.py ${iter} $I -tops>./logs_selection_variance/output/sort.${iter}.$I.out 2>./logs_selection_variance/error/sort.${iter}.$I.err	

	
	# # transfer efps to pascal for training	
	 	iter=$((${iter}+1))

	# transfer efps to amarel for training variance			
	echo "done with feature ${iter}"
	done
	iter=0	 
echo "Done with try ${try} !!!"

try=${end}

done

echo "Done with all tries !!"
