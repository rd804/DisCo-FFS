source ~/.bashrc

# This script is the full pipeline for DisCo-FFS.

#Quark Gluon tagging
#temp=/het/p1/ranit/qg/disco_ffs/temp/
#pascal_dir=/home/rd804/qg/disco_ffs/temp/

#Top tagging
dataset=tops
temp=./temp/
pascal_dir=/home/rd804/ypred_method/temp/


exp_name="m_pt_mw_efp_bip_test"
M=0
end="done"

while [ $I != ${end} ]

do
	while [ ${iter} -ne 21 ]

	do
		TIMESTAMP=`date "+%Y-%m-%d %H:%M:%S"`
		cat >> ./logs_selection_variance/time_stamps_${exp_name}.txt <<EOF
$TIMESTAMP : Finding feature ${iter} of experiment ${exp_name}
EOF
		python -u scripts/create_training_data.py -tops ${iter} $I>./logs_selection_variance/output/create_training_data.${iter}.$I.out 2>./logs_selection_variance/error/create_training_data.${iter}.$I.err		
		scp -r ${temp}features/*$I* rd804@pascal:${pascal_dir}features/	
		scp -r ${temp}features/*labels* rd804@pascal:${pascal_dir}features/	
		scp -r ${temp}features/*labels* rd804@amarel.rutgers.edu:/scratch/rd804/training_variance_checks/temp/features/
		scp -r ${temp}features/*_${M}_*$I* rd804@amarel.rutgers.edu:/scratch/rd804/training_variance_checks/temp/features/
		ssh rd804@amarel.rutgers.edu "bash -s" <./remote_commands.sh ${iter} $I tops

	# obtain classifer out	
		ssh rd804@pascal /opt/anaconda3/bin/python -u ./ypred_method/classifier_training.py -tops ${iter} $I>./logs_selection_variance/output/classifier_training.${iter}.$I.out 2>./logs_selection_variance/error/classifier_training.${iter}.$I.err
	# transfer classifier output set
		scp -r rd804@pascal:${pascal_dir}/ypred_batch ./temp/		
		mkdir -p ./temp/discor_$I
		mkdir -p ./temp/discor_$I/iteration_${iter}
	# calculate score
		condor_submit M=${iter} I=$I dataset=${dataset} find_next_variable.jdl.base
	#wait for condor to finish job
		condor_wait /het/p1/ranit/tops/training_method_ypred_cut_04_06_changing_threshold/logs_selection_variance/logs/findvar_7k_$I.log
		rm ./logs_selection_variance/logs/findvar_7k_$I.log	
	# find highest score
		python -u sort_and_find_new_variable.py ${iter} $I -tops>./logs_selection_variance/output/sort.${iter}.$I.out 2>./logs_selection_variance/error/sort.${iter}.$I.err	

	
	# transfer efps to pascal for training	
		M=$((${iter}+1))

	# transfer efps to amarel for training variance			
	echo "done with feature ${iter}"
	done
	M=0	 
echo "Done with iteration $I !!!"

I=${end}

done

echo "Done with all iterations !!"
