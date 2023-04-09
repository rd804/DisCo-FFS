source ~/.bashrc


I="scratch_efp_mpt"
M=0
end="done"
#rm -f ./logs/time_stamps.txt
while [ $I != ${end} ]

do
	while [ $M -ne 21 ]

	do
		TIMESTAMP=`date "+%Y-%m-%d %H:%M:%S"`
		cat >> ./logs_selection_variance/time_stamps_$I.txt <<EOF
$TIMESTAMP : starting with variable $M iteration $I
EOF
		
		mkdir -p ./temp/discor_$I
		mkdir -p ./temp/discor_$I/iteration_$M
	# calculate score
		condor_submit M=$M I=$I find_next_variable_scratch.jdl.base
	# wait for condor to finish job
		condor_wait /het/p1/ranit/tops/training_method_ypred_cut_04_06_changing_threshold/logs_selection_variance/logs/findvar_7k_$I.log
		rm ./logs_selection_variance/logs/findvar_7k_$I.log	
	# find highest score
		python -u sort_and_find_new_variable.py $M $I -s>./logs_selection_variance/output/sort.$M.$I.out 2>./logs_selection_variance/error/sort.$M.$I.err	

	
	# transfer efps to pascal for training	
		M=$(($M+1))

		python -u create_training_data.py $M $I -s>./logs_selection_variance/output/create_training_data.$M.$I.out 2>./logs_selection_variance/error/create_training_data.$M.$I.err		
	# transfer efps to amarel for training variance		
	
		scp -r ./temp/features/*$I* rd804@pascal:~/ypred_method/temp/features/	
		scp -r ./temp/features/*_${M}_*$I* rd804@amarel.rutgers.edu:/scratch/rd804/training_variance_checks/temp/features/
#		ssh rd804@amarel.rutgers.edu "bash -s" <./remote_commands_scratch.sh $M $I

	# obtain classifer out	
		ssh rd804@pascal /opt/anaconda3/bin/python -u ./ypred_method/classifier_training.py $M $I -s>./logs_selection_variance/output/classifier_training.$M.$I.out 2>./logs_selection_variance/error/classifier_training.$M.$I.err
	# transfer classifier output set
		scp -r rd804@pascal:~/ypred_method/temp/ypred_batch ./temp/		
	
#####################
		
	#	scp -r ./temp/efp/efp*$I* rd804@pascal:~/ypred_method/temp/efp/	
	#	scp -r ./temp/efp/efp*_${M}_*$I* rd804@amarel.rutgers.edu:/scratch/rd804/training_variance_checks/temp/efp/
	#	ssh rd804@amarel.rutgers.edu "bash -s" <./remote_commands_scratch.sh $M $I
	
	# obtain confusion set	
	#	ssh rd804@pascal /opt/anaconda3/bin/python -u ./ypred_method/zooming.py $M $I -s>./logs_selection_variance/output/zooming.$M.$I.out 2>./logs_selection_variance/error/zooming.$M.$I.err
	# transfer confusion set
	#	scp -r rd804@pascal:~/ypred_method/temp/ypred_batch ./temp/		

	echo "done with feature $M"
	done
	M=0	 
echo "Done with iteration $I !!!"

I=${end}

done

echo "Done with all iterations !!"
