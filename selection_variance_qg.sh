source ~/.bashrc

#Quark Gluon tagging
dataset=qg
temp=/het/p1/ranit/qg/disco_ffs/temp/
pascal_dir=/home/rd804/qg/disco_ffs/temp/

#Top tagging
#temp=./temp/
#pascal_dir=~/ypred_method/temp/


I="qg_7k_efps_test_2"
M=10
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
		python -u create_training_data.py -qg $M $I>./logs_selection_variance/output/create_training_data.$M.$I.out 2>./logs_selection_variance/error/create_training_data.$M.$I.err		
		scp -r ${temp}features/*$I* rd804@pascal:${pascal_dir}features/	
		scp -r ${temp}features/*labels* rd804@pascal:${pascal_dir}features/	
		scp -r ${temp}features/*labels* rd804@amarel.rutgers.edu:/scratch/rd804/training_variance_checks/temp/features/
		scp -r ${temp}features/*_${M}_*$I* rd804@amarel.rutgers.edu:/scratch/rd804/training_variance_checks/temp/features/
		ssh rd804@amarel.rutgers.edu "bash -s" <./remote_commands.sh $M $I qg

	# obtain classifer out	
		ssh rd804@pascal /opt/anaconda3/bin/python -u ./ypred_method/classifier_training.py -qg $M $I>./logs_selection_variance/output/classifier_training.$M.$I.out 2>./logs_selection_variance/error/classifier_training.$M.$I.err
	# transfer classifier output set
		scp -r rd804@pascal:${pascal_dir}/ypred_batch/*$I* ${temp}/ypred_batch/		
		mkdir -p ${temp}discor_$I
		mkdir -p ${temp}discor_$I/iteration_$M
	# calculate score
		condor_submit M=$M I=$I dataset=${dataset} find_next_variable.jdl.base
	#wait for condor to finish job
		condor_wait /het/p1/ranit/tops/training_method_ypred_cut_04_06_changing_threshold/logs_selection_variance/logs/findvar_7k_$I.log
		rm ./logs_selection_variance/logs/findvar_7k_$I.log
	# find highest score
		python -u sort_and_find_new_variable.py $M $I -qg>./logs_selection_variance/output/sort.$M.$I.out 2>./logs_selection_variance/error/sort.$M.$I.err	

	
	# transfer efps to pascal for training	
		M=$(($M+1))

	# transfer efps to amarel for training variance			
	echo "done with feature $M"
	done
	M=0	 
echo "Done with iteration $I !!!"

I=${end}

done

echo "Done with all iterations !!"
