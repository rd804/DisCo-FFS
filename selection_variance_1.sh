source ~/.bashrc

I=4
M=0

#rm -f ./logs/time_stamps.txt
while [ $I -ne 6 ]

do
	while [ $M -ne 21 ]

	do
		TIMESTAMP=`date "+%Y-%m-%d %H:%M:%S"`
		cat >> ./logs_selection_variance/time_stamps_$I.txt <<EOF
$TIMESTAMP : starting with variable $M iteration $I
EOF
		
	# obtain confusion set	
		ssh rd804@pascal /opt/anaconda3/bin/python -u ./ypred_method/zooming.py $M $I>./logs_selection_variance/output/zooming.$M.$I.out 2>./logs_selection_variance/error/zooming.$M.$I.err
	# transfer confusion set
		scp -r rd804@pascal:~/ypred_method/temp/ypred_batch ./temp/		
		mkdir -p ./temp/discor_$I
		mkdir -p ./temp/discor_$I/iteration_$M
	# calculate score
		condor_submit M=$M I=$I find_next_variable.jdl.base
	#wait for condor to finish job
		condor_wait /het/p1/ranit/tops/training_method_ypred_cut_04_06_changing_threshold/logs_selection_variance/logs/findvar_7k_$I.log
		rm ./logs_selection_variance/logs/findvar_7k_$I.log	
	# find highest score
		python -u sort_and_find_new_variable.py $M $I>./logs_selection_variance/output/sort.$M.$I.out 2>./logs_selection_variance/error/sort.$M.$I.err	

	
	# transfer efps to pascal for training	
		scp -r ./temp/efp/efp* rd804@pascal:~/ypred_method/temp/efp/	
		M=$(($M+1))	
	echo "done with feature $M"
	done
	M=0	 
echo "Done with iteration $I !!!"

I=$(($I+1))	

done

echo "Done with all iterations !!"
