source ~/.bashrc

cd /scratch/rd804/DisCo-FFS

iter=$1
exp_name=$2
data=$3

echo ${iter}
echo ${exp_name}

#while [ ! -f /temp/efp_${M}_${I}.npy ]
#do
#       sleep 1 # or less like 0.2
#       echo "sleeping ........ for 1 sec"
#done

echo "$(pwd)"
#squeue 
mv results/${exp_name}/features/train.npy results/${exp_name}/features/train_${iter}.npy
mv results/${exp_name}/features/test.npy results/${exp_name}/features/test_${iter}.npy
mv results/${exp_name}/features/val.npy results/${exp_name}/features/val_${iter}.npy



nohup bash training_variance.sh ${iter} ${exp_name} ${data} &>./logs/logs/training_variance_${iter}_${exp_name}.out &

#nohup bash training_variance.sh $M $I &>./logs_training/training_variance_${M}_${I}.out &
