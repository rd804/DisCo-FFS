source ~/.bashrc

cd /scratch/rd804/DisCo-FFS

iter=$1
exp_name=$2
data=$3

echo ${iter}
echo ${exp_name}


echo "$(pwd)"
mv results/${exp_name}/features/train.npy results/${exp_name}/features/train_${iter}.npy
mv results/${exp_name}/features/test.npy results/${exp_name}/features/test_${iter}.npy
mv results/${exp_name}/features/val.npy results/${exp_name}/features/val_${iter}.npy



nohup bash job_scripts/training_variance.sh ${iter} ${exp_name} ${data} &>./logs/logs/training_variance_${iter}_${exp_name}.out &
