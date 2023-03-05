source ~/.bashrc

cd /scratch/rd804/training_variance_checks

M=$1
I=$2
data=$3

echo $M
echo $I

#while [ ! -f /temp/efp_${M}_${I}.npy ]
#do
#       sleep 1 # or less like 0.2
#       echo "sleeping ........ for 1 sec"
#done

echo "$(pwd)"
#squeue 

nohup bash training_variance.sh $M $I ${data} &>./logs_training/logs/training_variance_${M}_${I}.out &

#nohup bash training_variance.sh $M $I &>./logs_training/training_variance_${M}_${I}.out &
