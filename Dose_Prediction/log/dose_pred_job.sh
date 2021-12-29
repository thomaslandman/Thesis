#!/bin/bash 
#SBATCH --job-name=Dose
#SBATCH --output=/exports/lkeb-hpc/tlandman/Dose_Prediction/log/dose_pred_job.txt
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8500
#SBATCH --partition=LKEBgpu
#SBATCH --gres=gpu:1 
#SBATCH --time=0 
#SBATCH --nodelist=res-hpc-lkeb03 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/exports/lkeb-hpc/mseelmahdy/cudnn7.4-for-cuda9.0/cuda/lib64/
source /exports/lkeb-hpc/tlandman/tf.14/bin/activate
echo "on Hostname = $(hostname)"
echo "on GPU      = $CUDA_VISIBLE_DEVICES"
echo
echo "@ $(date)"
echo
python /exports/lkeb-hpc/tlandman/Dose_Prediction/model_exec_new.py
