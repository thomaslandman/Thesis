import os
from utils import create_dirs

def job_script(setting, job_name=None, script_address=None, job_output_file=None):
    text = '#!/bin/bash \n'
    text = text + '#SBATCH --job-name=' + job_name.split('_')[0] + '\n'
    text = text + '#SBATCH --output=' + str(job_output_file) + '\n'
    text = text + '#SBATCH --ntasks=1 \n'
    text = text + '#SBATCH --cpus-per-task=' + str(setting['cluster_NumberOfCPU']) + '\n'
    text = text + '#SBATCH --mem-per-cpu=' + str(setting['cluster_MemPerCPU']) + '\n'
    text = text + '#SBATCH --partition=' + setting['cluster_Partition'] + '\n'

    # text = text + '#SBATCH --mem -0' + '\n'
    if setting['cluster_Partition'] in ['gpu', 'LKEBgpu'] and setting['NumberOfGPU']:
        text = text + '#SBATCH --gres=gpu:' + str(setting['NumberOfGPU']) + ' \n'
    text = text + '#SBATCH --time=0 \n'
    if setting['cluster_NodeList'] is not None:
        text = text + '#SBATCH --nodelist='+setting['cluster_NodeList']+' \n'

    text = text + 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/exports/lkeb-hpc/mseelmahdy/cudnn7.4-for-cuda9.0/cuda/lib64/' '\n'
    text = text + 'source /exports/lkeb-hpc/tlandman/tf.14/bin/activate' '\n'

    text = text + 'echo "on Hostname = $(hostname)"' '\n'
    text = text + 'echo "on GPU      = $CUDA_VISIBLE_DEVICES"' '\n'
    text = text + 'echo' '\n'
    text = text + 'echo "@ $(date)"' '\n'
    text = text + 'echo' '\n'

    text = text + 'python ' + str(script_address)
    text = text + '\n'
    return text

setting = dict()
setting['cluster_manager'] = 'Slurm'
setting['NumberOfGPU'] = 1
setting['cluster_MemPerCPU'] = 8500
setting['cluster_NumberOfCPU'] = 4
setting['cluster_NodeList'] = 'res-hpc-lkeb03'  # ['res-hpc-gpu01','res-hpc-gpu02','res-hpc-lkeb03',---,'res-hpc-lkeb07']

if 'lkeb' in setting['cluster_NodeList']:
    setting['cluster_queue'] = 'LKEBgpu'
    setting['cluster_Partition'] = 'LKEBgpu'
elif 'gpu' in setting['cluster_NodeList']:
    setting['cluster_queue'] = 'gpu'
    setting['cluster_Partition'] = 'gpu'

job_name = 'Dose_Prediction_Network'
script_address = '/exports/lkeb-hpc/tlandman/Dose_Prediction/model_exec_new.py'
log_dir_path = '/exports/lkeb-hpc/tlandman/Dose_Prediction/log/'
job_output_file = os.path.join(log_dir_path, 'dose_pred_job.txt')
job_script_address = os.path.join(log_dir_path, 'dose_pred_job.sh')
create_dirs([log_dir_path])

with open(job_script_address, "w") as string_file:

    string_file.write(job_script(setting, job_name=job_name, script_address=script_address,
                                 job_output_file=job_output_file))
    string_file.close()

submit_cmd = 'sbatch ' + str(job_script_address)
os.system(submit_cmd)


