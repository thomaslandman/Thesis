import os

from utils.config import process_config_gen
from utils.generate_jobs import submit_job

setting = dict()
setting['cluster_manager'] = 'Slurm'
setting['NumberOfGPU'] = 1
setting['cluster_MemPerCPU'] = 7500
setting['cluster_NumberOfCPU'] = 4             # Number of CPU per job
setting['cluster_NodeList'] = 'res-hpc-lkeb03'  # ['res-hpc-gpu01','res-hpc-gpu02','res-hpc-lkeb03',---,'res-hpc-lkeb07']


if 'lkeb' in setting['cluster_NodeList']:
    setting['cluster_queue'] = 'LKEBgpu'
    setting['cluster_Partition'] = 'LKEBgpu'
elif 'gpu' in setting['cluster_NodeList']:
    setting['cluster_queue'] = 'gpu'
    setting['cluster_Partition'] = 'gpu'

experiments_dict = {}
experiments_dict['segmentation_a'] ={'model_name':'Seg', 'task':'Single-Task', 'agent':'stlAgent', 'network':'Seg',
                                     'input':'If', "task_ids": ['seg'], 'num_featurmaps': [23, 45, 91], 'num_classes':5}

experiments_dict['segmentation_b'] ={'model_name':'Seg', 'task':'Single-Task', 'agent':'stlAgent', 'network':'Seg',
                                     'input':'If_Sm', "task_ids": ['seg'], 'num_featurmaps': [23, 45, 91], 'num_classes':5}


experiments_dict['segmentation_c'] ={'model_name':'Seg', 'task':'Single-Task', 'agent':'stlAgent', 'network':'Seg',
                                     'input':'If_Im_Sm', "task_ids": ['seg'], 'num_featurmaps': [23, 45, 91], 'num_classes':5}


experiments_dict['registration_a'] ={'model_name':'Reg', 'task':'Single-Task', 'agent':'stlAgent', 'network':'Reg',
                                     'input':'If_Im', 'task_ids': ['reg'], 'num_featurmaps': [23, 45, 91], 'num_classes':3}

experiments_dict['registration_b'] ={'model_name':'Reg', 'task':'Single-Task', 'agent':'stlAgent', 'network':'Reg',
                                     'input':'If_Im_Sm', 'task_ids': ['reg'], 'num_featurmaps': [23, 45, 91], 'num_classes':3}

experiments_dict['doseprediction_a'] ={'model_name':'Dose', 'task':'Single-Task', 'agent':'stlAgent', 'network':'Dose',
                                     'input':'Sf', 'task_ids': ['dose'], 'num_featurmaps': [23, 45, 91], 'num_classes':3}

experiments_dict['doseprediction_b'] ={'model_name':'Dose', 'task':'Single-Task', 'agent':'stlAgent', 'network':'Dose',
                                     'input':'Sf_If', 'task_ids': ['dose'], 'num_featurmaps': [23, 45, 91], 'num_classes':3}

experiments_dict['doseprediction_c'] ={'model_name':'Dose_Samp', 'task':'Single-Task', 'agent':'stlAgent', 'network':'Dose',
                                     'input':'Sf_If', 'task_ids': ['dose'], 'num_featurmaps': [23, 45, 91], 'num_classes':3}

experiments_dict['doseprediction_d'] ={'model_name':'Dose_Samp', 'task':'Single-Task', 'agent':'stlAgent', 'network':'Dose',
                                     'input':'Sf_If_Dm', 'task_ids': ['dose'], 'num_featurmaps': [23, 45, 91], 'num_classes':3}

experiments_dict['doseprediction_e'] ={'model_name':'Dose_Samp', 'task':'Single-Task', 'agent':'stlAgent', 'network':'Dose',
                                     'input':'Dm', 'task_ids': ['dose'], 'num_featurmaps': [23, 45, 91], 'num_classes':3}

experiments_dict['doseprediction_f'] ={'model_name':'Dose', 'task':'Single-Task', 'agent':'stlAgent', 'network':'Dose',
                                     'input':'Dm_DVF', 'task_ids': ['dose'], 'num_featurmaps': [23, 45, 91], 'num_classes':3}

experiments_dict['doseprediction_g'] ={'model_name':'Dose_Deep', 'task':'Single-Task', 'agent':'stlAgent', 'network':'Dose',
                                     'input':'Sf_If_Dm', 'task_ids': ['dose'], 'num_featurmaps': [16, 32, 64, 128], 'num_classes':4}

experiments_dict['doseprediction_h'] ={'model_name':'Dose_Masks', 'task':'Single-Task', 'agent':'stlAgent', 'network':'Dose',
                                     'input':'Sf_If_Dm', 'task_ids': ['dose'], 'num_featurmaps': [23, 45, 91], 'num_classes':4}

experiments_dict['doseprediction_i'] ={'model_name':'Dose_Masks', 'task':'Single-Task', 'agent':'stlAgent', 'network':'Dose',
                                     'input':'Sf_If_Dm_Ma', 'task_ids': ['dose'], 'num_featurmaps': [23, 45, 91], 'num_classes':4}

experiments_dict['doseprediction_j'] ={'model_name':'Dose_Masks', 'task':'Single-Task', 'agent':'stlAgent', 'network':'Dose',
                                     'input':'Sf_If_Dm', 'task_ids': ['dose'], 'num_featurmaps': [23, 45, 91], 'num_classes':4}

experiments_dict['doseprediction_k'] ={'model_name':'Dose_Masks2', 'task':'Single-Task', 'agent':'stlAgent', 'network':'Dose',
                                     'input':'Sf_If_Dm_Ma', 'task_ids': ['dose'], 'num_featurmaps': [23, 45, 91], 'num_classes':4}

experiments_dict['doseprediction_l'] ={'model_name':'Dose_Deep', 'task':'Single-Task', 'agent':'stlAgent', 'network':'Dose',
                                     'input':'Sf_If_Dm_Ma', 'task_ids': ['dose'], 'num_featurmaps': [16, 32, 64, 128], 'num_classes':4}

experiments_dict['doseprediction_m'] ={'model_name':'Dose_Deep_good', 'task':'Single-Task', 'agent':'stlAgent', 'network':'Dose',
                                     'input':'Sf_If_Dm_Ma', 'task_ids': ['dose'], 'num_featurmaps': [16, 32, 64, 128], 'num_classes':4}

experiments_dict['doseprediction_n'] ={'model_name':'Dose_Deep_weights', 'task':'Single-Task', 'agent':'stlAgent', 'network':'Dose',
                                     'input':'Sf_If_Dm_Ma', 'task_ids': ['dose'], 'num_featurmaps': [32, 64, 128, 256], 'num_classes':4}

experiments_dict['doseprediction_p'] ={'model_name':'Dose_Deep_no_weights', 'task':'Single-Task', 'agent':'stlAgent', 'network':'Dose',
                                     'input':'Sf_If_Dm_Ma', 'task_ids': ['dose'], 'num_featurmaps': [32, 64, 128, 256], 'num_classes':4}

experiments_dict['cross-stitch_a']  ={'model_name':'CS', 'task':'Multi-Task', 'agent':'mtlAgent', 'network':'CS',
                                    'weight':'equal', 'input_seg':'If_Sm', 'input_reg':'If_Im_Sm', 'loss_seg':'DSC', 'loss_reg':'NCC_DSCWarp',
                                    'input':'If_Im_Sm', "task_ids": ["seg", "reg", "seg_reg"], 'num_featurmaps': [16, 32, 64], 'num_classes':5}

experiments_dict['w-net_a']         ={'model_name':'w-net', 'task':'Single-Task', 'agent':'stlAgent_2', 'network':'w-net',
                                    'input':'Sm_If_Dm', 'task_ids': ['w-net'], 'num_featurmaps': None}




exp = experiments_dict['w-net_a']
exp['is_debug'] = False
is_local = False
exp['mode'] = 'train'            #['train', 'inference', 'eval']

base_json_script = '/exports/lkeb-hpc/tlandman/Thesis/MultiTask/configs/base_args.json'
script_address = '/exports/lkeb-hpc/tlandman/Thesis/MultiTask/main.py'
root_log_path = os.path.join('/exports/lkeb-hpc/tlandman/Thesis/MultiTask/experiments', exp['task'])

if exp['task'] == 'Single-Task':
    exp_name = f"{exp['model_name']}_input_{exp['input']}"
elif exp['task'] == 'Multi-Task':
    exp_name = f'{exp["model_name"]}_inSeg_{exp["input_seg"]}_inReg_{exp["input_reg"]}_lSeg_{exp["loss_seg"]}_lReg_{exp["loss_reg"]}_{exp["weight"]}'
if exp['is_debug']:
    exp_name = f'{exp_name}_debug'

config = process_config_gen(base_json_script, exp_name, exp)

json_script = os.path.join(config.log_dir)
if is_local == False:
    submit_job(exp_name, script_address, setting=setting, root_log_path=root_log_path, mode=exp['mode'], json_script=json_script)
if is_local == True:
    mode=exp['mode']
    text = 'source /exports/lkeb-hpc/mseelmahdy/fastMRI-env/bin/activate' '\n'
    text = text + 'python ' + str(script_address) + ' ' + os.path.join(json_script, f'args_{mode}.json')
    os.system(text)