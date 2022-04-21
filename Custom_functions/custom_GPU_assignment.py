import os
import subprocess
import numpy as np

def assign_free_gpus(max_gpus=1):
    # Get info about each of the available GPUs
    smi_query_result = subprocess.check_output('nvidia-smi -q -d Memory | grep -A4 GPU', shell=True)
    # Extract the usage information
    gpu_info = smi_query_result.decode('utf-8').split('\n')
    total_mem_info = list(filter(lambda info: 'Total' in info, gpu_info))
    total_mem_info = [int(x.split(':')[1].replace('MiB', '').strip()) for x in total_mem_info]
    used_mem_info = list(filter(lambda info: 'Used' in info, gpu_info))
    used_mem_info = [int(x.split(':')[1].replace('MiB', '').strip()) for x in used_mem_info]
    available_mem_info = np.subtract(total_mem_info, used_mem_info)
    too_few_gpus_str = None
    if max_gpus > len(available_mem_info):
        too_few_gpus_str = "Only {:.0f} GPU's available. Lowering max_gpus from {:.0f} to {:.0f}".format(len(available_mem_info), max_gpus, len(available_mem_info))
        max_gpus = len(available_mem_info)
    gpus_to_use_int = np.sort(np.flip(np.argsort(available_mem_info))[:max_gpus])
    gpus_to_use_string = ",".join([str(x) for x in gpus_to_use_int])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use_string
    available_mem_string_list = ["{:d}MB".format(x) for x in available_mem_info[gpus_to_use_int]]
    if len(available_mem_string_list) == 1: available_mem_string_list = available_mem_string_list[-1]
    if gpus_to_use_string: gpus_to_use_string = "Using GPU{:s}: {} with {} available memory".format("s" if len(gpus_to_use_string)>1 else "", gpus_to_use_string, available_mem_string_list)
    else: gpus_to_use_string = "No available GPU found"
    return too_few_gpus_str, gpus_to_use_string, available_mem_info[gpus_to_use_int]
