import os 
import numpy as np 
import shutil 
import pickle 
import sys 
import re 
from detectron2.config import get_cfg 
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from matplotlib import pyplot as plt 

# Save dictionary
def save_dictionary(dictObject, save_folder, dictName):                                                         # Function to save a dict in the specified folder 
    dict_file = open(os.path.join(save_folder, dictName+".pkl"), "wb")                                          # Opens a pickle for saving the dictionary 
    pickle.dump(dictObject, dict_file)                                                                          # Saves the dictionary 
    dict_file.close()                                                                                           # Close the pickle again  

# Find parent directory 
Mask2Former_dir = os.path.join("/mnt", "c", "Users", "Nico-", "Documents", "Python_Projects", "Mask2Former")                                                                # Home WSL
if not os.path.isdir(Mask2Former_dir):
    Mask2Former_dir = os.path.join("C:\\", Mask2Former_dir.split(os.path.sep, 1)[1])                                                     # Home windows computer
if not os.path.isdir(Mask2Former_dir):
    Mask2Former_dir = os.path.join("/mnt", "c", "Users", "wd974261", "Documents", "Python", "Mask2Former")                               # Work WSL
if not os.path.isdir(Mask2Former_dir):
    Mask2Former_dir = os.path.join("C:\\", Mask2Former_dir.split(os.path.sep, 1)[1])                                                     # Work windows computer
if not os.path.isdir(Mask2Former_dir):
    Mask2Former_dir = os.path.join("/home", "neal", "Panoptic_segmentation_using_deep_neural_networks", "Repositories", "Mask2Former")   # Larac server
if not os.path.isdir(Mask2Former_dir):
    Mask2Former_dir = os.path.join("/mnt", "home_shared", Mask2Former_dir.split(os.path.sep, 2)[2])                                      # Balder server
assert os.path.isdir(Mask2Former_dir), "The Mask2Former directory doesn't exist in the chosen location"
sys.path.append(Mask2Former_dir)                                                                                # Add Mask2Former directory to PATH
sys.path.append(os.path.join(Mask2Former_dir, "Custom_functions"))                                              # Add Custom_functions directory to PATH
os.chdir(os.path.join(Mask2Former_dir, "Custom_functions"))                                             # Switch the current directory to the Custom_functions directory
sys.path.append(os.path.join(Mask2Former_dir, "tools"))
ade20k_output_folder = os.path.join(Mask2Former_dir, "ade20k_outputs")
sys.path.append(ade20k_output_folder)

from mask2former import add_maskformer2_config
from custom_display_learning_curves_func import show_history, mov_avg_array

for segmentation_type in ["semantic", "instance", "panoptic"]:
    ade20k_logfile_list = [os.path.join(ade20k_output_folder, x) for x in os.listdir(ade20k_output_folder) if "ade20k_{}_logfile".format(segmentation_type) in x]
    if len(ade20k_logfile_list) != 1:
        print("For the {} segmentation {} logfiles were found!".format(segmentation_type, len(ade20k_logfile_list)))
        continue 
    ade20k_logfile_name = ade20k_logfile_list[0]
    with open(ade20k_logfile_name) as f:
        ade20k_logfile = f.readlines()

    hist_keys = ["total_loss", "loss_ce", "loss_dice", "loss_mask", "lr", "AP", "AP50", "AP75", "PQ", "SQ", "RQ",
                "PQ_st", "RQ_st", "SQ_st", "PQ_th", "SQ_th", "RQ_th", "mIoU", "fwIoU", "mACC", "pACC", "iter", "APs", "APm", "APl"]
    history = {"val_"+key if not any([x.lower() in key.lower() for x in ["loss", "lr", "iter"]]) else key: list() for key in hist_keys}
    ade20k_inst_log = list()
    for line_numb, line in enumerate(ade20k_logfile):
        if "loss" in line and "INFO" in line and "eta" in line:
            res = line.strip().split(" ")
            for kk, item in enumerate(res):
                if any([x in item.strip() for x in hist_keys]) and not item.strip()[-2].isdigit():
                    key_to_use = np.asarray(hist_keys)[np.asarray([x in item.strip() for x in hist_keys])].item()
                    val_to_use = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", res[kk+1].strip())[0])
                    if key_to_use == "lr":
                        val_to_use = float(res[kk+1])
                    history[key_to_use].append(val_to_use)
                
        elif "Task: sem_seg" in line:
            key_line = ade20k_logfile[line_numb+1].strip().split("copypaste:")[-1].strip().split(",")
            value_line = ade20k_logfile[line_numb+2].strip().split("copypaste:")[-1].strip().split(",")
            for key_name, value in zip(key_line, value_line):
                value = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", value.strip())[0])
                key_name = "val_"+key_name if not any([x.lower() in key_name.lower() for x in ["loss", "lr", "iter"]]) else key_name
                history[key_name.strip()].append(value)
        elif "Task: segm" in line:
            key_line = ade20k_logfile[line_numb+1].strip().split("copypaste:")[-1].strip().split(",")
            value_line = ade20k_logfile[line_numb+2].strip().split("copypaste:")[-1].strip().split(",")
            for key_name, value in zip(key_line, value_line):
                value = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", value.strip())[0])
                key_name = "val_"+key_name if not any([x.lower() in key_name.lower() for x in ["loss", "lr", "iter"]]) else key_name
                history[key_name.strip()].append(value)
        elif "Task: panoptic_seg" in line:
            key_line = ade20k_logfile[line_numb+1].strip().split("copypaste:")[-1].strip().split(",")
            value_line = ade20k_logfile[line_numb+2].strip().split("copypaste:")[-1].strip().split(",")
            for key_name, value in zip(key_line, value_line):
                value = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", value.strip())[0])
                key_name = "val_"+key_name if not any([x.lower() in key_name.lower() for x in ["loss", "lr", "iter"]]) else key_name
                history[key_name.strip()].append(value)

    if "semantic" in segmentation_type.lower():
        key_used = "mIoU"
    if "instance" in segmentation_type.lower():
        key_used = "AP"
    if "panoptic" in segmentation_type.lower():
        key_used = "PQ"
    num_epochs = len(history["val_"+key_used])
    history["train_epoch_num"] = np.arange(0, num_epochs).tolist()
    history["val_epoch_num"] = np.arange(0, num_epochs).tolist()
    save_dictionary(dictObject=history, save_folder=ade20k_output_folder, dictName="history_ade20k_{}".format(segmentation_type))


    for key in history.keys():                                                                          # Loop through each key in the dict[key]->value list
        if "loss" not in key.lower():
            continue 
        key_val, mov_avg_val = list(), list()                                                           # Initiate lists to store the actual values and the moving-average computed values
        for item in history[key]:
            key_val.append(item)                                                                        # Append the actual key value to the key_val list
            mov_avg_val.append(mov_avg_array(inp_array=key_val, mov_of_last_n_elements=7, output_last_n_elements=1).item())   # Compute the next mov_avg val for the last 50 elements
        history[key] = mov_avg_val

    # Create learning curves 
    config = get_cfg()
    add_deeplab_config(config)
    add_maskformer2_config(config)
    config_folder = os.path.join(Mask2Former_dir, "configs", "ade20k", segmentation_type+"-segmentation")
    resnet_config = [x for x in os.listdir(config_folder) if "R50" in x][-1]
    config.merge_from_file(os.path.join(config_folder, resnet_config))
    config.merge_from_file(os.path.join(config_folder, "Base-{}-{}Segmentation.yaml".format("ade20k".upper(), segmentation_type.capitalize())))
    config.OUTPUT_DIR = ade20k_output_folder
    class Namespace(object):
        pass
    FLAGS = Namespace()
    FLAGS.segmentation = [segmentation_type.lower().capitalize()]
    FLAGS.inference_only=False
    FLAGS.hp_optim=False
    metadata = MetadataCatalog.get(config.DATASETS.TRAIN[0])
    if "instance" in segmentation_type.lower():
        FLAGS.num_classes = len(metadata.thing_classes)    
    else:
        FLAGS.num_classes = len(metadata.stuff_classes)
    FLAGS.display_images=True

    _ = show_history(config=config, FLAGS=FLAGS, metrics_train=None, metrics_eval=None, history=history)


