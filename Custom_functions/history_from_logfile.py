import os 
import numpy as np 
import shutil 
import pickle 
import re 
from detectron2.config import get_cfg 
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog

# Save dictionary
def save_dictionary(dictObject, save_folder, dictName):                                                         # Function to save a dict in the specified folder 
    dict_file = open(os.path.join(save_folder, dictName+".pkl"), "wb")                                          # Opens a pickle for saving the dictionary 
    pickle.dump(dictObject, dict_file)                                                                          # Saves the dictionary 
    dict_file.close()                                                                                           # Close the pickle again  

# Find parent directory 
Mask2Former_dir = os.path.join("/mnt", "c", "Users", "Nico-", "Documents", "Python_Projects", "Mask2Former")                                                                # Home WSL
if not os.path.isdir(Mask2Former_dir): Mask2Former_dir = os.path.join("C:\\", Mask2Former_dir.split(os.path.sep, 1)[1])                                                     # Home windows computer
if not os.path.isdir(Mask2Former_dir): Mask2Former_dir = os.path.join("/mnt", "c", "Users", "wd974261", "Documents", "Python", "Mask2Former")                               # Work WSL
if not os.path.isdir(Mask2Former_dir): Mask2Former_dir = os.path.join("C:\\", Mask2Former_dir.split(os.path.sep, 1)[1])                                                     # Work windows computer
if not os.path.isdir(Mask2Former_dir): Mask2Former_dir = os.path.join("/home", "neal", "Panoptic_segmentation_using_deep_neural_networks", "Repositories", "Mask2Former")   # Larac server
if not os.path.isdir(Mask2Former_dir): Mask2Former_dir = os.path.join("/mnt", "home_shared", Mask2Former_dir.split(os.path.sep, 2)[2])                                      # Balder server
assert os.path.isdir(Mask2Former_dir), "The Mask2Former directory doesn't exist in the chosen location"

os.chdir(os.path.join(Mask2Former_dir, "Custom_functions"))                                                     # Switch the current directory to the Custom_functions directory
from custom_display_learning_curves_func import show_history 
os.chdir(Mask2Former_dir)
from mask2former import add_maskformer2_config
os.chdir(os.path.join(Mask2Former_dir, "Custom_functions"))                                                     # Switch the current directory to the Custom_functions directory


for segmentation_type in ["semantic", "instance", "panoptic"]:
    ade20k_logfile_list = [os.path.join(Mask2Former_dir, "ade20k_outputs", x) for x in os.listdir(os.path.join(Mask2Former_dir,  "ade20k_outputs")) if "ade20k_{}_logfile".format(segmentation_type) in x]
    if len(ade20k_logfile_list) != 1:
        print("For the {} segmentation {} logfiles were found!".format(segmentation_type, len(ade20k_logfile_list)))
        continue 
    ade20k_logfile_name = ade20k_logfile_list[0]
    with open(ade20k_logfile_name) as f:
        ade20k_logfile = f.readlines()

    hist_keys = ["total_loss", "loss_ce", "loss_dice", "loss_mask", "lr", "AP", "AP50", "AP75", "PQ", "SQ", "RQ",
                "PQ_st", "RQ_st", "SQ_st", "PQ_th", "SQ_th", "RQ_th", "mIoU", "fwIoU", "mACC", "pACC", "iter", "APs", "APm", "APl"]
    history = {key: list() for key in hist_keys}
    ade20k_inst_log = list()
    for line_numb, line in enumerate(ade20k_logfile):
        if "loss" in line and "INFO" in line and "eta" in line:
            res = line.strip().split(" ")
            for kk, item in enumerate(res):
                if any([x in item.strip() for x in hist_keys]) and not item.strip()[-2].isdigit():
                    key_to_use = np.asarray(hist_keys)[np.asarray([x in item.strip() for x in hist_keys])].item()
                    val_to_use = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", res[kk+1].strip())[0])
                    history[key_to_use].append(val_to_use)
        elif "Task: sem_seg" in line:,
            key_line = ade20k_logfile[line_numb+1].strip().split("copypaste:")[-1].strip().split(",")
            value_line = ade20k_logfile[line_numb+2].strip().split("copypaste:")[-1].strip().split(",")
            for key_name, value in zip(key_line, value_line):
                value = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", value.strip())[0])
                history[key_name.strip()].append(value)
        elif "Task: segm" in line:
            key_line = ade20k_logfile[line_numb+1].strip().split("copypaste:")[-1].strip().split(",")
            value_line = ade20k_logfile[line_numb+2].strip().split("copypaste:")[-1].strip().split(",")
            for key_name, value in zip(key_line, value_line):
                value = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", value.strip())[0])
                history[key_name.strip()].append(value)
        elif "Task: panoptic_seg" in line:
            key_line = ade20k_logfile[line_numb+1].strip().split("copypaste:")[-1].strip().split(",")
            value_line = ade20k_logfile[line_numb+2].strip().split("copypaste:")[-1].strip().split(",")
            for key_name, value in zip(key_line, value_line):
                value = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", value.strip())[0])
                history[key_name.strip()].append(value)


    save_dictionary(dictObject=history, save_folder=os.path.join(Mask2Former_dir, "ade20k_outputs"), dictName="history_ade20k_{}".format(segmentation_type))

    # Create learning curves 
    config = get_cfg()
    add_deeplab_config(config)
    add_maskformer2_config(config)
    config_folder = os.path.join(Mask2Former_dir, "configs", "ade20k", segmentation_type+"-segmentation")
    resnet_config = [x for x in os.listdir(config_folder) if "R50" in x][-1]
    config.merge_from_file(os.path.join(config_folder, resnet_config))
    config.merge_from_file(os.path.join(config_folder, "Base-{}-{}Segmentation.yaml".format("ade20k".upper(), segmentation_type.capitalize())))
    class Namespace(object):
        pass
    FLAGS = Namespace()
    FLAGS.segmentation = segmentation_type.lower().capitalize()
    FLAGS.inference_only=False
    FLAGS.hp_optim=False
    metadata = MetadataCatalog.get(config.DATASETS.TRAIN[0])
    FLAGS.num_classes = len(metadata.stuff_dataset_id_to_contiguous_id.keys())
    FLAGS.display_images=True

    show_history(config=config, FLAGS=None, metrics_train=None, metrics_eval=None, history=history)


