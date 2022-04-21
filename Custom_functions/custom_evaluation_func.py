# Import the libraries and functions used here
import sys
import os
import torch
import numpy as np
from copy import deepcopy
from detectron2.data import DatasetCatalog, DatasetMapper, build_detection_train_loader
from detectron2.evaluation import DatasetEvaluator
from mask2former import InstanceSegEvaluator
from detectron2.engine.defaults import DefaultPredictor
from tqdm import tqdm

# Build evaluator to compute both losses and metrics 

def evalResults(FLAGS, cfg, data_split="train", dataloader=None, evaluator=None, hp_optim=False):
    # Get the correct properties
    dataset_name = cfg.DATASETS.TRAIN[0] if "train" in data_split.lower() else cfg.DATASETS.TEST[0]         # Get the name of the dataset that will be evaluated
    total_runs = FLAGS.num_train_files if "train" in data_split.lower() else FLAGS.num_val_files            # Get the number of files 
    if "train" in data_split and hp_optim==True: total_runs = 10                                            # If we are performing hyperparameter optimization, only 10 train samples will be evaluated
    if "ade20k" in FLAGS.dataset_name.lower() and hp_optim: total_runs = int(np.ceil(np.divide(total_runs, 4))) # If we are on the ADE20K dataset, then only 1/4 of the dataset will be evaluated during HPO

    pred_out_dir = os.path.join(cfg.OUTPUT_DIR, "Predictions", data_split)                                  # The path of where to store the resulting evaluation
    os.makedirs(pred_out_dir, exist_ok=True)                                                                # Create the evaluation folder, if it doesn't already exist






# Create a custom 'process' function, where the mask GT image, that is send with the input dictionary is actually used...
class Instance_Evaluator(InstanceSegEvaluator):
    def compute_val_loss():
        pass 