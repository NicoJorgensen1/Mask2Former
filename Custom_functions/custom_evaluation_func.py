# Import the libraries and functions used here
import sys
import os
import torch
import numpy as np
from copy import deepcopy
from detectron2.data import DatasetCatalog, build_detection_train_loader
from mask2former import MaskFormerInstanceDatasetMapper
from detectron2.engine.defaults import DefaultPredictor
from tqdm import tqdm


data_split="train"
dataloader=None
evaluator=None
hp_optim=False


# Build evaluator to compute the evaluation metrics 
def evaluateResults(FLAGS, cfg, data_split="train", dataloader=None, evaluator=None, hp_optim=False):
    # Get the correct properties
    dataset_name = cfg.DATASETS.TRAIN[0] if "train" in data_split.lower() else cfg.DATASETS.TEST[0]             # Get the name of the dataset that will be evaluated
    total_runs = FLAGS.num_train_files if "train" in data_split.lower() else FLAGS.num_val_files                # Get the number of files 
    if "train" in data_split and hp_optim==True: total_runs = 10                                                # If we are performing hyperparameter optimization, only 10 train samples will be evaluated
    if "ade20k" in FLAGS.dataset_name.lower() and hp_optim: total_runs = int(np.ceil(np.divide(total_runs, 4))) # If we are on the ADE20K dataset, then only 1/4 of the dataset will be evaluated during HPO

    pred_out_dir = os.path.join(cfg.OUTPUT_DIR, "Predictions", data_split)                                      # The path of where to store the resulting evaluation
    os.makedirs(pred_out_dir, exist_ok=True)                                                                    # Create the evaluation folder, if it doesn't already exist

    # Build the dataloader if no dataloader has been sent to the function as an input
    if dataloader is None:                                                                                      # If no dataloader has been inputted to the function ...
        dataloader = iter(build_detection_train_loader(dataset=DatasetCatalog.get(dataset_name),                # ... create the dataloader for evaluation ...
            mapper=MaskFormerInstanceDatasetMapper(cfg, is_train=True, augmentations=[]), total_batch_size=1, num_workers=2))   # ... with batch_size = 1 and no augmentation on the mapper

    # Create the predictor and evaluator instances
    predictor = DefaultPredictor(cfg=cfg)


    for kk, data_batch in enumerate(dataloader):
        outputs, gt_mask = list(), list()
        for data in data_batch:
            img = torch.permute(data["image"], (1,2,0)).numpy()
            predictor.__call__(img)
        if kk+1 >= total_runs:
            break 
