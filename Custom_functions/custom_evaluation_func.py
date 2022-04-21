# Import the libraries and functions used here
import os
import numpy as np
from copy import deepcopy
from detectron2.data import DatasetCatalog, build_detection_train_loader
from mask2former import MaskFormerInstanceDatasetMapper
from detectron2.engine.defaults import DefaultPredictor
from mask2former import InstanceSegEvaluator
from detectron2.evaluation import inference_on_dataset

from detectron2.modeling import build_model
from detectron2.data import build_detection_test_loader


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

    # Build the dataloader if no dataloader, evaluator and model instances
    if dataloader is None:                                                                                      # If no dataloader has been inputted to the function ...
        dataloader = build_detection_test_loader(cfg=cfg, dataset_name=dataset_name,                            # Create the dataloader ...
            batch_size=1, mapper=MaskFormerInstanceDatasetMapper(cfg=cfg, is_train=True, augmentations=[]))     # ... with the default mapper and no augmentation 
    if evaluator is None:
        evaluator = InstanceSegEvaluator(dataset_name=dataset_name, output_dir=pred_out_dir, allow_cached_coco=False)
    model = build_model(cfg=cfg)

    # Compute the metrics 
    eval_results = inference_on_dataset(model=model, data_loader=dataloader, evaluator=evaluator)
    print(eval_results)

    # Return the metrics
    return eval_results, dataloader, evaluator



