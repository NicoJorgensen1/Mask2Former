# Import the libraries and functions used here
import os
import torch 
import sys 
import numpy as np
from copy import deepcopy 
from tqdm import tqdm 
from detectron2.engine.defaults import DefaultPredictor
from mask2former import MaskFormerInstanceDatasetMapper
from mask2former import InstanceSegEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.data import MetadataCatalog


# Define a function to change a certain value in a numpy array into another value
def changeValue(inp, origVal=-1, newVal=0):                                                                     # The function will take a numpy array as input and an original value that must be changed
    inp[inp==origVal] = newVal                                                                                  # Change all indices of the original value to the new value 
    return inp                                                                                                  # Return the modified numpy array 


# data_split="train"
# dataloader=None
# evaluator=None
# hp_optim=False

# Build evaluator to compute the evaluation metrics 
def evaluateResults(FLAGS, cfg, data_split="train", dataloader=None, evaluator=None, hp_optim=False):
    # Get the correct properties
    dataset_name = cfg.DATASETS.TRAIN[0] if "train" in data_split.lower() else cfg.DATASETS.TEST[0]             # Get the name of the dataset that will be evaluated
    total_runs = MetadataCatalog.get(dataset_name).num_files_in_dataset                                         # Get the number of files in the chosen dataset
    if "train" in data_split and hp_optim==True: total_runs = 10                                                # If we are performing hyperparameter optimization, only 10 train samples will be evaluated
    if "ade20k" in FLAGS.dataset_name.lower() and hp_optim: total_runs = int(np.ceil(np.divide(total_runs, 4))) # If we are on the ADE20k dataset, then only 1/4 of the dataset will be evaluated during HPO

    pred_out_dir = os.path.join(cfg.OUTPUT_DIR, "Predictions", data_split)                                      # The path of where to store the resulting evaluation
    os.makedirs(pred_out_dir, exist_ok=True)                                                                    # Create the evaluation folder, if it doesn't already exist

    # Build the dataloader, evaluator and model instances
    if dataloader is None:                                                                                      # If no dataloader has been inputted to the function ...
        dataloader = build_detection_test_loader(cfg=cfg, dataset_name=dataset_name,                            # Create the dataloader ...
            batch_size=1, mapper=MaskFormerInstanceDatasetMapper(cfg=cfg, is_train=True, augmentations=[]), num_workers=1)     # ... with the default mapper and no augmentation 
    if evaluator is None:                                                                                       # If there is no evaluator ...
        try: del MetadataCatalog.get(dataset_name).json_file                                                    # ... then the json_file attribute of the MetadataCatalog is removed
        except: pass
        evaluator = InstanceSegEvaluator(dataset_name=dataset_name, output_dir=pred_out_dir, allow_cached_coco=False)   # Build the evaluator instance
    evaluator.reset()                                                                                           # Reset the evaluator metrics 
    evaluator._max_dets_per_image = sorted(np.unique([1, 10, FLAGS.num_queries]).tolist())                      # The maximum allowed number of predictions pr image

    # Create the predictor instance 
    config_2 = deepcopy(cfg)                                                                                    # Create a new copy of the configuration
    config_2.TEST.DETECTIONS_PER_IMAGE = FLAGS.num_queries                                                      # Change the number of maximum detections pr test image 
    predictor = DefaultPredictor(config_2)                                                                      # Create an instance of the default predictor 

    # Create a progress bar to keep track on the evaluation
    with tqdm(total=total_runs, iterable=None, postfix="Evaluating the {:s} dataset".format(data_split), unit="img",  position=0,
            file=sys.stdout, desc="Image {:d}/{:d}".format(1, total_runs), colour="green", leave=True, ascii=True, 
            bar_format="{desc}  | {percentage:3.0f}% | {bar:35}| {n_fmt}/{total_fmt} | [Spent: {elapsed}. Remaining: {remaining} | {postfix}]") as tepoch:
        #Predict all the files in the dataset
        for kk, data_batch in enumerate(dataloader):                                                            # Iterate through all batches in the dataloader
            outputs = list()                                                                                    # Initiate lists to store the predicted arrays and the ground truth tensors
            for data in data_batch:                                                                             # Iterate over all dataset dictionaries in the list
                img = torch.permute(data["image"], (1,2,0)).numpy()                                             # Make input image numpy format and [H,W,C]
                outputs.append(predictor.__call__(img))                                                         # Append the current prediction for the input image to the list of outputs
            evaluator.process(inputs=[data], outputs=outputs)                                                   # Process the results by adding the results to the confusion matrix
            tepoch.desc = "Image {:d}/{:d} ".format(kk+1, total_runs)                                           # Update the description of the progress bar
            tepoch.update(1)                                                                                    # Update progress bar 
            if kk+1 >= total_runs: break                                                                        # If we have performed the total number of runs before emptying the dataloader (e.g. train datasplit during HPO), break the loop 

    # Compute the metrics. NaN values typically happen when an object is not present on any of the evaluated images
    eval_results = evaluator.evaluate()
    for key in eval_results.keys():
        for sub_key in eval_results[key].keys():
            if np.isnan(eval_results[key][sub_key]):
                eval_results[key][sub_key] = float(0)
    
    # Read the COCOeval instance => https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
    # The counts are [Thresholds-IoU, Recall-sample points, K classes, Area ranges, MaxDetections pr img] 
    # T => 10 is from the 10 IoU thresholds => coco_eval._paramsEval.iouThrs = array([0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95])
    # R => 101 is from the 101 sampling points on the Precision-Recall curve => coco_eval._paramsEval.recThrs = [0,0.01, 1]
    # K => 5 is the number of classes => len(coco_eval._paramsEval.catIds) = 5 
    # A => 4 is the area range (area=sum of positive pixels), i.e. the length of the different areas that are searched => coco_eval._paramsEval.areaRngLbl = ['all', 'small', 'medium', 'large'] 
    # M => 3 is the maximum number of detections made pr image, i.e. for when detecting only [1, 10, 100] objects on an image => coco_eval._paramsEval.maxDets = [1, 10, 100]
    # Thus np.mean(precision[0,:,:,0,1], axis=1) gives the 101 R=[0,0.01,1] sampled values on the precision-recall curve for IoU=0.50 for all classes, object areas with 10 detections pr image 
    # We use the changeValue function for precision and recall as the evaluation might contain a value of -1, which means that no GT object was present in that given setting that produced the -1 
    for coco_eval, task_key in zip([evaluator._coco_eval_bbox, evaluator._coco_eval_segm], ["bbox", "segm"]):   # Iterate over both coco eval instances 
        # coco_eval.summarize()                                                                                 # Print out the results for the current task in table form
        eval_results[task_key]["precision"] = changeValue(coco_eval.eval.get("precision"))                      # Precision is [T,R,K,A,M] and thus [10,101,5,4,3] for Vitrolife with K=5 
        eval_results[task_key]["scores"] = coco_eval.eval.get("scores")                                         # Confidence Scores is [T,R,K,A,M] and thus [10,101,5,4,3] 
        eval_results[task_key]["recall"] = changeValue(coco_eval.eval.get("recall"))                            # Recall is [T,K,A,M] and [10,5,4,3]
        eval_results[task_key]["precision_IoU50"] = np.mean(a=eval_results[task_key]["precision"][0,:,:,0,1], axis=1)   # The precision for IoU=0.50, Recall=[0,0.01,1], all classes and object sizes and 10 detections pr image 
        for class_id, class_lbl in zip(MetadataCatalog.get(dataset_name).thing_dataset_id_to_contiguous_id.keys(), MetadataCatalog.get(dataset_name).thing_classes):
            eval_results[task_key]["precision_IoU50_"+class_lbl] = np.mean(a=eval_results[task_key]["precision"][0,:,class_id,:,1], axis=1) # The precision for the given class with IoU@50, all recall values, all object sizes and 10 detections pr image 

    # Return the metrics
    return eval_results, dataloader, evaluator
