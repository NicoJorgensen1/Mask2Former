# Import the libraries and functions used here
import os
import torch 
import sys 
import numpy as np
from copy import deepcopy 
import tqdm 
from detectron2.engine.defaults import DefaultPredictor
from detectron2.evaluation import SemSegEvaluator 
from custom_print_and_log_func import printAndLog
from mask2former import MaskFormerInstanceDatasetMapper
from mask2former import InstanceSegEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.data import MetadataCatalog
from custom_image_batch_visualize_func import NMS_pred, putModelWeights
from custom_panoptic_evaluator import Custom_Panoptic_Evaluator
from panopticapi.utils import id2rgb

# Define a function to change a certain value in a numpy array into another value
def changeValue(inp, origVal=-1, newVal=0):                                                                     # The function will take a numpy array as input and an original value that must be changed
    inp[inp==origVal] = newVal                                                                                  # Change all indices of the original value to the new value 
    return inp                                                                                                  # Return the modified numpy array 


# data_split="test"
# dataloader=None
# cfg=config
# evaluators=None
# hp_optim=False

def evaluateResults(FLAGS, cfg, data_split="train", dataloader=None, evaluators=None, hp_optim=False):
    # Get the correct properties
    dataset_name = cfg.DATASETS.TRAIN[0] if "train" in data_split.lower() else cfg.DATASETS.TEST[0]             # Get the name of the dataset that will be evaluated
    meta_data = MetadataCatalog.get(dataset_name)                                                               # Get the metadata of the current dataset 
    total_runs = meta_data.num_files_in_dataset                                                                 # Get the number of files in the chosen dataset
    if "Panoptic" not in FLAGS.segmentation:
        if "train" in data_split and hp_optim==True:                                                            # If we are performing hyperparameter optimization and evaluating the training set ...
            total_runs = 10                                                                                     # ... only 10 train samples will be evaluated
        if "ade20k" in FLAGS.dataset_name.lower() and hp_optim:                                                 # If we are on the ADE20k dataset ...
            total_runs = int(np.ceil(np.divide(total_runs, 4)))                                                 # ... then only 1/4 of the dataset will be evaluated during HPO

    pred_out_dir = os.path.join(cfg.OUTPUT_DIR, "Predictions")                                                  # The path of where to store the resulting evaluation
    sem_seg_out_dir = os.path.join(pred_out_dir, data_split, "Semantic_segmentation")
    inst_seg_out_dir = os.path.join(pred_out_dir, data_split, "Instance_segmentation")
    panop_seg_out_dir = os.path.join(pred_out_dir, data_split, "Panoptic_segmentation")
    if "Semantic" in FLAGS.segmentation:                                                                        # Create the evaluation folder for semantic segmentation ...
        os.makedirs(sem_seg_out_dir, exist_ok=True)                                                             # ... if it doesn't already exist
    if "Instance" in FLAGS.segmentation:                                                                        # Create the evaluation folder for instance segmentation ...
        os.makedirs(inst_seg_out_dir, exist_ok=True)                                                            # ... if it doesn't already exist
    if "Panoptic" in FLAGS.segmentation:                                                                        # Create the evaluation folder for panoptic segmentation ...
        os.makedirs(panop_seg_out_dir, exist_ok=True)                                                           # ... if it doesn't already exist

    # Build the dataloader, evaluators and model instances
    if dataloader is None:                                                                                      # If no dataloader has been inputted to the function ...
        dataloader = build_detection_test_loader(cfg=cfg, dataset_name=dataset_name,                            # Create the dataloader ...
            batch_size=1, mapper=MaskFormerInstanceDatasetMapper(cfg=cfg, is_train=True, augmentations=[]), num_workers=1)     # ... with the default mapper and no augmentation 
    if evaluators is None:                                                                                      # If there is no evaluators ...
        try: del meta_data.json_file                                                                            # ... then the json_file attribute of the MetadataCatalog is removed
        except: pass
        evaluators = dict()
        if "Semantic" in FLAGS.segmentation:
            semantic_evaluator = deepcopy(SemSegEvaluator(dataset_name=dataset_name, output_dir=sem_seg_out_dir))   # Create an instance of the semantic segmentation evaluator 
            evaluators["Semantic"] = semantic_evaluator
        if "Instance" in FLAGS.segmentation:
            instance_evaluator = deepcopy(InstanceSegEvaluator(dataset_name=dataset_name, output_dir=inst_seg_out_dir, allow_cached_coco=False))    # Build the evaluator for instance segmentation 
            evaluators["Instance"] = instance_evaluator
        if "Panoptic" in FLAGS.segmentation:
            panoptic_evalutator = deepcopy(Custom_Panoptic_Evaluator(dataset_name=dataset_name, output_dir=panop_seg_out_dir))
            evaluators["Panoptic"] = panoptic_evalutator

    # Make the evaluator a dictionary of evaluators 
    for Segment_type in list(evaluators.keys()):
        evaluators[Segment_type].reset()                                                                        # Reset the evaluator 
        evaluators[Segment_type]._max_dets_per_image = sorted(np.unique([1, 10, FLAGS.num_queries]).tolist())   # The maximum allowed number of predictions pr image

    # Create the predictor instance 
    config_2 = deepcopy(cfg)                                                                                    # Create a new copy of the configuration
    config_2 = putModelWeights(config=config_2)                                                                 # Assign the newest model weights to the config used for evaluation 
    config_2.TEST.DETECTIONS_PER_IMAGE = FLAGS.num_queries                                                      # Change the number of maximum detections pr test image 
    predictor = DefaultPredictor(config_2)                                                                      # Create an instance of the default predictor 

    # Create a progress bar to keep track on the evaluation
    PN_pred_count = {x: list() for x in FLAGS.segmentation}
    PN_true_count = deepcopy(PN_pred_count)
    for kk, data_batch in tqdm.tqdm(enumerate(dataloader), total=total_runs, unit="img", position=0, colour="green",  ascii=True,
        postfix="Evaluating the {:s} dataset".format(data_split), file=sys.stdout, desc="Evaluating all {} images".format(total_runs),
        bar_format="{desc}  | {percentage:3.0f}% | {bar:35}| {n_fmt}/{total_fmt} | [Spent: {elapsed}. Remaining: {remaining} | {postfix}]"):    # Iterate through all batches in the dataloader
        for data in data_batch:                                                                                 # Iterate over all dataset dictionaries in the list
            img = torch.permute(data["image"], (1,2,0)).numpy()                                                 # Make input image numpy format and [H,W,C]
            y_pred = predictor.__call__(img)                                                                    # Compute the predicted output
            
            # Process the values using the evaluators 
            for Segment_type in list(evaluators.keys()):
                evaluators[Segment_type].process(inputs=[data], outputs=[y_pred])                               # Process the results by adding the results to the confusion matrix
            
            if "Panoptic" in FLAGS.segmentation:
                if "vitrolife" not in FLAGS.dataset_name.lower():
                    raise(NotImplementedError("Panoptic segmentation only supported for the Vitrolife dataset at the moment"))
                pan_img_pred = id2rgb(y_pred["panoptic_seg"][0].cpu().numpy())[:,:,0]
                unique_values = np.unique(pan_img_pred).tolist() 
                PN_count = 0
                for unique_value in unique_values:
                    if "PN" in meta_data.panoptic_classes[unique_value].upper():
                        PN_count += 1
                PN_pred_count["Panoptic"].append(PN_count)
                PN_true_count["Panoptic"].append(int(data["image_custom_info"]["PN_image"])) 

            # Count PN's for semantic segmentation 
            if all(["Semantic" in FLAGS.segmentation, "vitrolife" in dataset_name.lower(), "test" in data_split.lower()]):
                out_img = torch.nn.functional.softmax(torch.permute(y_pred["sem_seg"], (1,2,0)), dim=-1)        # Get the softmax output of the predicted image
                out_pred_img = torch.argmax(out_img, dim=-1).cpu().numpy()                                      # Convert the predicted image into a numpy mask 
                PN_pred_area = np.sum(out_pred_img == np.where(np.in1d(meta_data.stuff_classes, "PN"))[0].item())   # Count the area of predicted PNs
                PN_pred_count["Semantic"].append(int(np.ceil(np.divide(PN_pred_area, FLAGS.PN_mean_pixel_area))))   # Compute the predicted number of PN's 
                PN_true_count["Semantic"].append(int(data["image_custom_info"]["PN_image"]))                    # Get the true number of PN's 

            # Count PN's for instance segmentation
            if all(["Instance" in FLAGS.segmentation, "vitrolife" in dataset_name.lower(), "test" in data_split.lower()]):
                y_pred = y_pred["instances"].get_fields()
                PN_masks = NMS_pred(y_pred_dict=y_pred, data=data, meta_data=meta_data, conf_thresh=FLAGS.conf_threshold, IoU_thresh=FLAGS.IoU_threshold) 
                PN_pred_count["Instance"].append(int(len(PN_masks)))
                PN_true_count["Instance"].append(int(data["image_custom_info"]["PN_image"]))
            
            tqdm.tqdm.desc = "Image {:d}/{:d} ".format(kk+1, total_runs)                                        # Update the description of the progress bar
        if kk >= total_runs:                                                                                    # If we have performed the total number of runs before emptying the dataloader (e.g. train datasplit during HPO) ...
            break                                                                                               # ... break the loop 

    # Evaluate results for all evaluators 
    eval_results = dict()
    for Segment_type in evaluators.keys():
        eval_seg_type_results = evaluators[Segment_type].evaluate()
        for key in eval_seg_type_results.keys():
            for sub_key in eval_seg_type_results[key].keys():
                if np.isnan(eval_seg_type_results[key][sub_key]):
                    eval_seg_type_results[key][sub_key] = float(0)
        eval_results[Segment_type] = eval_seg_type_results
    
    
    # Read the COCOeval instance => https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
    # The counts are [Thresholds-IoU, Recall-sample points, K classes, Area ranges, MaxDetections pr img] 
    # T => 10 is from the 10 IoU thresholds => coco_eval._paramsEval.iouThrs = array([0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95])
    # R => 101 is from the 101 sampling points on the Precision-Recall curve => coco_eval._paramsEval.recThrs = [0,0.01, 1]
    # K => 5 is the number of classes => len(coco_eval._paramsEval.catIds) = 5 
    # A => 4 is the area range (area=sum of positive pixels), i.e. the length of the different areas that are searched => coco_eval._paramsEval.areaRngLbl = ['all', 'small', 'medium', 'large'] 
    # M => 3 is the maximum number of detections made pr image, i.e. for when detecting only [1, 10, 100] objects on an image => coco_eval._paramsEval.maxDets = [1, 10, 100]
    # Thus np.mean(precision[0,:,:,0,1], axis=1) gives the 101 R=[0,0.01,1] sampled values on the precision-recall curve for IoU=0.50 for all classes, object areas with 10 detections pr image 
    # We use the changeValue function for precision and recall as the evaluation might contain a value of -1, which means that no GT object was present in that given setting that produced the -1 
    if "Instance" in FLAGS.segmentation:
        for coco_eval, task_key in zip([evaluators["Instance"]._coco_eval_bbox, evaluators["Instance"]._coco_eval_segm], ["bbox", "segm"]):   # Iterate over both coco eval instances 
            # coco_eval.summarize()                                                                             # Print out the results for the current task in table form
            eval_results["Instance"][task_key]["precision"] = changeValue(coco_eval.eval.get("precision"))                  # Precision is [T,R,K,A,M] and thus [10,101,5,4,3] for Vitrolife with K=5 
            eval_results["Instance"][task_key]["scores"] = coco_eval.eval.get("scores")                                     # Confidence Scores is [T,R,K,A,M] and thus [10,101,5,4,3] 
            eval_results["Instance"][task_key]["recall"] = changeValue(coco_eval.eval.get("recall"))                        # Recall is [T,K,A,M] and [10,5,4,3]
            eval_results["Instance"][task_key]["precision_IoU50"] = np.mean(a=eval_results["Instance"][task_key]["precision"][0,:,:,0,1], axis=1)   # The precision for IoU=0.50, Recall=[0,0.01,1], all classes and object sizes and 10 detections pr image 
            for class_id, class_lbl in zip(MetadataCatalog.get(dataset_name).thing_dataset_id_to_contiguous_id.keys(), MetadataCatalog.get(dataset_name).thing_classes):
                eval_results["Instance"][task_key]["precision_IoU50_"+class_lbl] = np.mean(a=eval_results["Instance"][task_key]["precision"][0,:,class_id-1,:,1], axis=1) # The precision for the given class with IoU@50, all recall values, all object sizes and 10 detections pr image 

    # Return the metrics
    return eval_results, dataloader, evaluators, PN_pred_count, PN_true_count
