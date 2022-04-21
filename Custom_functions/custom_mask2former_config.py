import os                                                                                           # Used to navigate around the current os and folder structure
import torch                                                                                        # torch is implemented to check if a GPU is available
import numpy as np                                                                                  # Used to compute the step ranges
from copy import deepcopy 
from sys import path as sys_PATH                                                                    # Import the PATH variable
from detectron2.data import MetadataCatalog                                                         # Catalog containing metadata for all datasets available in Detectron2
from detectron2.config import get_cfg                                                               # Function to get the default configuration from Detectron2
from detectron2.projects.deeplab import add_deeplab_config                                          # Used to merge the default config with the deeplab config before training
from mask2former import add_maskformer2_config                                                      # Used to add the new configuration to the list of possible configurations


# Locate the folder containing other configurations
Mask2Former_dir = [x for x in sys_PATH if x.endswith("Mask2Former")][0]                             # Get the path of the Mask2Former directory
config_folder = os.path.join(Mask2Former_dir, "configs", "ade20k", "instance-segmentation")         # Get the path of the configs


# Define function to get all keys in a nested dictionary
def accumulate_keys(dct):
    key_list = list()
    def accumulate_keys_recursive(dct):
        for key in dct.keys():
            if isinstance(dct[key], dict): accumulate_keys_recursive(dct[key])
            else: key_list.append(key.upper())
    accumulate_keys_recursive(dct)
    return key_list


# Define a function to create a custom configuration in the chosen config_dir and takes a namespace option
def createVitrolifeConfiguration(FLAGS):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    if FLAGS.use_transformer_backbone==True:                                                        # If the user chose the transformer backbone ...
        swin_type = "tiny" if "nico" in Mask2Former_dir.lower() else "base"                         # If on home computer, use swin tiny. If on gpucluster, use swin base
        swin_config = [x for x in os.listdir(os.path.join(config_folder, "swin")) if all([swin_type in x, x.endswith(".yaml")])][-1]    # Find the corresponding swin config
        cfg.merge_from_file(os.path.join(config_folder, "swin", swin_config))                       # Merge the configuration with the swin configuration
    else:                                                                                           # If we are not using the swin backbone ...
        cfg.merge_from_file(os.path.join(config_folder, "maskformer2_R50_bs16_160k.yaml"))          # ... instead merge with the ResNet config 
    cfg.merge_from_file(os.path.join(config_folder, "Base-ADE20K-InstanceSegmentation.yaml"))       # Merge with the base config for ade20K dataset. This is the config selecting that we use the ADE20K dataset)

    if "vitrolife" in FLAGS.dataset_name.lower():                                                   # If we are working on the Vitrolife dataset ... 
        cfg["DATASETS"]["TRAIN"] = ("vitrolife_dataset_train",)                                     # ... define the training dataset by using the config as a dictionary
        cfg.DATASETS.TEST = ("vitrolife_dataset_val",)                                              # ... define the validation dataset by using the config as a CfgNode 
        class_labels = deepcopy(MetadataCatalog.get("vitrolife_dataset_test").thing_classes)        # Read all the class labels ...
        class_labels.remove("Background")                                                           # ... remove the background label
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_labels)                                         # ... and assign the length of the remaining list as the number of classes

    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False                                                  # The Sem_Seg head will be off
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True                                                   # The Instance Seg head will be on
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False                                                  # The Panoptic Seg head will be off 
    return cfg


# Define a function to change the configuration using the FLAGS input arguments
def changeConfig_withFLAGS(cfg, FLAGS):
    # The model values     
    cfg.MODEL.RESNETS.DEPTH = FLAGS.resnet_depth                                                    # The depth of the ResNet backbone (if used)
    cfg.MODEL.BACKBONE.FREEZE_AT = FLAGS.backbone_freeze_layers                                     # How many sections of a pretrained backbone that must be freezed
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'                               # Assign the device on which the model should run
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = FLAGS.dice_loss_weight                                      # Set the weight for the dice loss (original 2)
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = FLAGS.mask_loss_weight                                      # Set the weight for the mask predictive loss (original 20)
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = FLAGS.class_loss_weight                                    # Set the weight for the class weight loss 
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = FLAGS.no_object_weight                                 # The loss weight for the "no-object" label
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = float(0.25)                                      # The threshold for overlapping masks. Default to 0.80 
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False                                                  # Disable the panoptic head for the maskformer 
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = FLAGS.num_queries                                    # The number of queries to detect from the Transformer module 
    cfg.MODEL.PANOPTIC_FPN.COMBINE.ENABLED = False                                                  # Always disable the panoptic FPN head 
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512                                                  # The ROI head proposals per image 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50                                                    # Assign the IoU threshold used for the model
    if "vitrolife" in FLAGS.dataset_name.lower(): 
        cfg.MODEL.PIXEL_MEAN = [100.15, 102.03, 103.89]                                             # Write the correct image mean value for the entire vitrolife dataset
        cfg.MODEL.PIXEL_STD = [57.32, 59.69, 61.93]                                                 # Write the correct image standard deviation value for the entire vitrolife dataset
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = cfg.MODEL.ROI_HEADS.NUM_CLASSES                        # Set the number of classes for the sem_seg_head (that is unused)

    # Solver values
    cfg.SOLVER.IMS_PER_BATCH = int(FLAGS.batch_size)                                                # Batch size used when training => batch_size pr GPU = batch_size // num_gpus
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"                                                 # Default learning rate scheduler
    cfg.SOLVER.BASE_LR = FLAGS.learning_rate                                                        # Starting learning rate
    cfg.SOLVER.MAX_ITER = FLAGS.epoch_iter                                                          # This one is automatically computed for each epoch 
    cfg.SOLVER.OPTIMIZER = FLAGS.optimizer_used.upper()                                             # The optimizer to use for training the model
    cfg.SOLVER.NESTEROV = True                                                                      # Whether or not the learning algorithm will use Nesterow momentum
    cfg.SOLVER.WEIGHT_DECAY = FLAGS.weight_decay                                                    # A small lambda value for the weight decay. It is larger when training with transformers
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True                                                        # We need to clip the gradients during training in order to avoid exploding gradients, especially when performing HPO
    cfg.SOLVER.BACKBONE_MULTIPLIER = FLAGS.backbone_multiplier                                      # Backbone learning rate = learning_rate * backbone_multiplier
    cfg.SOLVER.CHECKPOINT_PERIOD = FLAGS.epoch_iter                                                 # Save a new model checkpoint after each epoch, i.e. after everytime the entire trainining set has been seen by the model
    cfg.SOLVER.STEPS = []                                                                           # We won't lower the learning rate in an epoch, use FLAGS.patience instead 
    cfg.SOLVER.GAMMA = 1                                                                            # After every "step" iterations the learning rate will be updated, as new_lr = old_lr*gamma
    
    # Dataloader values 
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = False                                                    # We'll simply shuffle the input data, we won't group them after aspect ratios, even though that would be more GPU efficient
    cfg.DATALOADER.NUM_WORKERS = FLAGS.num_workers                                                  # Set the number of workers to only 2
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False                                                 # Include all images, even the ones without any objects (which is none in the Vitrolife dataset)

    # Input values 
    if "vitrolife" in FLAGS.dataset_name.lower():
        cfg.INPUT.MASK_FORMAT = "bitmask"                                                           # The mask is of the format bitmask as the polygon values come from binary masks
        cfg.INPUT.MIN_SIZE_TRAIN = 500                                                              # The minimum size length for one side of the training images
        cfg.INPUT.MAX_SIZE_TRAIN = 500                                                              # The maximum size length for one side of the training images
        cfg.INPUT.MIN_SIZE_TEST = 500                                                               # The minimum size length for one side of the validation images
        cfg.INPUT.MAX_SIZE_TEST = 500                                                               # The maximum size length for one side of the validation images
    cfg.INPUT.CROP.ENABLED =  FLAGS.crop_enabled                                                    # Whether or not cropping of input images are allowed 
    cfg.INPUT.FORMAT = "BGR"                                                                        # The input format is set to be BGR, like the visualization method
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "random"                                                    # Random sampling of the input images 

    # Output values
    cfg.OUTPUT_DIR = os.path.join(Mask2Former_dir, "output_{:s}{:s}".format("vitrolife_" if "vitro" in FLAGS.dataset_name.lower() else "", FLAGS.output_dir_postfix))   # Get Mask2Former directory and name the output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)                                                      # Create the output folder, if it doesn't already exist

    # Test values 
    cfg.TEST.EVAL_PERIOD = 0                                                                        # We won't use the build in evaluation, only the custom evaluation function
    cfg.TEST.AUG = False                                                                            # No augmentation used for inference

    # Debugging value changes
    if FLAGS.debugging==True:                                                                       # If we are debugging the model ...
        cfg.SOLVER.WEIGHT_DECAY = float(0)                                                          # ... we don't want any weight decay

    # Change the config and add the FLAGS input arguments one by one ... Not pretty, but efficient and doesn't cost memory...
    cfg.custom_key = []
    for key in vars(FLAGS).keys():
        cfg.custom_key.append(tuple((key, vars(FLAGS)[key])))

    return cfg 


