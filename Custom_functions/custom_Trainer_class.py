try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass
import copy
import itertools
import torch
import numpy as np 
from typing import Any, Dict, List, Set
from PIL import Image 
from mask2former import MaskFormerInstanceDatasetMapper, InstanceSegEvaluator
from detectron2.utils import comm
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer, hooks 
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.solver.lr_scheduler import LRMultiplier
from fvcore.common.param_scheduler import CosineParamScheduler
from fvcore.nn.precise_bn import get_bn_modules
from detectron2.engine.hooks import PeriodicWriter


# Define a function that will return a list of augmentations to use for training
def custom_augmentation_mapper(config, is_train=True):
    if "val" in config.DATASETS.TRAIN[0].lower(): transform_list = []       # If we are validating the images, we won't use data augmentation
    else:
        transform_list = [                                                  # Initiate the list of image data augmentations to use
            T.Resize((500,500), Image.BILINEAR),                            # https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html#detectron2.data.transforms.Resize
            T.RandomBrightness(0.8, 1.5),                                   # https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html#detectron2.data.transforms.RandomBrightness
            T.RandomLighting(0.7),                                          # https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html#detectron2.data.transforms.RandomLighting
            T.RandomContrast(0.7, 1.3),                                     # https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html#detectron2.data.transforms.RandomContrast
            T.RandomSaturation(0.85, 1.15),                                 # https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html#detectron2.data.transforms.RandomSaturation
            T.RandomRotation(angle=[-45, 45]),                              # https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html#detectron2.data.transforms.RandomRotation
            T.RandomFlip(prob=0.25, horizontal=True, vertical=False),       # https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html#detectron2.data.transforms.RandomLighting 
            T.RandomFlip(prob=0.25, horizontal=False, vertical=True),       # https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html#detectron2.data.transforms.RandomLighting  
            T.RandomCrop("relative", (0.75, 0.75)),                         # https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html#detectron2.data.transforms.RandomCrop
            T.Resize((500,500), Image.BILINEAR)]                            # https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html#detectron2.data.transforms.Resize
    custom_mapper = MaskFormerInstanceDatasetMapper(config, is_train=is_train, augmentations=transform_list)    # Create the mapping from data dictionary to augmented training image
    # Something has to be done to the bounding boxes before this mapper can be used ... 
    return custom_mapper


# Cosine lr_scheduler
class CosineParamScheduler2(CosineParamScheduler):
    def __init__(self, start_value, end_value, warm_up=False):
        self._start_value = float(start_value)
        self._end_value = float(end_value)
        self.warm_up = warm_up
    
    def __call__(self, where):
        where = float(where)
        new_lr = float(np.add(self._end_value, np.multiply(np.multiply(0.75, np.subtract(self._start_value, self._end_value)), (np.add(1, np.cos(np.multiply(np.pi, where)))))))
        # new_lr = float(self._end_value + 0.5 * (self._start_value - self._end_value) * (1 + np.cos(np.pi * where)))
        return new_lr


# Custom Trainer class build on the DefaultTrainer class. This is mostly copied from the train_net.py
class My_GoTo_Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return None  

    @classmethod
    def build_train_loader(cls, cfg):
        # Instance segmentation dataset mapper
        mapper = MaskFormerInstanceDatasetMapper(cfg, True)
        # mapper = custom_augmentation_mapper(config=cfg)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer, start_val=1, end_val=1):
        for item in cfg.custom_key[::-1]:                                                                       # Iterate over the custom keys in reversed order
            if "epoch_num" in item[0]: current_epoch = item[1]                                                  # If the current item is the tuple with the epoch_number and the current epoch number is noted
            if "learning_rate" in item[0]: wanted_lr = item[1]                                                  # Get the initial learning rate
            if "warm_up_epochs" in item[0]: warm_ups = item[1]                                                  # Get the wanted number of warm up epochs
            if "num_trials" in item[0]: num_trials = item[1]                                                    # Get the total number of HPO trials to run
            if "HPO_current_trial" in item[0]: HPO_current_trial = item[1]                                      # Get the current HPO trial 
            if "hp_optim" in item[0]: do_HPO_bool = item[1]                                                     # Get the variable as whether or not we are doing HPO
        if all([HPO_current_trial < num_trials, do_HPO_bool==True]):                                            # If we are still in the HPO trials ...
            start_val, end_val = 1, 1                                                                           # ... then the learning rate will be constant
        elif warm_ups >= current_epoch:                                                                         # If we are still in the warm up phase ...
            learn_rates = np.linspace(start=np.divide(wanted_lr, 50), stop=wanted_lr, num=warm_ups+1)           # ... we'll create an array of the possible learning rates to choose from
            learn_rates = np.multiply(learn_rates, np.divide(1, wanted_lr))                                     # For some reason necessary ... 
            start_val = learn_rates[current_epoch-1]                                                            # ... and then choose the starting learning rate as the lower one
            end_val = learn_rates[current_epoch]                                                                # ... and then choose the next learning rate as the higher one
        elif warm_ups < current_epoch:                                                                          # Instead if we are in the regular training period ...
            start_val, end_val = float(1), float(0.50)                                                          # Then the learning rate will be cyclical between '--learning_rate' and '0.5 * --learning_rate'
        if "train" not in cfg.DATASETS.TRAIN[0]: start_val, end_val = 0, 0                                      # If we are using the validation or test set, then learning rates are set to 0
        
        sched = CosineParamScheduler2(start_value=start_val, end_value=end_val)
        scheduler = LRMultiplier(optimizer, multiplier=sched, max_iter=cfg.SOLVER.MAX_ITER)
        return scheduler


    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (cfg.SOLVER.CLIP_GRADIENTS.ENABLED and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model" and clip_norm_val > 0.0)

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0                                              # save some memory and time for PreciseBN

        cfg.TEST.PRECISE_BN.ENABLED = True
        dataset_used = cfg.DATASETS.TRAIN[0]                                        # Get the current dataset
        if "val" in dataset_used: cfg.TEST.PRECISE_BN.ENABLED = False               # If we are in validation, precise_bn is disabled
        num_files = MetadataCatalog[dataset_used].num_files_in_dataset              # Read the number of files of the used dataset
        precise_bn_period = int(num_files/2) if num_files < 51 else 25              # Every 25 iteration the PreciseBN will be calculated


        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PreciseBN(
                precise_bn_period,                                                  # Run precise_BN after precise_bn_period iterations
                self.model,                                                         # Assign the current model that must be used for the precise BN
                self.build_train_loader(cfg),                                       # Build a new data loader to not affect training
                precise_bn_period)                                                  # The number of iterations used for computing the precise values
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model) else None]

        # Do PreciseBN before checkpointer, because it updates the model and need to be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency, some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            write_period = int(np.min([10, MetadataCatalog[self.cfg.DATASETS.TRAIN[0]].num_files_in_dataset/np.min([75, MetadataCatalog[self.cfg.DATASETS.TRAIN[0]].num_files_in_dataset])]))
            ret.append(PeriodicWriter(self.build_writers(), period=write_period))
        return ret