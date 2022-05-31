# Import libraries 
import shutil                                                                                               # Used to copy/rename the metrics.json file after each training/validation step
import torch                                                                                                # Used for emptying GPU cache and hopefully avoid OOM errors 
import os                                                                                                   # For joining paths
import numpy as np                                                                                          # For algebraic equations 
from time import time                                                                                       # Used to time the epoch/training duration
from copy import  deepcopy                                                                                  # Used to create a new copy in memory
from custom_Trainer_class import My_GoTo_Trainer                                                            # To instantiate the Trainer class
from custom_mask2former_setup_func import save_dictionary                                                   # Save history_dict
from custom_print_and_log_func import printAndLog                                                           # Function to log the results 
from custom_mask2former_config import createVitrolifeConfiguration, changeConfig_withFLAGS                  # Create the config used for hyperparameter optimization 
from custom_image_batch_visualize_func import visualize_the_images, putModelWeights                         # Functions visualize the image batch and assigning the latest model checkpoint to the config model.weights 
from custom_display_learning_curves_func import show_history, combineDataToHistoryDictionaryFunc            # Function used to plot the learning curves for the given training and to add results to the history dictionary
from custom_evaluation_func import evaluateResults                                                          # Function to evaluate the metrics for the segmentation
from custom_callback_functions import early_stopping, lr_scheduler, keepAllButLatestAndBestModel, updateLogsFunc    # Callback functions for model training


# Run the training function 
def run_train_func(cfg):
    Trainer = My_GoTo_Trainer(cfg)
    Trainer.resume_or_load(resume=False)
    Trainer.train()
    return


# Function to launch the training
def launch_custom_training(FLAGS, config, dataset, epoch=0, run_mode="train", hyperparameter_opt=False, quit_training=False):
    FLAGS.epoch_iter = int(np.floor(np.divide(FLAGS.num_train_files, FLAGS.batch_size)))                    # Compute the number of iterations per training epoch with the given batch size
    config.SOLVER.MAX_ITER = FLAGS.epoch_iter * (5 if all(["train" in run_mode, hyperparameter_opt==False, "vitrolife" in FLAGS.dataset_name.lower()]) else 1)  # Increase training iteration count for precise BN computations
    if all(["train" in run_mode, hyperparameter_opt==True]):
        if "vitrolife" in FLAGS.dataset_name.lower(): config.SOLVER.MAX_ITER = int(FLAGS.epoch_iter * 1)    # ... Transformer and ResNet backbones need a ...
        elif "ade20k" in FLAGS.dataset_name.lower(): config.SOLVER.MAX_ITER = int(FLAGS.epoch_iter * 1/15)  # ... few thousand samples to accomplish anything
    if "val" in run_mode and "ade20k" in FLAGS.dataset_name.lower(): config.SOLVER.MAX_ITER = int(np.ceil(np.divide(FLAGS.epoch_iter, 4)))
    if "nico" in config.OUTPUT_DIR.lower():                                                                 # If I am working on my own local computer ...
        config.SOLVER.MAX_ITER = int(np.min([FLAGS.epoch_iter, 15]))                                        # ... the maximum number of iterations is lowered 
    config.SOLVER.CHECKPOINT_PERIOD = config.SOLVER.MAX_ITER                                                # Save a new model checkpoint after each epoch
    if "train" in run_mode and hyperparameter_opt==False:                                                   # If we are training ... 
        for idx, item in enumerate(config.custom_key[::-1]):                                                # Iterate over the custom keys in reversed order
            if "epoch_num" in item[0]:                                                                      # If the current item is the tuple with the epoch_number
                config.custom_key[-idx-1] = (item[0], item[1]+1)                                            # The current epoch number is updated 
                break                                                                                       # And the loop is broken out of 
    config = putModelWeights(config)                                                                        # Assign the latest saved model to the config
    if "val" in run_mode.lower(): config.SOLVER.BASE_LR = float(0)                                          # If we are on the validation split set the learning rate to 0
    else: config.SOLVER.BASE_LR = FLAGS.learning_rate                                                       # Else, we are on the training split, so assign the latest saved learning rate to the config
    config.DATASETS.TRAIN = dataset                                                                         # Change the config dataset used to the dataset sent along ...
    run_train_func(cfg=config)                                                                              # Run the training for the current epoch
    shutil.copyfile(os.path.join(config.OUTPUT_DIR, "metrics.json"),                                        # Rename the metrics.json to "run_mode"_metricsX.json ...
        os.path.join(config.OUTPUT_DIR, run_mode+"_metrics_{:d}.json".format(epoch+1)))                     # ... where X is the current epoch number
    os.remove(os.path.join(config.OUTPUT_DIR, "metrics.json"))                                              # Remove the original metrics file
    shutil.copyfile(os.path.join(config.OUTPUT_DIR, "model_final.pth"),                                     # Rename the metrics.json to "run_mode"_metricsX.json ...
        os.path.join(config.OUTPUT_DIR, "model_epoch_{:d}.pth".format(epoch+1)))                            # ... where X is the current epoch number    
    [os.remove(os.path.join(config.OUTPUT_DIR, x)) for x in os.listdir(config.OUTPUT_DIR) if all(["model_" in x, "epoch" not in x, x.endswith(".pth")])]    # Remove all irrelevant models
    return config, quit_training


# Define a function to create the hyper parameters of the trials
def get_HPO_params(config, FLAGS, trial, hpt_opt=False):
    # If we are performing hyperparameter optimization, the config should be updated
    if all([hpt_opt==True, trial is not None, FLAGS.hp_optim==True]):
        # Set limits on the possible HPO values 
        batch_size_max = 1
        if FLAGS.use_transformer_backbone==True:
            batch_size_max = np.max([1, int(np.floor(np.min(FLAGS.available_mem_info)/3000))]) * FLAGS.num_gpus
        else:
            batch_size_max = int(np.ceil(np.min(FLAGS.available_mem_info)/2500))
        
        # Change the FLAGS parameters and then change the config
        FLAGS.learning_rate = trial.suggest_float(name="learning_rate", low=1e-8, high=5e-5)
        FLAGS.batch_size = trial.suggest_int(name="batch_size", low=1, high=int(batch_size_max))
        FLAGS.optimizer_used = trial.suggest_categorical(name="optimizer_used", choices=["ADAMW", "SGD"])
        FLAGS.weight_decay = trial.suggest_float(name="weight_decay", low=1e-8, high=2e-2)
        FLAGS.backbone_multiplier = trial.suggest_float("backbone_multiplier", low=1e-8, high=0.15) 
        FLAGS.dice_loss_weight = trial.suggest_float(name="dice_loss_weight", low=1, high=25)
        FLAGS.mask_loss_weight = trial.suggest_float(name="mask_loss_weight", low=1, high=25)
        FLAGS.class_loss_weight = trial.suggest_float(name="class_loss_weight", low=1, high=25) 
        FLAGS.no_object_weight = trial.suggest_float(name="no_object_weight", low=1e-4, high=2)
        if "vitrolife" in FLAGS.dataset_name:
            FLAGS.num_queries = trial.suggest_int(name="num_queries", low=15, high=150) 
            FLAGS.dropout = trial.suggest_float(name="dropout", low=1e-11, high=0.25)
        if FLAGS.use_transformer_backbone==False and "Instance" not in FLAGS.segmentation:
            FLAGS.resnet_depth = trial.suggest_categorical(name="resnet_depth", choices=[50, 101])
            FLAGS.backbone_freeze_layers = trial.suggest_int(name="backbone_freeze", low=0, high=5)
        del config 
        config = createVitrolifeConfiguration(FLAGS=FLAGS)
        config = changeConfig_withFLAGS(cfg=config, FLAGS=FLAGS)
    elif all([hpt_opt==False, trial is not None, FLAGS.hp_optim==True]):
        # Let the FLAGS parameters take the values of the best found parameters 
        FLAGS.learning_rate = trial.params["learning_rate"]
        FLAGS.batch_size = trial.params["batch_size"]
        FLAGS.optimizer_used = trial.params["optimizer_used"]
        FLAGS.weight_decay = trial.params["weight_decay"]
        FLAGS.backbone_multiplier = trial.params["backbone_multiplier"] 
        FLAGS.dice_loss_weight = trial.params["dice_loss_weight"]
        FLAGS.mask_loss_weight = trial.params["mask_loss_weight"]
        FLAGS.class_loss_weight = trial.params["class_loss_weight"] 
        FLAGS.no_object_weight = trial.params["no_object_weight"]
        if "vitrolife" in FLAGS.dataset_name:
            FLAGS.num_queries = trial.params["num_queries"]
            FLAGS.dropout = trial.params["dropout"]
        if FLAGS.use_transformer_backbone==False and "Instance" not in FLAGS.segmentation:
            FLAGS.resnet_depth = trial.params["resnet_depth"]
            FLAGS.backbone_freeze_layers = trial.params["backbone_freeze"]
        del config 
        config = createVitrolifeConfiguration(FLAGS=FLAGS)
        config = changeConfig_withFLAGS(cfg=config, FLAGS=FLAGS)
    else: config = deepcopy(config)
    return config, FLAGS


# logs=log_file
# data_batches = None
# hyperparameter_optimization = True
# epoch = 0
# trial = None 

# Create function to train the objective function
def objective_train_func(trial, FLAGS, cfg, logs, data_batches=None, hyperparameter_optimization=False):
    # Setup training variables before starting training
    objective_mode = "training"
    if FLAGS.inference_only: objective_mode = "inference"
    if hyperparameter_optimization: objective_mode = "hyperparameter optimization trial {:d}/{:d}".format(FLAGS.HPO_current_trial+1, FLAGS.num_trials)
    printAndLog(input_to_write="Start {:s}...".format(objective_mode).upper(), logs=logs, postfix="\n")     # Print and log a message saying that a new iteration is now starting
    train_loader, val_loader, train_evaluators, val_evaluators, history, eval_train_results = None, None, None, None, None, None    # Initiates all the these as None values
    train_mode = "min" if "loss" in FLAGS.eval_metric else "max"                                            # Compute the mode of which the performance should be measured. Either a negative or a positive value is better
    new_best = np.inf if train_mode=="min" else -np.inf                                                     # Initiate the original "best_value" as either infinity or -infinity according to train_mode
    best_epoch = 0                                                                                          # Initiate the best epoch as being epoch_0, i.e. before doing any model training
    train_dataset = cfg.DATASETS.TRAIN                                                                      # Get the training dataset name
    val_dataset = cfg.DATASETS.TEST                                                                         # Get the validation dataset name
    lr_update_check = np.zeros((FLAGS.patience, 1), dtype=bool)                                             # Preallocating validation array to determine whether or not the learning rate was updated
    quit_training = False                                                                                   # Boolean value determining whether or not to commit early stopping
    epochs_to_run = 1 if hyperparameter_optimization else FLAGS.num_epochs                                  # We'll run only 1 epoch if we are performing HPO
    epochs_to_run = 1 if FLAGS.inference_only else epochs_to_run                                            # If we are just performing inference, then we'll only see the dataset once 
    train_start_time = time()                                                                               # Now the training starts
    epoch_next_display = FLAGS.display_rate - 1                                                             # The next epoch where the images must be visualized 

    # Change the FLAGS and config parameters and perform either hyperparameter optimization, use the best found parameters or simply just train
    config, FLAGS = get_HPO_params(config=cfg, FLAGS=FLAGS, trial=trial, hpt_opt=hyperparameter_optimization)
    
    # Train the model 
    run_type = "trial" if hyperparameter_optimization else "epoch"                                          # Either we are in a HPO trial or an epoch 
    total_runs = FLAGS.num_trials if hyperparameter_optimization else FLAGS.num_epochs                      # Get the total number of trials or epochs to run for 
    for epoch in range(epochs_to_run):                                                                      # Iterate over the chosen amount of epochs
        run_numb = FLAGS.HPO_current_trial+1 if hyperparameter_optimization else epoch+1                    # Get the current trial or the current epoch number 
        try:
            epoch_start_time = time()                                                                       # Now this new epoch starts
            if FLAGS.inference_only==False:
                config, quit_training = launch_custom_training(FLAGS=FLAGS, config=config, dataset=train_dataset,   # Launch the training ...
                        epoch=epoch, run_mode="train", hyperparameter_opt=hyperparameter_optimization)              # ... loop for one epoch
                if quit_training: break  
                if not hyperparameter_optimization:
                    eval_train_results, train_loader, train_evaluators,_,_ = evaluateResults(FLAGS, config, data_split="train", # Evaluate the results ...
                            dataloader=train_loader, evaluators=train_evaluators, hp_optim=hyperparameter_optimization)         # ... on the training set
            
            # Validation period. Will 'train' with lr=0 on validation data, correct the metrics files and evaluate performance on validation data
            config, quit_training = launch_custom_training(FLAGS=FLAGS, config=config, dataset=val_dataset, # Launch the training loop ...
                            epoch=epoch, run_mode="val", hyperparameter_opt=hyperparameter_optimization)    # ... for one single epoch
            if quit_training: break                                                                         # If the training must be quitted, break the for loop 
            eval_val_results, val_loader, val_evaluators,_,_ = evaluateResults(FLAGS, config,               # Evaluate the result metrics ...
                    data_split="val", dataloader=val_loader, evaluators=val_evaluators)                     # ... on the training set
            
            # Prepare for the training phase of the next epoch. Switch back to training dataset, save history and learning curves and visualize segmentation results
            config.DATASETS.TRAIN = train_dataset                                                           # Set the training dataset back 
            history = show_history(config=config, FLAGS=FLAGS, metrics_train=eval_train_results,            # Create and save the learning curves ...
                        metrics_eval=eval_val_results, history=history)                                     # ... including all training and validation metrics
            save_dictionary(dictObject=history, save_folder=config.OUTPUT_DIR, dictName="history")          # Save the history dictionary after each epoch
            [os.remove(os.path.join(config.OUTPUT_DIR, x)) for x in os.listdir(config.OUTPUT_DIR) if "events.out.tfevent" in x]
            
            # Performing callbacks
            if FLAGS.inference_only==False and hyperparameter_optimization==False: 
                config = keepAllButLatestAndBestModel(config=config, history=history, FLAGS=FLAGS)          # Keep only the best and the latest model weights. The rest are deleted.
                if epoch+1 >= FLAGS.patience:                                                               # If the model has trained for more than 'patience' epochs and we aren't debugging ...
                    config, lr_update_check = lr_scheduler(cfg=config, history=history, FLAGS=FLAGS, lr_updated=lr_update_check)  # ... change the learning rate, if needed
                    FLAGS.learning_rate = config.SOLVER.BASE_LR                                             # Update the FLAGS.learning_rate value
                if epoch+1 >= FLAGS.early_stop_patience:                                                    # If the model has trained for more than 'early_stopping_patience' epochs ...
                    quit_training = early_stopping(history=history, FLAGS=FLAGS)                            # ... perform the early stopping callback
            earlier_HPO_best = deepcopy(FLAGS.HPO_best_metric)                                              # Read the earlier best HPO value 
            earlier_train_best = deepcopy(new_best)                                                         # Read the earlier best train value 
            used_best_val = earlier_HPO_best if hyperparameter_optimization else earlier_train_best         # If we are performing HPO, use the best HPO_metric as baseline, else best training metric 
            new_best, best_epoch = updateLogsFunc(log_file=logs, FLAGS=FLAGS, history=history, best_val=new_best,
                    train_start=train_start_time, epoch_start=epoch_start_time, best_epoch=best_epoch,
                    cur_epoch=FLAGS.HPO_current_trial if hyperparameter_optimization else epoch)
            metrics_has_improved = all(["loss" in FLAGS.eval_metric, new_best < used_best_val]) or all(["loss" not in FLAGS.eval_metric, new_best > used_best_val])
            HPO_visualize = True if metrics_has_improved and hyperparameter_optimization else False
            train_visualize = True if epoch==epoch_next_display or all([metrics_has_improved, hyperparameter_optimization==False]) else False
            if HPO_visualize or train_visualize:                                                            # At least every 'display_rate' epochs or if the model has improved ...
                _,data_batches,config,FLAGS = visualize_the_images(config=config, FLAGS=FLAGS, data_batches=data_batches, epoch_num=epoch+1)  # ... the model will segment and save visualizations
                epoch_next_display = epoch + FLAGS.display_rate                                             # Increase the counter for when the images must be displayed again
            if all([quit_training, hyperparameter_optimization==False]):                                    # If the early stopping callback says we need to quit the training ...
                printAndLog(input_to_write="Committing early stopping at epoch {:d}. The best {:s} is {:.3f} from epoch {:d}".format(epoch+1, FLAGS.eval_metric, new_best, best_epoch), logs=logs)
                break                                                                                       # break the for loop and stop running more epochs
        except Exception as ex:
            error_string = "An exception of type {} occured while doing {} {}/{}. Arguments:\n{!r}".format(type(ex).__name__, run_type, run_numb, total_runs, ex.args)
            if isinstance(ex, FloatingPointError):
                printAndLog(input_to_write="Lowering the learning rate as the loss became nan or inf", logs=logs)
                config.SOLVER.BASE_LR = config.SOLVER.BASE_LR * FLAGS.lr_gamma
            printAndLog(input_to_write=error_string, logs=logs, prefix="", postfix="\n")

    # Evaluation on the vitrolife test dataset. There is no ADE20K-test dataset.
    test_history = {}                                                                                       # Initialize the test_history dictionary as an empty dictionary
    if all([FLAGS.debugging == False, "vitrolife" in FLAGS.dataset_name.lower(), hyperparameter_optimization==False]):  # Inference will only be performed when training the Vitrolife model
        config.DATASETS.TEST = ("vitrolife_dataset_test",)                                                  # The inference will be done on the test dataset
        eval_test_results,_,_,PN_pred_count,PN_true_count = evaluateResults(FLAGS, config, data_split="test")   # Evaluate the result metrics on the validation set with the best performing model
        if len(FLAGS.segmentation) > 1:
            raise(NotImplementedError("Only one type of segmentation at a time is allowed at the moment"))
        if "Instance" in FLAGS.segmentation:
            eval_metrics = eval_test_results[FLAGS.segmentation[0]]["segm"]
        if "Panoptic" in FLAGS.segmentation:
            eval_metrics = eval_test_results[FLAGS.segmentation[0]]["panoptic_seg"]
        history_test = combineDataToHistoryDictionaryFunc(config=config, eval_metrics=eval_metrics, data_split="test")
        for key in history_test.keys():                                                                     # Iterate over all the keys in the history dictionary
            if "test" in key: test_history[key] = history_test[key][-1]                                     # If "test" is in the key, assign the value to the test_dictionary 
        save_dictionary(dictObject=test_history, save_folder=config.OUTPUT_DIR, dictName="test_history")    # Save the test results in a dictionary 
        if "Instance" in FLAGS.segmentation:
            for dict_key in ["precision", "scores", "recall"]:                                              # Iterate over the three, long keys in the test dictionary ...
                del test_history["test_" + dict_key]                                                        # ... and delete them all after having saved the dictionary to a local file 

    # Return the results
    if hyperparameter_optimization: return new_best
    else: return history, test_history, new_best, best_epoch, config, PN_pred_count, PN_true_count
    
