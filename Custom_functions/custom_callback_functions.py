# Import libraries
import os
import numpy as np
from time import time                                                                           # Used to time the epoch training duration
from copy import deepcopy                                                                       # Used to make a new copy of the config 
from natsort import natsorted                                                                   # Used to sort the list of model_files saved 
from custom_image_batch_visualize_func import extractNumbersFromString                          # Function to extract numbers from a string
from custom_print_and_log_func import printAndLog                                               # Used to update the log file


# Define a function to commit early stopping
def early_stopping(history, FLAGS, quit_training=False):
    mode = "min" if "loss" in FLAGS.eval_metric.lower() else "max"                              # Whether a lower value or a higher value is better
    metric_monitored = history[FLAGS.eval_metric][-FLAGS.early_stop_patience+1:]                # Getting the last 'early_stop_patience' values of the 'monitor' metric
    if np.max(history["train_epoch_num"]) > FLAGS.early_stop_patience+FLAGS.warm_up_epochs:     # If we have run for at least FLAGS.early_stop_patience epochs, we'll continue
        if mode=="max": val_used = np.max(metric_monitored)                                     # If we monitor an increasing metric, we want to find the largest value
        if mode=="min": val_used = np.min(metric_monitored)                                     # If we monitor a decreasing metric, we want to find the smallest value    
        if all([mode=="max", val_used <= metric_monitored[0] + FLAGS.min_delta]) or all([mode=="min", val_used >= metric_monitored[0] - FLAGS.min_delta]):  # If the model hasn't ...
            quit_training = True                                                                # ... improved in the last 'early_stop_patience' epochs, the training is terminated
            FLAGS.quit_training = quit_training
    return quit_training


# Define a function to lower the learning rate
def lr_scheduler(cfg, history, FLAGS, lr_updated):
    cfg.SOLVER.BASE_LR = FLAGS.learning_rate                                                    # Assign the given learning rate to the config
    lr_updated[-1] = False                                                                      # As we are include the current epoch in the last patience epochs, the final value must be set to False
    lr_updated = np.roll(a=lr_updated, shift=1)                                                 # Shifts all value indices by 1, i.e. a[0:]=a[-1:]
    metric_monitored = history[FLAGS.eval_metric][-FLAGS.patience:]                             # Getting the last 'patience' values of the 'monitor' metric
    if not any(lr_updated) and np.max(history["train_epoch_num"]) >= FLAGS.patience+FLAGS.warm_up_epochs:   # If no learning rate updates has been made in the last 'patience' epochs...
        mode = "min" if "loss" in FLAGS.eval_metric.lower() else "max"                          # Whether a lower value or a higher value is better
        if mode=="max": val_used = np.max(metric_monitored)                                     # If we monitor an increasing metric, we want to find the largest value
        if mode=="min": val_used = np.min(metric_monitored)                                     # If we monitor a decreasing metric, we want to find the smallest value    
        if mode=="max" and val_used <= metric_monitored[0] + FLAGS.min_delta or mode=="min" and val_used >= metric_monitored[0] - FLAGS.min_delta:  # If the model hasn't improved in the ...
            cfg.SOLVER.BASE_LR = FLAGS.learning_rate * FLAGS.lr_gamma                           # ... last 'patience' epochs the learning rate is lowered
    lr_updated[0] = False if cfg.SOLVER.BASE_LR == FLAGS.learning_rate else True                # If the learning rate was updated, so is the lr_updated array
    return cfg, lr_updated


# Define a function to delete all models but the 
def keepAllButLatestAndBestModel(config, history, FLAGS, bestOrLatest="latest", delete_leftovers=True, logs=None, model_done_training=False):
    cfg = deepcopy(config) 
    del config
    model_files = natsorted([x for x in os.listdir(cfg.OUTPUT_DIR) if "model_epoch" in x.lower()])  # Get a list of available models
    if len(model_files) >= 1:                                                                   # If any model files have been saved yet ...
        mode = "min" if "loss" in FLAGS.eval_metric.lower() else "max"                          # Whether a lower value or a higher value is better
        epoch_numbers = [extractNumbersFromString(x, dtype=int) for x in model_files]           # Extract which epoch each model is from
        metric_list = history[FLAGS.eval_metric]                                                # Read the list of values for the metric chosen to use as 
        metric_list = np.asarray(metric_list)[np.subtract(epoch_numbers,1)]                     # Get the remaining values (i.e. if some models have already been deleted, their corresponding values should be removed)
        best_model_idx = np.argmin(metric_list) if mode=="min" else np.argmax(metric_list)      # Get the idx of the best epoch
        best_epoch = np.asarray(epoch_numbers)[best_model_idx]                                  # Get the best model filename
        best_model = model_files[best_model_idx]                                                # Get the model name of the best model
        latest_model = model_files[np.argmax(epoch_numbers)]                                    # Get the model name of the latest model
        models_to_delete = []                                                                   # Initiate a list of the model filenames that will be deleted
        if delete_leftovers == True:                                                            # If we want to delete the leftovers ...
            for model in model_files:                                                           # ... we'll iterate over all model files ...
                if any([best_model in model, latest_model in model]) and not model_done_training: continue  # ... and if the model hasn't finished training and the current model name is neither best or latest ...
                elif all([model_done_training, best_model in model]): continue                  # ... or if the model has finished training and the current model isn't the best model ...
                os.remove(os.path.join(cfg.OUTPUT_DIR, model))                                  # ... it is deleted ...
                models_to_delete.append(model)                                                  # ... and the model name is added to the list of models that must be deleted
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, latest_model if "latest" in bestOrLatest.lower() else best_model)  # Set the model weights as either the best or the latest model
        model_to_keep = latest_model if "latest" in bestOrLatest.lower() else best_model        # Whether or not the latest model or the best model will be the one we use 
        if logs is not None:
            printAndLog(input_to_write="The model files: {}. The metrics_list {}".format(model_files, metric_list), logs=logs, prefix="\n\n")
            printAndLog(input_to_write="The best model idx is: {}, thus epoch {}, named {}. The latest model is {}".format(best_model_idx, best_epoch, best_model, latest_model), logs=logs)
            printAndLog(input_to_write="bestOrLatest: {} => {}. Thus model_to_keep = {}".format(bestOrLatest, latest_model if "latest" in bestOrLatest.lower() else best_model, model_to_keep), logs=logs)
            printAndLog(input_to_write="The model_weights in the config: {}".format(cfg.MODEL.WEIGHTS), logs=logs)
            printAndLog(input_to_write="The models to delete: {}".format(models_to_delete), logs=logs, postfix="\n\n")
    return cfg                                                                                  # Return the config where the cfg.MODEL.WEIGHTS are set to the chosen model


# Function for converting seconds into days:hr:min:sec
def secondsToDaysHrsMinSec(seconds):
    days = np.floor(seconds/(3600*24))                                                          # Compute the number of days the amount of seconds span
    hrs = np.floor((seconds-days*3600*24)/3600)                                                 # Compute the remaining amount of hours the amount of seconds span
    mins = np.floor((seconds-days*3600*24-hrs*3600)/60)                                         # Compute the remaining amount of minutes the amount of seconds span
    secs = np.floor(np.mod(seconds, 60))                                                        # Compute the remaining amount of seconds the amount of seconds span
    days_string = "{:.0f}days:".format(days) if days > 0 else ""                                # Initiate a string with how many days are in the amount of seconds
    hrs_string = "{:.0f}hr:".format(hrs) if (days > 0 or hrs > 0) else ""                       # Initiate a string with how many hours are in the amount of seconds
    mins_string = "{:.0f}min:".format(mins) if (days > 0 or hrs > 0 or mins > 0) else ""        # Initiate a string with how many minuts are in the amount of seconds
    secs_string = "{:.0f}sec".format(secs) if (days > 0 or hrs > 0 or mins > 0 or secs > 30) else "{:.1f}sec".format(secs)  # Initiate a string with how many seconds are in the amount of seconds
    time_string = days_string+hrs_string+mins_string+secs_string                                # Combine all these strings into a single formatted string
    return time_string


# Compute remaining time 
def computeRemainingTime(epoch=0, num_epochs=None, train_start_time=time(), epoch_start_time=time()):
    # Compute time left for training and print it out if wanted
    epoch_duration = time() - epoch_start_time                                                  # Calculate how long this epoch has taken
    train_duration = time() - train_start_time                                                  # Calculate how long the training so far has taken
    estTrainTime = train_duration * num_epochs/(epoch+1)                                        # Estimate how long the entire training will take
    timeLeft = estTrainTime - train_duration                                                    # Estimate how much of the entire training time is remaining
    epochs_left = num_epochs-epoch-1                                                            # Compute how many epochs are remaining

    # Print and log the timed results
    string1 = "Now this {:.0f}. of {:.0f} epochs took {:s}, making the total training time {:s}".format(epoch+1, num_epochs, secondsToDaysHrsMinSec(epoch_duration), secondsToDaysHrsMinSec(train_duration))
    string2 = "Now the last {:.0f} epoch{:s} are expected to take {:s}\n".format(epochs_left, "s" if epochs_left>1 else "", secondsToDaysHrsMinSec(timeLeft))
    return string1, string2


# Write to the log file if the model has improved
def updateLogsFunc(log_file, FLAGS, history, best_val, train_start, epoch_start, best_epoch, cur_epoch):
    train_mode = "min" if "loss" in FLAGS.eval_metric else "max"
    epoch = cur_epoch+1
    string1, string2 = computeRemainingTime(epoch=epoch-1, num_epochs=FLAGS.num_epochs, train_start_time=train_start, epoch_start_time=epoch_start)

    # Read the keys to use 
    if len(FLAGS.segmentation) > 1:
        raise(NotImplementedError("Only one type of segmentation at a time is allowed at the moment"))
    if "Instance" in FLAGS.segmentation:
        keywords = ["AP"]
    if "Panoptic" in FLAGS.segmentation:
        keywords = ["PQ", "RQ", "SQ"]
        
    # Read the latest evaluation results
    metrics_train_keys = [x for x in history.keys() if "train" in x and any([y in x for y in keywords])]
    metrics_train = {key: history[key][-1] for key in metrics_train_keys}

    # Update the logfile 
    if any([FLAGS.HPO_current_trial >= FLAGS.num_trials, FLAGS.hp_optim==False]):
        printAndLog(input_to_write=string1, logs=log_file, postfix="\n")
    printAndLog(input_to_write="Train metrics:".ljust(25), logs=log_file, prefix="", postfix="")
    printAndLog(input_to_write=metrics_train, logs=log_file, oneline=True, prefix="", postfix="\n", length=15)
    metrics_val_keys = [x for x in history.keys() if "val" in x and any([y in x for y in keywords])]
    metrics_val = {key: history[key][-1] for key in metrics_val_keys}
    printAndLog(input_to_write="Validation metrics:".ljust(25), logs=log_file, prefix="")
    printAndLog(input_to_write=metrics_val, logs=log_file, oneline=True, prefix="", postfix="\n", length=15)
    if train_mode=="min": new_best = np.min(history[FLAGS.eval_metric])
    if train_mode=="max": new_best = np.max(history[FLAGS.eval_metric])
    currently_training = any([FLAGS.HPO_current_trial >= FLAGS.num_trials, FLAGS.hp_optim==False])
    if np.abs(new_best-best_val) >= FLAGS.min_delta:
        train_type = "Epoch" if currently_training else "Trial"
        performance_string = "{:s}{:d}: The {:s} has improved from {:.3f} to {:.3f}".format(train_type.ljust(8), epoch, FLAGS.eval_metric, best_val, new_best)
        if currently_training == False:
            study_direction = "minimize" if "loss" in FLAGS.eval_metric else "maximize"
            if "minimize" in study_direction and new_best < FLAGS.HPO_best_metric: FLAGS.HPO_best_metric = float(new_best)
            if "maximize" in study_direction and new_best > FLAGS.HPO_best_metric: FLAGS.HPO_best_metric = float(new_best)
            performance_string = performance_string + ". The best value obtained is: {:.3f}".format(FLAGS.HPO_best_metric)
        printAndLog(input_to_write=performance_string, logs=log_file, prefix="", postfix="\n")
        best_val = new_best
        best_epoch = epoch
    if all([epoch < FLAGS.num_epochs, currently_training==True, FLAGS.quit_training==False]):
        printAndLog(input_to_write=string2, logs=log_file, prefix="")
    return best_val, best_epoch


