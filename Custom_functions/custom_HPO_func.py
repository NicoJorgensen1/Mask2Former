# Import important libraries
import os                                                                                           # Library for altering paths 
import optuna                                                                                       # Library used to perform hyperparameter optimization 
import numpy as np                                                                                  # For algebraic equations and isnan boolean values
import gc as garb_collect                                                                           # Used for garbage collecting after each hyperparameter trial
from time import time                                                                               # Used for timing the trials 
from copy import deepcopy                                                                           # Used for creating a new copy of a variable to memory
from sys import path as sys_PATH                                                                    # Import the PATH variable
from custom_mask2former_setup_func import setup_func, save_dictionary                               # Import the setup function and function to save dictionary 
from custom_print_and_log_func import printAndLog                                                   # Function to log the results 
from custom_training_func import objective_train_func                                               # The objective training function
from custom_callback_functions import computeRemainingTime                                          # Function to compute the remaining time of an iterating process 


# Function to tweak figures created by the Optuna visualization method
def tweak_figure_of_axes(axes):
    if isinstance(axes, np.ndarray):
        fig = axes[0,0].figure
        fig.set_size_inches((np.multiply(axes.shape, (4,5))))
        for row in range(axes.shape[0]):
            for col in range(axes.shape[1]):
                axes[row,col].xaxis.get_label().set_fontsize(20)
                axes[row,col].yaxis.get_label().set_fontsize(20)
                axes[row,col].title.set_fontsize(30)
                for tick in axes[row,col].xaxis.get_ticklabels():
                    tick.set_fontsize(15)
                for tick in axes[row,col].yaxis.get_ticklabels():
                    tick.set_fontsize(15)
    if hasattr(axes, "plot"):
        fig = axes.figure
        fig.set_size_inches((18,11))
        axes.xaxis.get_label().set_fontsize(15)
        axes.yaxis.get_label().set_fontsize(15)
        axes.title.set_fontsize(30)
        for tick in axes.xaxis.get_ticklabels():
            tick.set_fontsize(15)
        for tick in axes.yaxis.get_ticklabels():
            tick.set_fontsize(15)
    return fig


# The objective function that will be run by the Optuna HPO study
def object_func(trial):
    HPO_trial_start = time()
    new_best = float("nan")
    it_count = 0
    while np.isnan(new_best):                                                                       # Sometimes the iteration fails for some reason. We'll allow 3 attempts before skipping and moving on
        try:
            new_best = objective_train_func(trial=trial, FLAGS=FLAGS, cfg=cfg, logs=log_file, data_batches=["this is an empty list"], hyperparameter_optimization=True)
        except Exception as ex:
            error_str = "An exception of type {0} occured. Arguments:\n{1!r}".format(type(ex).__name__, ex.args)
            printAndLog(input_to_write=error_str, logs=log_file, postfix="\n")
            new_best = float("nan")
        it_count += 1
        if it_count >= 3:
            printAndLog(input_to_write="", logs=log_file, print_str=False)
            break 
    [os.remove(os.path.join(cfg.OUTPUT_DIR, x)) for x in os.listdir(cfg.OUTPUT_DIR) if all([x.endswith(".pth"), "model" in x.lower()])]
    [os.remove(os.path.join(cfg.OUTPUT_DIR, x)) for x in os.listdir(cfg.OUTPUT_DIR) if all([x.endswith(".json"), "metrics" in x.lower()])]
    [os.remove(os.path.join(cfg.OUTPUT_DIR, x)) for x in os.listdir(cfg.OUTPUT_DIR) if "events.out.tfevents" in x]
    string1, string2 = computeRemainingTime(epoch=FLAGS.HPO_current_trial, num_epochs=FLAGS.num_trials, train_start_time=HPO_start, epoch_start_time=HPO_trial_start)
    string1 = string1.replace("epoch", "trial").replace("training", "search")
    string2 = string2.replace("epoch", "trial")
    printAndLog(input_to_write=string1, logs=log_file, prefix="", postfix="\n")
    if FLAGS.HPO_current_trial < FLAGS.num_trials-1:
        printAndLog(input_to_write=string2, logs=log_file, prefix="")
    FLAGS.HPO_current_trial += 1
    return new_best


FLAGS, cfg, log_file = setup_func()                                                                 # Get the FLAGS, config and log_file 
Mask2Former_dir = [x for x in sys_PATH if x.endswith("Mask2Former")][0]                             # Get the path of the MaskFormer directory
HPO_start = time()                                                                                  # Set the time for the start of the HPO 
def perform_HPO():                                                                                  # The function that will perform the HPO
    trial = None                                                                                    # Initiate the trial as a None variable in order to return it if no HPO is performed
    if FLAGS.hp_optim:                                                                              # If the user chose to perform a HPO, then do so...
        warm_ups = deepcopy(FLAGS.warm_up_epochs)                                                   # Make a deepcopy of the number of warmups
        FLAGS.warm_up_epochs = 0                                                                    # Set the number of warmup epochs to 0 when performing HPO
        TPE_sampler = optuna.samplers.TPESampler(n_startup_trials=FLAGS.num_random_trials)          # Initiate the search with some random samples to explore the search space, before starting to optimize
        study_name = "Hyperparameter optimization for {:s} dataset".format(FLAGS.dataset_name)      # Unique identifier for the study saved in memory
        storage_file = "sqlite:///{}.db".format(os.path.join(cfg.OUTPUT_DIR, study_name))           # Create a database file in local memory to store the study 
        study_direction = "minimize" if "loss" in FLAGS.eval_metric else "maximize"                 # The direction in which we want the HPO metric to go
        study = optuna.create_study(sampler=TPE_sampler, study_name=study_name, direction=study_direction, storage=storage_file, load_if_exists=True)    # Needs all these arguments to reload a study ...
        study.optimize(object_func, n_trials=FLAGS.num_trials, callbacks=[lambda study, trial: garb_collect.collect()], catch=(MemoryError,), gc_after_trial=True)
        trial = study.best_trial 
        best_params = trial.params 
        save_dictionary(dictObject=best_params, save_folder=cfg.OUTPUT_DIR, dictName="best_HPO_params")
        printAndLog(input_to_write="Hyperparameter optimization completed.\nBest {:s}: {:.3f}".format(FLAGS.eval_metric, trial.value), logs=log_file, prefix="\n")
        printAndLog(input_to_write="Best hyperparameters: ".ljust(25), logs=log_file)
        printAndLog(input_to_write={key: best_params[key] for key in sorted(best_params.keys(), reverse=True)}, logs=log_file, prefix="", postfix="\n", length=15)
        FLAGS.warm_up_epochs = warm_ups

        # Create a new study object only containing the 5 best and worst trials - just used for plotting the parallel plot
        trials_list, eval_metric_list = list(), list()
        for hpo_trial in study.trials:
            if hpo_trial.values is None: continue
            if np.isnan(hpo_trial.values[-1]): continue
            trials_list.append(hpo_trial)
            eval_metric_list.append(hpo_trial.values[-1])
        vals_to_keep = np.unique(np.argsort(eval_metric_list)[:5].tolist() + np.argsort(eval_metric_list)[-5:].tolist())
        trials_to_keep = np.asarray(trials_list)[vals_to_keep]
        other_study_name = study_name+"_5_best_5_worst_trials"
        other_storage_file = storage_file.replace(study_name, other_study_name)
        small_study = optuna.create_study(sampler=TPE_sampler, study_name=other_study_name, direction=study_direction, storage=other_storage_file)
        small_study.add_trials(trials_to_keep)


        #################### THIS IS JUST NOW FOR DEBUGGING ##########################
        # try:
        #     reload_old_study_storage = storage_file.replace(os.path.basename(cfg.OUTPUT_DIR), "output_vitrolife_13_27_01APR2022")
        #     reload_old_study = optuna.create_study(sampler=TPE_sampler, study_name=study_name, direction=study_direction, storage=reload_old_study_storage, load_if_exists=True)
        #     study.add_trials(reload_old_study.trials)
        # except: pass
        #################### THIS IS JUST NOW FOR DEBUGGING ##########################

        
        # Plot the results. Some problems have occured earlier, thus all plots are wrapped in try-except loops...
        HPO_fig_folder = os.path.join(cfg.OUTPUT_DIR, "Visualizations", "HPO_figures")
        params_to_use = ["learning_rate", "mask_loss_weight", "weight_decay"] if "nico" in Mask2Former_dir.lower() else None
        os.makedirs(HPO_fig_folder, exist_ok=True) 
        try:
            contour_axes = optuna.visualization.matplotlib.plot_contour(study, params=params_to_use, target_name=FLAGS.eval_metric)
            fig_contour = tweak_figure_of_axes(axes=contour_axes)
            fig_contour.savefig(os.path.join(HPO_fig_folder, "Contour_plot_of_params.jpg"))
        except Exception as ex:
            error_string = "An exception of type {} occured while creating the contour plot. Arguments:\n{!r}".format(type(ex).__name__, ex.args)
            printAndLog(input_to_write=error_string, logs=log_file, prefix="", postfix="\n")
        try:
            hp_importance_axes = optuna.visualization.matplotlib.plot_param_importances(study, params=params_to_use)
            fig_hp_importance = tweak_figure_of_axes(axes=hp_importance_axes)
            fig_hp_importance.savefig(os.path.join(HPO_fig_folder, "Importance_of_hyperparameters.jpg"), bbox_inches="tight")
        except Exception as ex:
            error_string = "An exception of type {} occured while creating the param_importance_plot. Arguments:\n{!r}".format(type(ex).__name__, ex.args)
            printAndLog(input_to_write=error_string, logs=log_file, prefix="", postfix="\n")
        try:
            optim_axes = optuna.visualization.matplotlib.plot_optimization_history(study)
            fig_optim = tweak_figure_of_axes(axes=optim_axes)
            fig_optim.savefig(os.path.join(HPO_fig_folder, "Optimization_history.jpg"), bbox_inches="tight")
        except Exception as ex:
            error_string = "An exception of type {} occured while creating the plot_optimization_history plot. Arguments:\n{!r}".format(type(ex).__name__, ex.args)
            printAndLog(input_to_write=error_string, logs=log_file, prefix="", postfix="\n")            
        try:
            parallel_axes = optuna.visualization.matplotlib.plot_parallel_coordinate(study, params=params_to_use, target_name=FLAGS.eval_metric)
            fig_parallel = tweak_figure_of_axes(axes=parallel_axes)
            fig_parallel.savefig(os.path.join(HPO_fig_folder, "Parallel_axes.jpg"), bbox_inches="tight")
        except Exception as ex:
            error_string = "An exception of type {} occured while creating parallel_coordinates_plot for all trials. Arguments:\n{!r}".format(type(ex).__name__, ex.args)
            printAndLog(input_to_write=error_string, logs=log_file, prefix="", postfix="\n")
        try:
            parallel_axes2 = optuna.visualization.matplotlib.plot_parallel_coordinate(small_study, params=params_to_use, target_name=FLAGS.eval_metric)
            fig_parallel2 = tweak_figure_of_axes(axes=parallel_axes2)
            fig_parallel2.savefig(os.path.join(HPO_fig_folder, "Parallel_axes_10_best_10_worst.jpg"), bbox_inches="tight")
        except Exception as ex:
            error_string = "An exception of type {} occured while creating parallel_coordinates_plot for the 10 best and 10 worst trials. Arguments:\n{!r}".format(type(ex).__name__, ex.args)
            printAndLog(input_to_write=error_string, logs=log_file, prefix="", postfix="\n")
        try:
            EDF_axes = optuna.visualization.matplotlib.plot_edf(study, target_name=FLAGS.eval_metric)
            fig_EDF = tweak_figure_of_axes(axes=EDF_axes)
            fig_EDF.savefig(os.path.join(HPO_fig_folder, "Empirical_Distribution_Plot.jpg"), bbox_inches="tight")
        except Exception as ex:
            error_string = "An exception of type {} occured while creating EDF plot. Arguments:\n{!r}".format(type(ex).__name__, ex.args)
            printAndLog(input_to_write=error_string, logs=log_file, prefix="", postfix="\n")

    return FLAGS, cfg, trial, log_file
