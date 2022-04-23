# Add the Mask2Former directory to PATH
import os                                                                                               # Used to navigate the folder structure in the current os
import sys                                                                                              # Used to control the PATH variable
Mask2Former_dir = os.path.join("/mnt", "c", "Users", "Nico-", "Documents", "Python_Projects", "Mask2Former")                                                                # Home WSL
if not os.path.isdir(Mask2Former_dir): Mask2Former_dir = os.path.join("C:\\", Mask2Former_dir.split(os.path.sep, 1)[1])                                                     # Home windows computer
if not os.path.isdir(Mask2Former_dir): Mask2Former_dir = os.path.join("/mnt", "c", "Users", "wd974261", "Documents", "Python", "Mask2Former")                               # Work WSL
if not os.path.isdir(Mask2Former_dir): Mask2Former_dir = os.path.join("C:\\", Mask2Former_dir.split(os.path.sep, 1)[1])                                                     # Work windows computer
if not os.path.isdir(Mask2Former_dir): Mask2Former_dir = os.path.join("/home", "neal", "Panoptic_segmentation_using_deep_neural_networks", "Repositories", "Mask2Former")   # Larac server
if not os.path.isdir(Mask2Former_dir): Mask2Former_dir = os.path.join("/mnt", "home_shared", Mask2Former_dir.split(os.path.sep, 2)[2])                                      # Balder server
assert os.path.isdir(Mask2Former_dir), "The Mask2Former directory doesn't exist in the chosen location"
sys.path.append(Mask2Former_dir)                                                                        # Add Mask2Former directory to PATH
sys.path.append(os.path.join(Mask2Former_dir, "Custom_functions"))                                      # Add Custom_functions directory to PATH
sys.path.append(os.path.join(Mask2Former_dir, "tools"))                                                 # Add the tools directory to PATH

# Add the environmental variable DETECTRON2_DATASETS
dataset_dir = os.path.join("/mnt", "c", "Users", "Nico-", "OneDrive - Aarhus Universitet", "Biomedicinsk Teknologi", "5. semester", "Speciale", "Datasets")                 # Home WSL
if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("C:\\", dataset_dir.split(os.path.sep,1)[1])                                                                  # Home windows computer
if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("/mnt", "c", "Users", "wd974261", "Documents", "Python", "Datasets")                                          # Work WSL
if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("C:\\", dataset_dir.split(os.path.sep,1)[1])                                                                  # Work windows computer
if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("/home", "neal", "Panoptic_segmentation_using_deep_neural_networks", "Datasets")                              # Larac server
if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("/mnt", "home_shared", dataset_dir.split(os.path.sep, 2)[2])                                                  # Balder server
assert os.path.isdir(dataset_dir), "The dataset directory doesn't exist in the chosen location"
os.environ["DETECTRON2_DATASETS"] = dataset_dir

# Import important libraries
from custom_callback_functions import keepAllButLatestAndBestModel                                      # Used for setting model weights on the config
from custom_mask2former_setup_func import getBestEpochResults, zip_output, write_config_to_file         # Get metrics from the best epoch, zip output directory and write config to file
from custom_print_and_log_func import printAndLog                                                       # Function to log the results
from custom_analyze_model_func import analyze_model_func                                                # Analyze the model FLOPS, number of parameters and activations computed
from custom_training_func import objective_train_func                                                   # Function to launch the training with the given dataset
from custom_image_batch_visualize_func import visualize_the_images                                      # Functions visualize the image batch
from custom_HPO_func import perform_HPO                                                                 # Function to perform HPO and read the input variables


##### Need a better way to visualize the image batch
    # Perhaps take a look at these three files here:
    # /mnt/c/Users/Nico-/Documents/Python_Projects/Mask2Former/mask2former/modeling/matcher.py
    # /mnt/c/Users/Nico-/Documents/Python_Projects/Mask2Former/mask2former/maskformer_model.py
    # /mnt/c/Users/Nico-/Documents/Python_Projects/Mask2Former/mask2former/modeling/criterion.py
    # Or check out this implementation from scipy => remember how the HungarianMatcher class computes the cost matrix, C ... 
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
##### Try-except from both HPO function and custom_train_func needs to be in-commented again 


# Get the FLAGS, the config and the logfile. 
FLAGS, cfg, trial, log_file = perform_HPO()                                                             # Perform HPO if that is chosen 
printAndLog(input_to_write="FLAGS input arguments:", logs=log_file)                                     # Print the new, updated FLAGS ...
printAndLog(input_to_write={key: vars(FLAGS)[key] for key in sorted(vars(FLAGS).keys())},               # ...  input arguments to the logfile ...
            logs=log_file, oneline=False, length=27)                                                    # ... sorted by the key names 

# Analyze the model with the found parameters from the HPO
model_analysis, FLAGS = analyze_model_func(config=cfg, args=FLAGS)                                      # Analyze the model with the FLAGS input parameters
printAndLog(input_to_write="Model analysis:".upper(), logs=log_file)                                    # Print the model analysis ...
printAndLog(input_to_write=model_analysis, logs=log_file, oneline=False, length=27)                     # ... and write it to the logfile

# Visualize some random images before training 
fig_list_before, data_batches, cfg, FLAGS = visualize_the_images(config=cfg, FLAGS=FLAGS)               # Visualize some segmentations on random images before training

# Train the model with the best found hyperparameters
history, test_history, new_best, best_epoch, cfg = objective_train_func(trial=trial, FLAGS=FLAGS,       # Start the training with ...
            cfg=cfg, logs=log_file, data_batches=data_batches, hyperparameter_optimization=False)       # ... the optimal hyper parameters

# Visualize the same images, now after training
cfg = keepAllButLatestAndBestModel(cfg=cfg, history=history, FLAGS=FLAGS, bestOrLatest="best")          # Put the model weights for the best performing model on the config
write_config_to_file(config=cfg)                                                                        # Save the config file with the final parameters used in the output dir
visualize_the_images(config=cfg,FLAGS=FLAGS, data_batches=data_batches, model_done_training=True)       # Visualize the images again

# Print and log the best metric results
printAndLog(input_to_write="Final results:".upper(), logs=log_file)
if FLAGS.inference_only==False: 
    printAndLog(input_to_write="Best validation results:".ljust(30)+"Epoch {:d}: {:s} = {:.3f}\n{:s}".
        format(best_epoch, FLAGS.eval_metric, new_best, "All best validation results:".upper().ljust(30)), logs=log_file)
    printAndLog(input_to_write=getBestEpochResults(history, best_epoch), logs=log_file, prefix="", length=15)
if "vitrolife" in FLAGS.dataset_name.lower():                                                           # As only the Vitrolife dataset includes a test set...
    printAndLog(input_to_write="All test results:".upper().ljust(30), logs=log_file)
    printAndLog(input_to_write=test_history, logs=log_file, prefix="", length=15)

# Remove all metrics.json files, the default log-file and zip the resulting output directory
[os.remove(os.path.join(cfg.OUTPUT_DIR, x)) for x in os.listdir(cfg.OUTPUT_DIR) if "metrics" in x.lower() and x.endswith(".json")]
os.remove(os.path.join(cfg.OUTPUT_DIR, "log.txt"))
zip_output(cfg)





