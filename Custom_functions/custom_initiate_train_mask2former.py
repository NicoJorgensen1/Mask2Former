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



#### Modify the function that will allow for displaying learning curves
#### Create function to visualize an image batch with predicted outputs 
#### Create a new function to compute confusion matrixes. The values might as well be the AP@50 of each class. Visualize using the custom_conf_matrix_visualization_func script 
#### Modify this initiate_train script in order to complete the training after performing HPO 



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

