# Import libraries
import os                                                                                               # Used to navigate different paths on the system
import json                                                                                             # Used to read the metrics_files from the output_dir
import pickle                                                                                           # Used for serializing and deserializing pkl objects 
import numpy as np                                                                                      # Used for division and floor/ceil operations here
import matplotlib.pyplot as plt                                                                         # The plotting package
from natsort import natsorted                                                                           # Function to natural sort a list or array 
from copy import deepcopy                                                                               # Used to make a copy of the key-name before replacing class name with class idx
from custom_image_batch_visualize_func import extractNumbersFromString                                  # Function to extract numbers from a string
from detectron2.data import MetadataCatalog                                                             # Catalogs for metadata for registered datasets


# Define a function to compute the moving average of an input array or list
def mov_avg_array(inp_array, mov_of_last_n_elements=4, output_last_n_elements=1):                       # Define a function to compute the moving average of an array or a list
    assert output_last_n_elements <= mov_of_last_n_elements, "The moving average can't be outputted for more values than it is being calculated for"
    if mov_of_last_n_elements > len(inp_array): mov_of_last_n_elements = len(inp_array)                 # If the list/array isn't as long as the wanted moving-average value, the n is lowered
    used_array_part = inp_array[-mov_of_last_n_elements:]                                               # Extract the last mov_of_last_n_elements from the list to compute the moving average for
    used_array_cumsum = np.cumsum(used_array_part)                                                      # Compute the cumulated sum for the used array part
    used_array_mov_avg = np.divide(used_array_cumsum, np.arange(1,1+mov_of_last_n_elements))            # Compute the moving average of the used array part
    return used_array_mov_avg[-output_last_n_elements:]                                                 # Output the last output_last_n_elements of the moving average array 


# Define a function to load the metrics.json in each output directory
def load_json_metrics(config, data_split="train"):
    metrics_list = natsorted([x for x in os.listdir(config["OUTPUT_DIR"]) if x.startswith("{:s}_metrics_".format(data_split))  # Loop through all files in the output directory ...
            and x.endswith(".json") and not np.isnan(extractNumbersFromString(x))])                     # ... and gather all the split_metrics_x.json files, where x=epoch_number and split=run_mode
    metrics = {"epoch_num": list()}                                                                     # Initiate the dictionary to store all the files
    for epoch_idx, metric_file in enumerate(metrics_list):                                              # Iterate over all found metrics files in the output directory
        for it_idx, line in enumerate(open(os.path.join(config.OUTPUT_DIR, metric_file))):              # Iterate over all lines in the current metrics file
            vals = json.loads(line)                                                                     # Read the current line in the current metrics file as a dictionary
            for key in vals:                                                                            # Iterate over all keys 
                if key not in list(metrics.keys()): metrics[key] = list()                               # If the key isn't already existing, create it as an empty list
                metrics[key].append(vals[key])                                                          # Append the current value from the current key
        metrics["epoch_num"].extend(np.repeat(a=epoch_idx+1, repeats=it_idx+1).tolist())                # Create a key named 'epoch_num' and repeat 'num_lines in metrics_file' the current epoch_numb to the dictionary
    for key in metrics.keys():                                                                          # Looping through all key values in the training metrics dictionary
        if "loss" not in key.lower(): continue                                                          # If the key is not a loss-key, skip to the next key
        key_val, mov_avg_val = list(), list()                                                           # Initiate lists to store the actual values and the moving-average computed values
        for item in metrics[key]:                                                                       # Loop through each item in the dict[key]->value list
            key_val.append(item)                                                                        # Append the actual item value to the key_val list
            mov_avg_val.append(mov_avg_array(inp_array=key_val, mov_of_last_n_elements=50, output_last_n_elements=1).item())    # Compute the next mov_avg val for the last 50 elements
        metrics[key] = mov_avg_val                                                                      # Assign the newly computed moving average of the dict[key]->values to the dictionary
    return metrics                                                                                      # Return the moving average value dictionary
# metrics = load_json_metrics(config=config)


# Create a function to replace the class_name in a string with the corresponding class_idx
def changeClassNameForClassIdxFunc(key, config):
    class_names = MetadataCatalog[config.DATASETS.TRAIN[0]].thing_classes                               # Get the class names for the dataset
    try: class_indices = list(MetadataCatalog[config.DATASETS.TRAIN[0]].thing_dataset_id_to_contiguous_id.keys())   # Get the class indices for the dataset ...
    except: class_indices = list(range(len(class_names)))                                               # ... or if that is not possible, assume class indices are a list from [0, K-1]
    if "vitrolife" in config.DATASETS.TRAIN[0].lower():                                                 # If we are using the vitrolife dataset ...
        class_indices = np.add(class_indices, 1).tolist()                                               # ... we add 1 to indices to start class_counting from 1 instead of 0, due to missing Background-class
    for class_name, class_lbl in zip(class_names, class_indices):                                       # Iterate over all the class names and the corresponding class labels
        if class_name.lower() in key.lower():                                                           # If the class name is in the key name ...
            key = key.lower().replace(class_name.lower(), "C{:d}".format(class_lbl))                    # ... the class_name part is replaced with the corresponding class label
            key = key.replace("ap", "AP").replace("-", "_").replace("iou", "IoU")                       # ... and we make sure the key is written in proper format
            break                                                                                       # ... and we stop iterating over the rest of the class_names
    return key                                                                                          # Return the new key name


# Create a function to extract the list of lists containing the keys that are relevant to show
def extractRelevantHistoryKeys(history, FLAGS):
    # Get the generic keys used for all segmentation types 
    loss_total = [key for key in history.keys() if "total_loss" in key.lower()]                         # Find all keys with loss_ce
    loss_ce = [key for key in history.keys() if "loss_ce" in key.lower() and key.endswith("ce")]        # Find all keys with loss_ce
    loss_dice = [key for key in history.keys() if "loss_dice" in key.lower() and key.endswith("dice")]  # Find all keys with loss_dice
    loss_mask = [key for key in history.keys() if "loss_mask" in key.lower() and key.endswith("mask")]  # Find all keys with loss_mask
    learn_rate = [key for key in history.keys() if "lr" in key.lower() and "val" not in key.lower()]    # Find the training learning rate

    # Get the instance keys 
    AP_total = [key for key in history.keys() if key.endswith("AP")]                                    # Find the total average precision AP@0.50:0.05:0.95
    AP_50 = [key for key in history.keys() if "AP" in key and key.endswith("50")]                       # Find the average precision AP50
    AP_75 = [key for key in history.keys() if "AP" in key and key.endswith("75")]                       # Find the AP75
    Precision_IoU50 = [key for key in history.keys() if "precision" in key and key.endswith("50")]      # Get the precision-recall curve from AP50

    # Get the panoptic keys 
    PQ_SQ_RQ_all = [key for key in history.keys() if any([key.endswith(x) for x in ["PQ", "RQ", "SQ"]])]
    PQ_SQ_RQ_things = [key for key in history.keys() if any([x in key for x in ["PQ", "RQ", "SQ"]]) and key.endswith("th")]
    PQ_SQ_RQ_stuff = [key for key in history.keys() if any([x in key for x in ["PQ", "RQ", "SQ"]]) and key.endswith("st")]
    PQ_all = [key for key in history.keys() if key.endswith("PQ")]
    SQ_all = [key for key in history.keys() if key.endswith("SQ")]
    RQ_all = [key for key in history.keys() if key.endswith("RQ")]
    PQ_things = [key for key in history.keys() if key.endswith("th") and "PQ" in key]
    SQ_things = [key for key in history.keys() if key.endswith("th") and "SQ" in key]
    RQ_things = [key for key in history.keys() if key.endswith("th") and "RQ" in key]
    PQ_stuff = [key for key in history.keys() if key.endswith("st") and "PQ" in key]
    SQ_stuff = [key for key in history.keys() if key.endswith("st") and "SQ" in key]
    RQ_stuff = [key for key in history.keys() if key.endswith("st") and "RQ" in key]
    SQ_RQ_all = [key for key in history.keys() if any([key.endswith(x) for x in ["RQ", "SQ"]])]
    SQ_RQ_things = [key for key in history.keys() if any([x in key for x in ["RQ", "SQ"]]) and key.endswith("th")]
    SQ_RQ_stuff = [key for key in history.keys() if any([x in key for x in ["RQ", "SQ"]]) and key.endswith("st")]

    # Return the list of lists of keys 
    if len(FLAGS.segmentation) > 1:
        raise(NotImplementedError("Only one type of segmentation at a time is allowed at the moment"))
    if "Instance" in FLAGS.segmentation:
        hist_keys_list = [loss_total, AP_total, Precision_IoU50, loss_ce,
                        loss_dice, loss_mask, AP_50, AP_75, learn_rate]
    if "Panoptic" in FLAGS.segmentation:
        hist_keys_list = [loss_total, PQ_all, SQ_RQ_all, PQ_things, SQ_RQ_things, PQ_stuff,
            SQ_RQ_stuff, loss_ce, loss_dice, loss_mask, learn_rate]
    return hist_keys_list


# Create a function to create the history dictionary
def combineDataToHistoryDictionaryFunc(config, eval_metrics=None, data_split="train", history=None, json_metrics=None):
    if history == None: history = {}                                                                    # Initiate the history dictionary that will be used
    if "train" in data_split: json_metrics = load_json_metrics(config=config, data_split="train")       # Load the metrics into the history dictionary
    if "val" in data_split: json_metrics = load_json_metrics(config=config, data_split="val")           # Load the metrics into the history dictionary
    if json_metrics is not None:                                                                        # If the json metrics has been loaded ...
        for key in json_metrics: history[data_split+"_"+key] = json_metrics[key]                        # ... append all the metrics losses to the dictionary with the split prefix on the key
    if eval_metrics is not None:                                                                        # If any evaluation metrics are available
        for key in eval_metrics.keys():                                                                 # Iterate over all keys in the history
            old_key = deepcopy(key)                                                                     # Make a copy now of the original key name from the metrics_train/eval dictionary
            key = changeClassNameForClassIdxFunc(key=key, config=config)                                # Exchange class_name with class_label in the key-name
            if data_split+"_"+key not in history:                                                       # If the given key doesn't exist ...
                history[data_split+"_"+key] = list()                                                    # ... add the key as an empty list and ... 
            if all(["precision" in key.lower(), "_iou" in key.lower()]):                                # If we are reading the precision-recall values for the current epoch ...
                history[data_split+"_"+key] = eval_metrics[old_key]                                     # ... simply replace the values from the earlier epoch 
            elif all([key in ["precision", "recall", "scores"], "test" not in data_split]):             # If the key is one of the precision, recall or score keys for training or validation ...
                epoch_split = "train" if "train" in data_split else "val"                               # Get the slit to read the epoch number from
                history[data_split+"_"+key].append({"Epoch_{}".format(np.max(history["{}_epoch_num".format(epoch_split)])): eval_metrics[old_key]}) # ... append a dict with key "epoch_num" and the key-value 
            elif all([key in ["precision", "recall", "scores"], "test" in data_split]):                 # If the key is one of the precision, recall or score keys for test values ...
                history[data_split+"_"+key] = eval_metrics[old_key]                                     # ... simply add the precision values without putting it in a dictionary first ... 
            else: history[data_split+"_"+key].append(eval_metrics[old_key])                             # Append the current key-value from the metrics_train to the corresponding list 
    return history

# for key in history.keys():
#     print(key)

# try:
#     config = config
# except:
#     config = cfg
# metrics_train = eval_train_results
# metrics_eval = eval_val_results
# history = None
# history = show_history(config=config, FLAGS=FLAGS, metrics_train=eval_train_results, metrics_eval=eval_val_results, history=history)


# Function to display learning curves
def show_history(config, FLAGS, metrics_train, metrics_eval, history=None):                             # Define a function to visualize the learning curves
    if len(FLAGS.segmentation) > 1:
        raise(NotImplementedError("Only one type of segmentation at a time is allowed at the moment"))
    if "Instance" in FLAGS.segmentation:
        segment_key_type = "segm" 
    if "Panoptic" in FLAGS.segmentation:
        segment_key_type = "panoptic_seg"
    if metrics_train is not None:
        metrics_train = metrics_train[FLAGS.segmentation[0]][segment_key_type]
    metrics_eval = metrics_eval[FLAGS.segmentation[0]][segment_key_type]
    
    # Create history and list of relevant history keys
    if FLAGS.inference_only==False or FLAGS.hp_optim==False or metrics_train is not None:
        history = combineDataToHistoryDictionaryFunc(config=config, eval_metrics=metrics_train, data_split="train", history=history)
    history = combineDataToHistoryDictionaryFunc(config=config, eval_metrics=metrics_eval, data_split="val", history=history)

    hist_keys = extractRelevantHistoryKeys(history, FLAGS)
    if "Instance" in FLAGS.segmentation:
        ax_titles = ["Total_loss", "AP@.5:.05:.95", "Precision_IoU@50", "Loss_CE",
                "Loss_DICE", "Loss_mask", "AP50", "AP75", "Learning_rate"] 
        n_cols = (3,3,3)
    if "Panoptic" in FLAGS.segmentation:
        ax_titles = ["Total_loss", "PQ_all", "RQ_SQ_all", "PQ_things", "SQ_RQ_things", "PQ_stuff", 
                    "SQ_RQ_stuff", "Loss_CE", "Loss_DICE", "Loss_mask", "Learning_rate"]
        n_cols = (3,4,4)
    colors = ["blue", "red", "black", "green", "magenta", "cyan", "yellow", "deeppink", "purple",       # Create colors for ... 
                "peru", "darkgrey", "gold", "springgreen", "orange", "crimson", "lawngreen"]            # ... the line plots
    n_rows, ax_count = 3, 0                                                                             # Initiate values for the number of rows and ax_counter 
    if FLAGS.num_classes > 10 and "Instance" in FLAGS.segmentation:                                     # If there are more than 10 classes (i.e. for ADE20K_dataset) ...
        n_rows, n_cols = 3, (4,4,4)                                                                     # ... the number of rows and columns gets reduced ...
        class_names = MetadataCatalog[config.DATASETS.TRAIN[0]].thing_classes                           # Get the class names for the dataset
        ax_tuple = [(ii,x) for (ii,x) in enumerate(ax_titles) if "PV_space" not in x and not any([y in x for y in class_names])]    # ... remove the class_specific ax_titles
        ax_titles = [x[1] for x in ax_tuple]                                                            # Get the new list of kept ax_titles
        indices = [x[0] for x in ax_tuple]                                                              # Get the indices of the accepted ax_titles
        hist_keys = np.asarray(hist_keys, dtype=object)[indices].tolist()                               # Get the new list of kept metrics to visualize
    
    # Display the figure
    fig = plt.figure(figsize=(int(np.ceil(np.max(n_cols)*5.7)), int(np.ceil(n_rows*5))))                # Create the figure
    for row in range(n_rows):                                                                           # Loop through all rows
        for col in range(n_cols[row]):                                                                  # Loop through all columns in the current row
            plt.subplot(n_rows, n_cols[row], 1+row*n_cols[row]+col)                                     # Create a new subplot
            plt.xlabel(xlabel="Epoch #")                                                                # Set correct xlabel
            plt.ylabel(ylabel=ax_titles[ax_count].replace("_", " "))                                    # Set correct ylabel
            plt.grid(True)                                                                              # Activate the grid on the plot
            plt.xlim(left=0, right=np.max(history["val_epoch_num"]))                                    # Set correct xlim
            plt.title(label=ax_titles[ax_count].replace("_", " "))                                      # Set plot title
            y_top_val = 0                                                                               # Initiate a value to determine the y_max value of the plot
            for kk, key in enumerate(sorted(hist_keys[ax_count], key=str.lower)):                       # Looping through all keys in the history dict that will be shown on the current subplot axes
                if np.max(history[key]) > y_top_val:                                                    # If the maximum value in the array is larger than the current y_top_val ...
                    y_top_val = np.ceil(np.max(history[key])/2)*2                                       # ... y_top_val is updated and rounded to the nearest 2
                start_val = np.min(history["val_epoch_num"])-(0 if any([x in key.lower() for x in ["ap", "precision", "pq", "sq", "rq"]]) else 1)   # The evaluation metrics must be plotted from after the first epoch, the losses from epoch=0
                x_vals = np.linspace(start=start_val, stop=np.max(history["val_epoch_num"]), num=len(history[key])) # Create the x-axis values as a linearly spaced array from epoch start_val to the latest epoch 
                if "precision" in key:                                                                  # If "precision" is in the key it means we are plotting a precision-recall curve  ...
                    x_vals = np.round(np.linspace(start=0, stop=1, num=len(history[key])), 2)           # ... with equally spaced recall values of R=[0, 0.01, 1] 
                    plt.xlim(left=np.min(x_vals), right=np.max(x_vals))                                 # Set the xlim for the precision-recall plot 
                    plt.xlabel(xlabel="Recall")                                                         # Set the xlabel for the precision-recall plot 
                    plt.ylabel(ylabel="Precision")                                                      # Set the ylabel for the precision-recall plot 
                y_val = np.asarray(history[key]).ravel()                                                # Read the y-values to plot
                plt.plot(x_vals, y_val, color=colors[kk], linestyle="-", marker=".")                    # Plot the x and y values
            plt.legend(sorted([key for key in hist_keys[ax_count]], key=str.lower),                     # Create a legend for the subplot with ...
                    framealpha=0.35, loc="best" if len(hist_keys[ax_count])<4 else "upper left")        # ... the history keys displayed
            ax_count += 1                                                                               # Increase the subplot counter
            if y_top_val <= 0.05 and "lr" not in key.lower():                                           # If the max y-value is super low ...
                plt.ylim(bottom=-0.05, top=0.05)                                                        # ... the limits are changed ...
            elif y_top_val < 10:                                                                        # If the max y-value is low ...
                plt.ylim(bottom=0, top=y_top_val*1.075)                                                 # ... the limits are changed ...
            else:                                                                                       # Otherwise ... 
                plt.ylim(bottom=0, top=y_top_val)                                                       # ... set the final, updated y_top_value as the y-top-limit on the current subplot axes
            if "learn" in ax_titles[ax_count].lower():                                                  # If we are plotting the learning rate ...
                plt.ylim(bottom=np.min(history[key])*0.9, top=np.max(history[key])*1.075)               # ... the y_limits are changed
                plt.yscale('log')                                                                       # ... the y_scale will be logarithmic
            if "precision" in ax_titles[ax_count].lower() and y_top_val > 0.20:
                plt.ylim(bottom=0, top=1) 
    try: fig.savefig(os.path.join(config.OUTPUT_DIR, "Learning_curves.jpg"), bbox_inches="tight")       # Try and save the figure in the OUTPUR_DIR ...
    except: pass                                                                                        # ... otherwise simply skip saving the figure
    fig.tight_layout()
    fig.show() if FLAGS.display_images==True else plt.close(fig)                                        # If the user chose to not display the figure, the figure is closed
    return history                                                                                      # The history dictionary is returned


# config = cfg
# config.OUTPUT_DIR = "/mnt/c/Users/Nico-/Documents/Python_Projects/Mask2Former/output_vitrolife_01_24_27APR2022"
# history_file = [os.path.join(config.OUTPUT_DIR,x) for x in os.listdir(config.OUTPUT_DIR) if "history" in x.lower()][0]
# with open(history_file, "rb") as f:
#     history = pickle.load(f)
