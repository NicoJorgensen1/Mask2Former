# Import libraries
import os                                                                                               # Used to navigate different paths on the system
import json                                                                                             # Used to read the metrics_files from the output_dir
import numpy as np                                                                                      # Used for division and floor/ceil operations here
import matplotlib.pyplot as plt                                                                         # The plotting package
from natsort import natsorted                                                                           # Function to natural sort a list or array 
from copy import deepcopy                                                                               # Used to make a copy of the key-name before replacing class name with class idx
from visualize_image_batch import extractNumbersFromString                                              # Function to extract numbers from a string
from detectron2.data import MetadataCatalog                                                             # Catalogs for metadata for registered datasets

# config = cfg
# config.OUTPUT_DIR = "/mnt/c/Users/Nico-/Documents/Python_Projects/MaskFormer/output_vitrolife_13_25_14MAR2022"


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
    class_names = MetadataCatalog[config.DATASETS.TRAIN[0]].stuff_classes                               # Get the class names for the dataset
    try: class_indices = list(MetadataCatalog[config.DATASETS.TRAIN[0]].stuff_dataset_id_to_contiguous_id.keys())   # Get the class indices for the dataset ...
    except: class_indices = list(range(len(class_names)))                                               # ... or if that is not possible, assume class indices are a list from [0, K-1]
    for class_name, class_lbl in zip(class_names, class_indices):                                       # Iterate over all the class names and the corresponding class labels
        if class_name.lower() in key.lower():                                                           # If the class name is in the key name ...
            key = key.lower().replace(class_name.lower(), "C{:d}".format(class_lbl))                    # ... the class_name part is replaced with the corresponding class label
            key = key.replace("-", "_").replace("acc", "ACC").replace("iou", "IoU").replace("-", "_")   # ... and we make sure the key is written in proper format
            break                                                                                       # ... and we stop iterating over the rest of the class_names
    return key                                                                                          # Return the new key name


# Create a function to extract the list of lists containing the keys that are relevant to show
def extractRelevantHistoryKeys(history):
    loss_total = [key for key in history.keys() if "total_loss" in key.lower()]                         # Find all keys with loss_ce
    m_fw_IoU = [key for key in history.keys() if key.endswith("IoU")]                                   # Find the mIoU and fIoU keys
    pq_rq_sq = [key for key in history.keys() if any([x in key for x in ["_RQ", "_SQ", "_PQ"]]) and not any([x in key for x in ["_RQ_C", "_SQ_C", "_PQ_C"]])]   # Find all the [rq, sq, pq] keys
    mACC_pACC = [key for key in history.keys() if key.endswith("ACC")]                                  # Find the keys with the pixel accuracy
    loss_ce = [key for key in history.keys() if "loss_ce" in key.lower() and key.endswith("ce")]        # Find all keys with loss_ce
    loss_dice = [key for key in history.keys() if "loss_dice" in key.lower() and key.endswith("dice")]  # Find all keys with loss_dice
    loss_mask = [key for key in history.keys() if "loss_mask" in key.lower() and key.endswith("mask")]  # Find all keys with loss_mask
    IoU_per_class = [key for key in history.keys() if "_IoU_" in key and not key.endswith("IoU")]       # Find the class specific IoU keys
    pACC_per_class = [key for key in history.keys() if "_ACC_" in key and not key.endswith("ACC")]      # Find the class specific pixel accuracies
    PQ_per_class = [key for key in history.keys() if "_PQ_" in key and not key.endswith("PQ")]          # Extract all keys with the per_class PQ
    RQ_per_class = [key for key in history.keys() if "_RQ_" in key and not key.endswith("RQ")]          # Extract all keys with the per_class RQ
    SQ_per_class = [key for key in history.keys() if "_SQ_" in key and not key.endswith("SQ")]          # Extract all keys with the per_class SQ
    learn_rate = [key for key in history.keys() if "lr" in key.lower() and "val" not in key.lower()]    # Find the training learning rate
    hist_keys_list = [loss_total, m_fw_IoU, pq_rq_sq, mACC_pACC, loss_ce, loss_dice, loss_mask,         # Combine the key-lists into a list of ...
                learn_rate, pACC_per_class, PQ_per_class, RQ_per_class, SQ_per_class, IoU_per_class]    # lists containing all relevant keys
    return hist_keys_list


# Create a function to create the history dictionary
def combineDataToHistoryDictionaryFunc(config, eval_metrics=None, pq_metrics=None, data_split="train", history=None, json_metrics=None):
    if history == None: history = {}                                                                    # Initiate the history dictionary that will be used
    if "train" in data_split: json_metrics = load_json_metrics(config=config, data_split="train")       # Load the metrics into the history dictionary
    if "val" in data_split: json_metrics = load_json_metrics(config=config, data_split="val")           # Load the metrics into the history dictionary
    if json_metrics is not None:                                                                        # If the json metrics has been loaded ...
        for key in json_metrics: history[data_split+"_"+key] = json_metrics[key]                        # ... append all the metrics losses to the dictionary with the split prefix on the key
    if eval_metrics is not None:                                                                        # If any evaluation metrics are available
        for key in eval_metrics.keys():                                                                 # Iterate over all keys in the history
            old_key = deepcopy(key)                                                                     # Make a copy now of the original key name from the metrics_train/eval dictionary
            key = changeClassNameForClassIdxFunc(key=key, config=config)                                # Exchange class_name with class_label in the key-name
            if data_split+"_"+key not in history: history[data_split+"_"+key] = list()                  # If the given key doesn't exist add the key with ...
            history[data_split+"_"+key].append(eval_metrics[old_key])                                   # Append the current key-value from the metrics_train
    if pq_metrics is not None:                                                                          # If any panoptic quality metrics are available ...
        for master_key in pq_metrics.keys():                                                            # Iterate over all keys in the PQ results ...
            if "all" in master_key.lower():                                                             # Available master_keys are ['All', 'per_class', 'Stuff']. All and Stuff are identical (at least AFAIK)
                for key in pq_metrics[master_key].keys():                                               # ... each element in the PQ_results is a new dictionary
                    if "n" in key.lower(): continue                                                     # If the key in the new dictionary is just the 'n' (number of classes), then skip that one
                    if data_split+"_"+key.upper() not in history: history[data_split+"_"+key.upper()] = list()  # If the given key doesn't exist add the key with ...
                    history[data_split+"_"+key.upper()].append(pq_metrics[master_key][key]*100)         # Append the current key-value from the metrics_train
            if "per_class" in master_key.lower():                                                       # If we are looking at the per_class panoptic scores ...
                for class_key in pq_metrics[master_key].keys():                                         # ... we'll iterate over all classes
                    for key in pq_metrics[master_key][class_key]:                                       # Loop through all the class_variables of PQ, SQ, RQ
                        if data_split+"_{:s}_C{:d}".format(key.upper(), class_key) not in history: history[data_split+"_{:s}_C{:d}".format(key.upper(), class_key)] = list()
                        history[data_split+"_{:s}_C{:d}".format(key.upper(), class_key)].append(pq_metrics[master_key][class_key][key]*100)
    new_history = {key: history[key] for key in history.keys() if all(np.isnan(history[key]))==False and "_C{:d}".format(MetadataCatalog[config.DATASETS.TRAIN[0]].ignore_label) not in key}
    return new_history


# Function to display learning curves
def show_history(config, FLAGS, metrics_train, metrics_eval, pq_train, pq_val, history=None):           # Define a function to visualize the learning curves
    """"
    Mean intersection-over-union averaged across classes (mIoU)
    Frequency Weighted IoU (fwIoU)
    Mean pixel accuracy averaged across classes (mACC)
    Pixel Accuracy (pACC)
    """
    # Create history and list of relevant history keys
    if FLAGS.inference_only==False or FLAGS.hp_optim==False:
        history = combineDataToHistoryDictionaryFunc(config=config, eval_metrics=metrics_train, pq_metrics=pq_train, data_split="train", history=history)
    history = combineDataToHistoryDictionaryFunc(config=config, eval_metrics=metrics_eval, pq_metrics=pq_val, data_split="val", history=history)
    hist_keys = extractRelevantHistoryKeys(history)
    ax_titles = ["Total_loss", "mIoU and fwIoU", "PQ, RQ, SQ", "mACC and pACC", "Loss_CE", "Loss_DICE", "Loss_Mask",        # Create titles and ylabels ...
            "Learning_rate", "Pixel_accuracy_per_class", "PQ_per_class", "RQ_per_class", "SQ_per_class", "IoU_per_class"]   # ... for the axes
    colors = ["blue", "red", "black", "green", "magenta", "cyan", "yellow", "deeppink", "purple",       # Create colors for ... 
                "peru", "darkgrey", "gold", "springgreen", "orange", "crimson", "lawngreen"]            # ... the line plots
    n_rows, n_cols, ax_count = 3, (3,5,5), 0                                                            # Initiate values for the number of rows and columns
    if FLAGS.use_per_pixel_baseline==True:                                                              # If we train with the per_pixel_classification method ...
        n_rows, n_cols = 3, tuple(np.subtract(n_cols,(0,2,1)))                                          # ... the number of rows and columns gets reduced
        ax_tuple = [(ii,x) for (ii,x) in enumerate(ax_titles) if not x.lower().startswith("loss")]      # Get a list of tuples of ax_titles and their indexes to keep
        ax_titles = [x[1] for x in ax_tuple]                                                            # Get the new list of kept ax_titles
        indices = [x[0] for x in ax_tuple]                                                              # Get the indices of the accepted ax_titles
        hist_keys = np.asarray(hist_keys, dtype=object)[indices].tolist()                               # Get the new list of kept metrics to visualize
    if FLAGS.num_classes > 10:                                                                          # If there are more than 10 classes (i.e. for ADE20K_dataset) ...
        n_rows, n_cols = 3, tuple(np.subtract(n_cols,(1,2,2)))                                          # ... the number of rows and columns gets reduced ...
        ax_tuple = [(ii,x) for (ii,x) in enumerate(ax_titles) if "per_class" not in x]                  # ... remove the ax_titles with "per class", as we can't visualize that many classes simultaneously
        ax_titles = [x[1] for x in ax_tuple]                                                            # Get the new list of kept ax_titles
        indices = [x[0] for x in ax_tuple]                                                              # Get the indices of the accepted ax_titles
        hist_keys = np.asarray(hist_keys, dtype=object)[indices].tolist()                               # Get the new list of kept metrics to visualize
    if FLAGS.num_classes > 10 and FLAGS.use_per_pixel_baseline==True: n_rows, n_cols = 2, (2,3)         # If both Loss_CE, Loss_DICE, Loss_mask and all per_pixel classes should be removed ...
    
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
                    y_top_val = np.ceil(np.max(history[key])/10)*10                                     # ... y_top_val is updated and rounded to the nearest 10
                plt.plot(np.linspace(start=np.min(history["val_epoch_num"])-(0 if any([x in key for x in ["ACC", "IoU", "PQ", "RQ", "SQ"]]) else 1),    # Plot the data, using a linspace ...
                    stop=np.max(history["val_epoch_num"]), num=len(history[key])), history[key], color=colors[kk], linestyle="-", marker=".")   # ... argument for the x-axis and the data itself for the y-axis
            plt.legend(sorted([key for key in hist_keys[ax_count]], key=str.lower),                     # Create a legend for the subplot with ...
                    framealpha=0.35, loc="best" if len(hist_keys[ax_count])<4 else "upper left")        # ... the history keys displayed
            ax_count += 1                                                                               # Increase the subplot counter
        if y_top_val <= 0.05 and "lr" not in key.lower(): plt.ylim(bottom=-0.05, top=0.05)              # If the max y-value is super low, the limits are changed
        else: plt.ylim(bottom=0, top=y_top_val)                                                         # Set the final, updated y_top_value as the y-top-limit on the current subplot axes
        if "lr" in key.lower():                                                                         # If we are plotting the learning rate ...
            plt.ylim(bottom=np.min(history[key])*0.9, top=np.max(history[key])*1.075)                   # ... the y_limits are changed
            plt.yscale('log')                                                                           # ... the y_scale will be logarithmic
    try: fig.savefig(os.path.join(config.OUTPUT_DIR, "Learning_curves.jpg"), bbox_inches="tight")       # Try and save the figure in the OUTPUR_DIR ...
    except: pass                                                                                        # ... otherwise simply skip saving the figure
    fig.show() if FLAGS.display_images==True else plt.close(fig)                                        # If the user chose to not display the figure, the figure is closed
    return history                                                                                      # The history dictionary is returned


# config=cfg
# metrics_train=eval_train_results["sem_seg"]
# metrics_eval=eval_val_results["sem_seg"]
# pq_train=train_pq_results
# pq_val=val_pq_results
# history = None
# history = show_history(config=cfg, FLAGS=FLAGS, metrics_train=eval_train_results["sem_seg"], metrics_eval=eval_val_results["sem_seg"], pq_train=train_pq_results, pq_val=val_pq_results, history=None)


