import os
import re
import torch
import torch.nn as nn
import numpy as np
# import matplotlib
# matplotlib.use("pdf")
from matplotlib import pyplot as plt
from tqdm import tqdm                                                                       # Used to set a progress bar
from custom_register_vitrolife_dataset import vitrolife_dataset_function
from detectron2.data import DatasetCatalog, MetadataCatalog, DatasetMapper, build_detection_train_loader
from detectron2.engine.defaults import DefaultPredictor

# from custom_goto_trainer_class import custom_augmentation_mapper                            # A function that returns a custom mapper using data augmentation


# Move the figure to the wanted position when displaying
try:
    import pyautogui
    def move_figure_position(fig=plt.figure(), screensize=list(pyautogui.size()),           # Define a function to move a figure ...
                            dpi=100, position=[0.10, 0.09, 0.80, 0.75]):                    # ... to a specified position on the screen
        fig = plt.figure(fig)                                                               # Make the wanted figure the current figure again
        # screensize[1] = np.round(np.divide(screensize[1], 1.075))                         # Reduce height resolution as the processbar in the bottom is part of the screen size
        screensize_inches = np.divide(screensize,dpi)                                       # Convert the screensize into inches
        fig.set_figheight(position[3] * screensize_inches[1])                               # Set the wanted height of the figure
        fig.set_figwidth(position[2] * screensize_inches[0])                                # Set the wanted width of the figure
        figManager = plt.get_current_fig_manager()                                          # Get the current manager (i.e. window execution commands) of the current figure
        upper_left_corner_position = "+{:.0f}+{:.0f}".format(                               # Define a string with the upper left corner coordinates ...
            screensize[0]*position[0], screensize[1]*position[1])                           # ... which are read from the position inputs
        figManager.window.wm_geometry(upper_left_corner_position)                           # Move the figure to the upper left corner coordinates
        return fig                                                                          # Return the figure handle
except: pass


# Define function to apply a colormap on the images
def apply_colormap(mask, config):
    colors_used = list(MetadataCatalog[config.DATASETS.TEST[0]].thing_colors)               # Read the colors used in the Metadatacatalog. If no colors are assigned, random colors are used
    if "vitrolife" in config.DATASETS.TEST[0].lower():                                      # If we are working on the vitrolife dataset ...
        labels_used = MetadataCatalog[config.DATASETS.TEST[0]].thing_dataset_id_to_contiguous_id    # ... labels_used will be read from the MetadataCatalog
    else: labels_used = list(range(len(MetadataCatalog["ade20k_instance_train"].thing_classes)))    # Else, labels is just 0:num_classes-1
    color_array = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)               # Allocate a RGB 3D array of zeros
    for label_idx, label in enumerate(labels_used):                                         # Loop through each label from the labels_used found from the MetadataCatalog
        color_array[mask == label] = colors_used[label_idx]                                 # Assign all pixels in the mask with the current label_value the colors_used[idx] value
    return color_array                                                                      # Return the colored mask


# Define a function to extract numbers from a string
def extractNumbersFromString(str, dtype=float, numbersWanted=1):
    try: vals = dtype(str)                                                                  # At first, simply try to convert the string into the wanted dtype
    except:                                                                                 # Else, if that is not possible ...
        vals = [float(s) for s in re.findall(r"[-+]?\d*\.\d+|\d+", str)]                    # Extract all the numbers from the string and put them in a list
        if len(vals) > 0:                                                                   # If any numbers is found ...
            for kk in range(len(vals)):                                                     # Loop through all the found numbers
                vals[kk] = dtype(vals[kk])                                                  # Convert each of the found numbers into the wanted dtype
                if kk+1 == numbersWanted: break                                             # If we have convert all the numbers wanted, we'll stop the loop
            if numbersWanted < len(vals): vals = vals[:numbersWanted]                       # Then we'll only use up to 'numbersWanted' found numbers
            if numbersWanted==1: vals = vals[0]                                             # If we only want 1 number, then we'll extract that from the list
        else: vals = np.nan                                                                 # ... else if no numbers were found, return NaN
    return vals                                                                             # Return the wanted numbers, either as a type 'dtype' or, if multiple numbers, a list of 'dtypes'


