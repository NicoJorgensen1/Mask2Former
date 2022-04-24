import os
import re
import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm                                                                       # Used to set a progress bar
from copy import deepcopy
from custom_register_vitrolife_dataset import vitrolife_dataset_function
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from mask2former import MaskFormerInstanceDatasetMapper
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode, _create_text_labels, GenericMask
from detectron2.modeling import build_model
from mask2former.modeling.matcher import HungarianMatcher



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


# Define a function to put the latest saved model as the model_weights in the config before creating the dataloader
def putModelWeights(config, delete_remaining=False):
    model_files = [x for x in os.listdir(config.OUTPUT_DIR) if not np.isnan(extractNumbersFromString(x)) and all([y in x for y in ["model", "_epoch_", ".pth"]])]   # Find all saved model checkpoints
    if len(model_files) >= 1:                                                               # If any model checkpoint is found, 
        iteration_numbers = [extractNumbersFromString(x, int) for x in model_files]         # Find the iteration numbers for when they were saved
        latest_iteration_idx = np.argmax(iteration_numbers)                                 # Find the index of the model checkpoint with the latest iteration number
        config.MODEL.WEIGHTS = os.path.join(config.OUTPUT_DIR, model_files[latest_iteration_idx])   # Assign the latest model checkpoint to the config
        if delete_remaining==True:                                                          # If the user chose to delete all other but the final model, ... 
            for model_file in model_files:                                                  # ... loop through all found model checkpoint files
                if os.path.join(config.OUTPUT_DIR,model_file) != config.MODEL.WEIGHTS:      # If the current model_file is not the newest checkpoint file ...
                    os.remove(os.path.join(config.OUTPUT_DIR,model_file))                   # ... remove the current model_file
    return config                                                                           # Return the updated config


# Function to create directories in which the visualization results are saved
def get_save_dirs(config, dataset_split):
    for data_split in ["train", "val", "test"]:
        if "vitrolife" not in config.DATASETS.TRAIN[0].lower() and data_split=="test": continue     # There are no test dataset for the ade20k dataset 
        os.makedirs(os.path.join(config.OUTPUT_DIR, "Visualizations", data_split), exist_ok=True)   # Create a folder to store the segmentations of the images
    return os.path.join(config.OUTPUT_DIR, "Visualizations", dataset_split)                         # Return the folder name of the current dataset split


# Function to sort the dictionaries by the number of PN's found
def sort_dictionary_by_PN(data):
    PNs_idx = np.argsort(data["PN"])
    new_data = {}
    for key in data:
        new_data[key] = [data[key][x] for x in PNs_idx]
    return new_data


# Define a function to predict some label-masks for the dataset
def create_batch_img_ytrue_ypred(config, data_split, FLAGS, data_batch=None, model_done_training=False):
    if model_done_training==False: config = putModelWeights(config)                         # Change the config and append the latest model as the used checkpoint, if the model is still training
    predictor = DefaultPredictor(cfg=config)                                                # Create a default predictor instance
    if data_batch == None:                                                                  # If no batch with data was send to the function ...
        if "vitrolife" in FLAGS.dataset_name.lower():                                       # ... and if we are using the vitrolife dataset
            dataset_dicts = vitrolife_dataset_function(data_split, debugging=True)          # ... the list of dataset_dicts from vitrolife is computed.
            dataset_dicts = dataset_dicts[:FLAGS.num_images]                                # We'll maximally show the first FLAGS.num_images images
        else: dataset_dicts = DatasetCatalog.get("ade20k_sem_seg_{:s}".format(data_split))  # Else we use the ADE20K dataset
        if "train" in data_split:
            augmentations = []
        else:
            augmentations = []
        data_mapper = MaskFormerInstanceDatasetMapper(cfg=config, is_train=True, augmentations=augmentations)   # Use the standard instance segmentation mapper 
        dataloader = build_detection_train_loader(dataset_dicts, mapper=data_mapper, total_batch_size=np.min([FLAGS.num_images, len(dataset_dicts)]))   # Create the dataloader
        data_batch = next(iter(dataloader))                                                 # Extract the next batch from the dataloader
    dataset_name = config.DATASETS.TRAIN[0] if "train" in data_split else config.DATASETS.TEST[0]
    if "train" in data_split: meta_data = MetadataCatalog.get(dataset_name)
    elif any([data_split in x for x in["val", "test"]]): meta_data = MetadataCatalog.get(dataset_name)



    # Perhaps take a look at these three files here:
    # /mnt/c/Users/Nico-/Documents/Python_Projects/Mask2Former/mask2former/modeling/matcher.py
    # /mnt/c/Users/Nico-/Documents/Python_Projects/Mask2Former/mask2former/maskformer_model.py
    # /mnt/c/Users/Nico-/Documents/Python_Projects/Mask2Former/mask2former/modeling/criterion.py



    matcher = HungarianMatcher(cost_class=FLAGS.class_loss_weight, cost_dice=FLAGS.dice_loss_weight,
            cost_mask=FLAGS.mask_loss_weight, num_points=config.MODEL.MASK_FORMER.TRAIN_NUM_POINTS)
    model = build_model(cfg=config)

    img_ytrue_ypred = {"input": list(), "y_pred": list(), "y_true": list(), "PN": list()}   # Initiate a dictionary to store the input images, ground truth masks and the predicted masks
    class_colors = meta_data.thing_colors                                                   # Get the colors that the classes must be visualized with 
    class_names = deepcopy(meta_data.thing_classes)                                         # Read the class names present in the dataset
    for data in data_batch:                                                                 # Iterate over each data sample in the batch from the dataloade
        # The input image
        img = torch.permute(data["image"], (1,2,0)).numpy()                                 # Make input image numpy format and [H,W,C]
        
        # The ground truth prediction image 
        # Make this into a function that accepts a list of class labels and masks and boxes (should allow for boxes=None)
        true_classes = data["instances"].get_fields()["gt_classes"]                         # Get the true class labels for the instances on the current image
        true_masks = data["instances"].get_fields()["gt_masks"]                             # Get the true binary masks for the instances on the current image
        y_true = np.zeros(shape=tuple(true_masks.shape[-2:])+(3,), dtype=np.uint8)          # Initiate a colored image to show the true masks
        PN_count = 0                                                                        # Make a counter to keep track of the PN's in the current image
        for true_class, true_mask in zip(true_classes, true_masks):                         # Iterate through the true objects in the current image
            true_class = true_class.numpy().item()                                          # Extract the current objects true class label as a scalar 
            class_name = class_names[true_class]                                            # Find the class label of the current object 
            col_idx = np.where(np.in1d(class_names, class_name))[0].item() + PN_count       # Compute which color the current object will have in the final image 
            col = class_colors[col_idx]                                                     # Extract the thing color needed for the current object
            if class_name == "PN":                                                          # If the current object is a PN ...
                PN_count += 1                                                               # ... increase the PN counter
            true_mask = true_mask.numpy()                                                   # Convert the true binary mask from a torch tensor to a numpy tensor 
            y_true[true_mask] = col                                                         # Assign all pixels from the current object with the specified pixel color value 
            bbox_coordinates = np.asarray(np.where(true_mask))                              # Get all pixel coordinates for the true pixels in the mask
            x1, y1 = np.amin(bbox_coordinates, axis=1)                                      # Extract the minimum x and y true pixel values
            x2, y2 = np.amax(bbox_coordinates, axis=1)                                      # Extract the maximum x and y true pixel values
            y_true = cv2.rectangle(y_true, (y1,x1), (y2,x2),col, 2)                         # Overlay the bounding box for the current object on the current image 
        # plt.imshow(y_true)
        # plt.title("Image with {} PN".format(data["image_custom_info"]["PN_image"]))
        # plt.show(block=False)

        # The predicted image 
        y_pred = predictor.__call__(img)                                                    # y_pred is a dict with keys "instances"
        y_pred_2 = y_pred["instances"]                                                      # y_pred_2 is a detectron2.structures.instances.Instances object
        y_pred_3 = y_pred_2.get_fields()                                                    # y_pred_3 is a dict with keys ['pred_masks', 'pred_boxes', 'scores', 'pred_classes']
        pred_masks = y_pred_3["pred_masks"]                                                 # pred_masks is a tensor of shape [100, H, W] of float values
        pred_boxes = y_pred_3["pred_boxes"].tensor                                          # pred_boxes is a tensor of shape [100, 4] containing float values 
        pred_scores = y_pred_3["scores"]                                                    # pred_scores is a tensor of shape [100] containing 0<score<1 float values
        pred_classes = y_pred_3["pred_classes"]                                             # pred_classes is a tensor of shape [100] only containing integer class_id's

        ### DO SOMETHING WITH THE HUNGARIAN MATCHING CLASS
        ### CREATE MY OWN INSTANCE AND CHANGE FAST_FORWARD FUNCTION TO ACCEPT THESE OUTPUTS
        ### THEN THE MATCHING WILL TELL WHICH OF THE PREDICTIONS SHOULD BE USED
        ### WHEN WE KNOW WHICH PREDICTIONS TO USE, WE CAN CREATE AN IMAGE SIMILAR TO THE Y_TRUE IMAGE 


        # visualizer = Visualizer(img[:, :, ::-1], metadata=meta_data, scale=1)        
        # y_pred = predictor.__call__(img)
        # y_pred_labeled = visualizer.draw_instance_predictions(data["instances"].to("cpu"))
        # y_pred_col = y_pred_labeled.get_image()[:, :, ::-1]


        # Append the input image, y_true and y_pred to the dictionary
        img_ytrue_ypred["input"].append(img)                                                # Append the input image to the dictionary
        img_ytrue_ypred["y_true"].append(y_true)                                            # Append the ground truth to the dictionary
        img_ytrue_ypred["y_pred"].append(img)                               # Append the predicted mask to the dictionary
        if "vitrolife" in FLAGS.dataset_name.lower():                                       # If we are visualizing the vitrolife dataset
            img_ytrue_ypred["PN"].append(int(data["image_custom_info"]["PN_image"]))        # Read the true number of PN on the current image
    return img_ytrue_ypred, data_batch, FLAGS, config


# position=[0.55, 0.08, 0.40, 0.75]
# epoch_num = None
# data_batches = None
# data_batch=None
# model_done_training = False 
# data_split = "train"

# Define function to plot the images
def visualize_the_images(config, FLAGS, position=[0.55, 0.08, 0.40, 0.75], epoch_num=None, data_batches=None, model_done_training=False):
    # Get the datasplit and number of images to show
    fig_list, data_batches_final = list(), list()                                           # Initiate the list to store the figures in
    if data_batches==None:                                                                  # If no previous data has been sent ...
        data_batches = [None, None, None]                                                   # ... it must be a list of None's...
    data_split_count = 1                                                                    # Initiate the datasplit counter
    fontdict = {'fontsize': 25}                                                             # Set the font size for the plot
    for data_split, data_batch in tqdm(zip(["train", "val", "test"], data_batches),         # Iterate through the three splits available
                unit="Data_split", ascii=True, desc="Dataset split {:d}/{:d}".format(data_split_count, 3),
                bar_format="{desc}  | {percentage:3.0f}% | {bar:35}| {n_fmt}/{total_fmt} [Spent: {elapsed}. Remaining: {remaining}{postfix}]"):      
        data_split_count += 1
        if "vitrolife" not in FLAGS.dataset_name.lower() and data_split=="test": continue   # Only vitrolife has a test dataset. ADE20K doesn't. 
        # Extract information about the dataset used
        img_ytrue_ypred, data_batch, FLAGS, config = create_batch_img_ytrue_ypred(config=config,    # Create the batch of images that needs to be visualized ...
            data_split=data_split, FLAGS=FLAGS, data_batch=data_batch, model_done_training=model_done_training) # ... and return the images in the data_batch dictionary
        if "vitrolife" in FLAGS.dataset_name.lower():                                       # If we are working on the vitrolife dataset sort the ...
            data_batch = sorted(data_batch, key=lambda x: x["image_custom_info"]["PN_image"])   # ... data_batch after the number of PN per found image
            img_ytrue_ypred = sort_dictionary_by_PN(data=img_ytrue_ypred)                   # And then also sort the data dictionary
        num_rows, num_cols = 3, len(data_batch)                                             # The figure will have three rows (input, y_pred, y_true) and one column per image
        fig = plt.figure(figsize=(int(np.ceil(len(data_batch)*4)), 12))                     # Create the figure object
        row = 0                                                                             # Initiate the row index counter (all manual indexing could have been avoided by having created img_ytrue_ypred as an OrderedDict)
        for key in img_ytrue_ypred.keys():                                                  # Loop through all the keys in the batch dictionary
            if key.lower() not in ['input', 'y_true', 'y_pred']: continue                   # If the key is not one of (input, y_pred, y_true), we simply skip to the next one
            for col, img in enumerate(img_ytrue_ypred[key]):                                # Loop through all available images in the dictionary
                plt.subplot(num_rows, num_cols, row*num_cols+col+1)                         # Create the subplot instance
                plt.axis("off")                                                             # Remove axis tickers
                if "vitrolife" in FLAGS.dataset_name.lower():                               # If we are visualizing the vitrolife dataset
                    plt.title("{:s} with {:.0f} PN".format(key, img_ytrue_ypred["PN"][col]), fontdict=fontdict) # Create the title for the plot with the number of PN
                else: plt.title("{:s}".format(key), fontdict=fontdict)                      # Otherwise simply put the key, i.e. either input, y_pred or y_true.
                plt.imshow(img, cmap="gray")                                                # Display the image
            row += 1                                                                        # Increase the row counter by 1
        try: fig = move_figure_position(fig=fig, position=position)                         # Try and move the figure to the wanted position (only possible on home computer with a display)
        except: pass                                                                        # Except, simply just let the figure retain the current position
        fig_name_init = "Segmented_{:s}_data_samples_from_".format(data_split)              # Initialize the figure name
        if epoch_num is not None: fig_name = "{:s}epoch_{:d}.jpg".format(fig_name_init, epoch_num)                      # If an epoch number has been specified, the figure name will contain that
        else: fig_name = "{:s}{:s}_training.jpg".format(fig_name_init, "after" if model_done_training else "before")    # Otherwise the visualization happens before/after training
        fig.tight_layout()                                                                  # Assures the subplots are plotted tight around each other
        fig.savefig(os.path.join(get_save_dirs(config=config, dataset_split=data_split), fig_name), bbox_inches="tight")    # Save the figure in the correct output directory
        fig_list.append(fig)                                                                # Append the current figure to the list of figures
        data_batches_final.append(data_batch)                                               # Append the current data_batch to the list of data_batches
        fig.show() if FLAGS.display_images==True else plt.close(fig)                        # Display the figure if that is the chosen option
    return fig_list, data_batches_final, config, FLAGS                                      # Return the figure, the dictionary with the used images, the config and the FLAGS arguments

