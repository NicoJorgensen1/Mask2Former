import os
import re
import torch
import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm                                                                       # Used to set a progress bar
from copy import deepcopy
from custom_register_vitrolife_dataset import vitrolife_dataset_function
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.engine.defaults import DefaultPredictor
from mask2former.modeling.matcher import HungarianMatcher
from custom_print_and_log_func import printAndLog                                           # Function to log results
from custom_Trainer_class import custom_augmentation_mapper as custom_mapper                # A function that returns a custom mapper using data augmentation


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


# Function to create an image from a list of masks and labels
def draw_mask_image(mask_list, lbl_list, meta_data):
    class_colors = meta_data.thing_colors                                                   # Get the colors that the classes must be visualized with 
    class_names = deepcopy(meta_data.thing_classes)                                         # Read the class names present in the dataset
    final_im = np.zeros(shape=mask_list[0].shape+(3,), dtype=np.uint8)                      # Initiate a colored image to show the masks as a single image 
    PN_count = 0                                                                            # Make a counter to keep track of the PN's in the current image
    for lbl, mask in zip(lbl_list, mask_list):                                              # Iterate through the labels and masks
        class_name = class_names[lbl]                                                       # Find the class name of the current object 
        col_idx = np.where(np.in1d(class_names, class_name))[0].item() + PN_count           # Compute which color the current object will have in the final image 
        col = class_colors[col_idx]                                                         # Extract the thing color needed for the current object
        if len(np.where(mask)[0]) < 4: continue                                             # If less than four points are positive in the current mask, skip this mask, as we then can't draw a bounding box
        if class_name == "PN":                                                              # If the current object is a PN ...
            PN_count += 1                                                                   # ... increase the PN counter
        final_im[mask.astype(bool)] = col                                                   # Assign all pixels from the current object with the specified pixel color value 
        bbox_coordinates = np.asarray(np.where(mask))                                       # Get all pixel coordinates for the white pixels in the mask
        x1, y1 = np.amin(bbox_coordinates, axis=1)                                          # Extract the minimum x and y white pixel values
        x2, y2 = np.amax(bbox_coordinates, axis=1)                                          # Extract the maximum x and y white pixel values
        final_im = cv2.rectangle(final_im, (y1,x1), (y2,x2),col, 2)                         # Overlay the bounding box for the current object on the current image 
    # plt.imshow(final_im)
    # plt.show(block=False)
    return final_im                                                                         # Return the final image 


# Function to perform the Hungarian matching between true masks and predicted masks
def hungarian_matching(y_pred_dict, data, predictor, matcher, FLAGS, meta_data):
    img = torch.permute(data["image"], (1,2,0)).numpy()                                     # Read the true image as a numpy array from the data dictionary 
    true_classes = data["instances"].get_fields()["gt_classes"].numpy().tolist()            # Read the true classes from the data dictionary
    pred_masks = y_pred_dict["pred_masks"]                                                  # pred_masks is a tensor of shape [Q, H, W] of float values
    if len(true_classes) < 1: y_pred = np.zeros_like(img)                                   # If no true objects are present on the image, no matching can be made ...
    else:                                                                                   # Else, if any ground truth object is present on the original image ...
        img_torch = torch.reshape(data["image"], (1,)+tuple(data["image"].shape))           # Reshape the torch-type image into shape [1, C, H, W]
        features = predictor.model.backbone(img_torch.to(predictor.model.device).to(torch.float))   # Compute the image features using the backbone model
        outputs = predictor.model.sem_seg_head(features)                                    # Compute the outputs => a dictionary with keys [pred_logits, pred_masks, aux_outputs].
        targets = [{"labels": data["instances"].get_fields()["gt_classes"],                 # The target must be a list of length batch_size containing ...
                    "masks": data["instances"].get_fields()["gt_masks"]}]                   # ... dicts with keys "labels" and "masks" for each image
        matched_output = matcher.forward(outputs=outputs, targets=targets)[0]               # Performs the Hungarian matching and outputs a list of [[idx_row], [idx_col]] ...
        row_pred_indices = matched_output[0].numpy()                                        # This is the row_idx of the matching, i.e. the index of the "chosen" predicted mask and lbl
        col_pred_indices = matched_output[1].numpy()                                        # This is the col_idx of the matching, i.e. the index of the "chosen" true instance label idx
        assert pred_masks.shape[0] == FLAGS.num_queries == outputs["pred_masks"].shape[1], "This fucking has to work"
        y_pred_masks = [x for x in pred_masks[row_pred_indices].numpy().astype(bool)]       # This is the matced predicted masks
        y_pred_lbls = np.asarray(true_classes)[col_pred_indices].tolist()                   # This is the matched predicted class labels for each of the predicted masks
        y_pred = draw_mask_image(mask_list=y_pred_masks, lbl_list=y_pred_lbls, meta_data=meta_data) # Create a mask image for the true masks
    return y_pred


# Function to compute non max suppression for the predicted object instances 
def NMS_pred(y_pred_dict, data, meta_data, conf_thresh, IoU_thresh):
    # Read the image and initiate an empty prediction
    img = torch.permute(data["image"], (1,2,0)).cpu().numpy()                               # Read the true image as a numpy array from the data dictionary 
    y_pred_mask = np.zeros_like(img)                                                        # Initiate the 
    
    # Get the predicted classes, scores and masks
    pred_classes = y_pred_dict["pred_classes"].cpu().numpy()                                # pred_classes is an array of shape [Q] only containing integer class_id's. For inference only, when no true labels are available
    pred_scores = y_pred_dict["scores"].cpu().numpy()                                       # pred_scores  is an array of shape [Q] containing the class confidence scores from the predictions 
    pred_masks = y_pred_dict["pred_masks"].cpu().numpy()                                    # pred_masks is an array of shape [Q, H, W] of float values

    # Filter out all masks with a confidence score below threshold
    thres_idx = (pred_scores >= conf_thresh)
    used_masks, used_lbls = list(), list()
    if thres_idx.sum() >= 1:
        pred_classes = pred_classes[thres_idx]
        pred_scores = pred_scores[thres_idx]
        pred_masks = pred_masks[thres_idx]

        # Sort all predictions by their confidence score
        conf_idx = np.argsort(pred_scores)[::-1]
        pred_classes = pred_classes[conf_idx]
        pred_scores = pred_scores[conf_idx]
        pred_masks = pred_masks[conf_idx]

        # Iterate through all predictions 
        for mask, lbl in zip(pred_masks, pred_classes):
            # Append the mask and lbl with the highest confidence scores among the remaining predictions
            IoU_with_used = list()
            if len(used_masks) >= 1:
                for used_mask in used_masks:
                    intersection_ = np.logical_and(mask, used_mask).sum()
                    union_ = np.logical_or(mask, used_mask).sum()
                    IoU_with_used.append(np.divide(intersection_, union_))
                if any(np.greater_equal(IoU_with_used, IoU_thresh)): continue
            used_masks.append(mask)
            used_lbls.append(lbl)
        y_pred_mask = draw_mask_image(mask_list=used_masks, lbl_list=used_lbls, meta_data=meta_data)
    return y_pred_mask, used_masks 


# Define a function to predict some label-masks for the dataset
def create_batch_img_ytrue_ypred(config, data_split, FLAGS, data_batch=None, model_done_training=False, device="cpu", matching_type="Hungarian"):
    if model_done_training==False:                                                          # If the model hasn't finished training ...
        config = putModelWeights(config)                                                    # ... change the config and append the latest model as the used checkpoint, if the model is still training
    if data_batch is None:                                                                  # If no batch with data was send to the function ...
        if "vitrolife" in FLAGS.dataset_name.lower():                                       # ... and if we are using the vitrolife dataset
            dataset_dicts = vitrolife_dataset_function(data_split, debugging=True)          # ... the list of dataset_dicts from vitrolife is computed.
            dataset_dicts = dataset_dicts[:FLAGS.num_images]                                # We'll maximally show the first FLAGS.num_images images
        else:                                                                               # Else ...
            dataset_dicts = DatasetCatalog.get("ade20k_sem_seg_{:s}".format(data_split))    # ... we use the ADE20K dataset
        data_mapper = custom_mapper(config=config, is_train="train" in data_split)          # Using my own custom data mapper, only use data augmentation on training dataset 
        dataloader = build_detection_train_loader(dataset_dicts, mapper=data_mapper, total_batch_size=np.min([FLAGS.num_images, len(dataset_dicts)]), num_workers=1)   # Create the dataloader
        data_batch = next(iter(dataloader))                                                 # Extract the next batch from the dataloader
    dataset_name = config.DATASETS.TRAIN[0] if "train" in data_split else config.DATASETS.TEST[0]   # Extract the dataset name 
    meta_data = MetadataCatalog.get(dataset_name)                                           # Read the metadata for the current image 
    
    # Create the object instances used for performing the predictions 
    config_prediction = deepcopy(config)                                                    # Create a copy of the config
    config_prediction.TEST.DETECTIONS_PER_IMAGE = FLAGS.num_queries                         # Set detections done at checktime to the num_queries (i.e. detections at test time is the same as now for training)
    config_prediction.MODEL.DEVICE = device                                                 # Set the model to perform this on the chosen device. We script it to start trying on cuda, if OOM, then cpu 
    predictor = DefaultPredictor(config_prediction)                                         # Create an instance of the DefaultPredictor with the changed config 
    matcher = HungarianMatcher(cost_class=FLAGS.class_loss_weight, cost_dice=FLAGS.dice_loss_weight,    # Create an instance of the ...
            cost_mask=FLAGS.mask_loss_weight, num_points=config.MODEL.MASK_FORMER.TRAIN_NUM_POINTS)     # ... Hungarian Matcher class
    img_ytrue_ypred = {"input": list(), "y_pred": list(), "y_true": list(), "PN": list()}   # Initiate a dictionary to store the input images, ground truth masks and the predicted masks
    for data in data_batch:                                                                 # Iterate over each data sample in the batch from the dataloade
        img = torch.permute(data["image"], (1,2,0)).numpy()                                 # Make input image numpy format and [H,W,C]
        
        # The ground truth prediction image 
        true_classes = data["instances"].get_fields()["gt_classes"].numpy().tolist()        # Get the true class labels for the instances on the current image
        true_masks = [x.astype(bool) for x in data["instances"].get_fields()["gt_masks"].numpy()]   # Get the true binary masks for the instances on the current image
        if len(true_classes) < 1: y_true = np.zeros_like(a=img)                             # If no objects are in the image, the y_true is simply a black image 
        else: y_true = draw_mask_image(mask_list=true_masks, lbl_list=true_classes, meta_data=meta_data)  # If objects are present, create a mask image for the true masks
        # plt.imshow(y_true)
        # plt.show(block=False)

        # The predicted image 
        y_pred_dict = predictor.__call__(img)["instances"].get_fields()                     # y_pred_dict is a dict with keys ['pred_masks', 'pred_boxes', 'scores', 'pred_classes']
        if "hungarian" in matching_type.lower():                                            # If we are matching objects using Hungarian algorithm ... 
            y_pred = hungarian_matching(y_pred_dict=y_pred_dict, data=data,                 # ... compute the Hungarian matching ...
                    predictor=predictor, matcher=matcher, FLAGS=FLAGS, meta_data=meta_data) # ... between ground truth and predictions
        else: y_pred,_ = NMS_pred(y_pred_dict=y_pred_dict, data=data, conf_thresh=FLAGS.conf_threshold, # Else, the matching will be done ...
                IoU_thresh=FLAGS.IoU_threshold, meta_data=meta_data)                        # ... using plain non max suppression 
        
        # Append the input image, y_true and y_pred to the dictionary
        img_ytrue_ypred["input"].append(img)                                                # Append the input image to the dictionary
        img_ytrue_ypred["y_true"].append(y_true)                                            # Append the ground truth to the dictionary
        img_ytrue_ypred["y_pred"].append(y_pred)                                            # Append the predicted mask to the dictionary
        if "vitrolife" in FLAGS.dataset_name.lower():                                       # If we are visualizing the vitrolife dataset
            img_ytrue_ypred["PN"].append(int(data["image_custom_info"]["PN_image"]))        # Read the true number of PN on the current image
    del matcher, predictor
    return img_ytrue_ypred, data_batch, FLAGS, config


# position=[0.55, 0.08, 0.40, 0.75]
# epoch_num = None
# data_batches = None
# data_batch=None
# model_done_training = False 
# data_split = "train"
# device="cpu"
# config = cfg

# Define function to plot the images
def visualize_the_images(config, FLAGS, position=[0.55, 0.08, 0.40, 0.75], epoch_num=None, data_batches=None, model_done_training=False, device="cpu", matching_type="NMS"):
    # Get the datasplit and number of images to show
    fig_list, data_batches_final = list(), list()                                           # Initiate the list to store the figures in
    if data_batches is None:                                                                # If no previous data has been sent ...
        data_batches = [None, None, None]                                                   # ... it must be a list of None's...
    data_split_count = 1                                                                    # Initiate the datasplit counter
    fontdict = {'fontsize': 25}                                                             # Set the font size for the plot
    thing_colors = list(reversed(deepcopy(MetadataCatalog.get(config.DATASETS.TRAIN[0]).thing_colors[:-1]))) # Create a list of the thing colors
    thing_names = deepcopy(MetadataCatalog.get(config.DATASETS.TRAIN[0]).thing_classes)[:-1] + ["PN{}".format(x) for x in range(1,8)]   # Create a list of the class names used with numbered PNs
    colors_with_alpha = list()                                                              # Initiate a list of colors with alpha values 
    for thing_color in thing_colors:                                                        # Iterate over all colors in the thing_colors list
        t_col = tuple()                                                                     # Create a new tuple value 
        for col_val in thing_color:                                                         # For each color value item in the current tuple_color
            t_col += (np.divide(float(col_val),255),)                                       # Squeeze the value from [0, 255] to [0, 1] and append it to the earlier new tuple
        colors_with_alpha.append(t_col + (float(1),))                                       # When all colors have been added to the new tuple, the alpha value of 1 gets added as well
    thing_color_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Thing colors cmap", colors_with_alpha, len(colors_with_alpha))  # Convert the list of colors into a pyplot colormap
    boundaries = np.linspace(0, len(colors_with_alpha), len(colors_with_alpha)+1)           # Create the boundaries for the colormap
    norm = matplotlib.colors.BoundaryNorm(boundaries, len(colors_with_alpha))               # Convert it into some kind of boundary normalization, used for discretizing the colormap
    for data_split, data_batch in tqdm(zip(["train", "val", "test"], data_batches),         # Iterate through the three splits available
                unit="Data_split", ascii=True, desc="Dataset split {:d}/{:d}".format(data_split_count, 3),
                bar_format="{desc}  | {percentage:3.0f}% | {bar:35}| {n_fmt}/{total_fmt} [Spent: {elapsed}. Remaining: {remaining}{postfix}]"):      
        data_split_count += 1                                                               # Increase the datasplit counter for the progress bar 
        if "vitrolife" not in FLAGS.dataset_name.lower() and data_split=="test": continue   # Only vitrolife has a test dataset. ADE20K doesn't. 
        # Extract information about the dataset used
        img_ytrue_ypred, data_batch, FLAGS, config = create_batch_img_ytrue_ypred(config=config, data_split=data_split, # Create the batch of images that needs to be visualized ...
            FLAGS=FLAGS, data_batch=data_batch, model_done_training=model_done_training, device=device, matching_type=matching_type) # ... and return the images in the data_batch dictionary
        class_names = list(reversed(thing_names))
        # max_PN_num = np.max(img_ytrue_ypred["PN"])
        # max_PN_num = 0
        # if max_PN_num > 0:
        #     class_names = thing_names[:np.where(["PN{}".format(max_PN_num) in x for x in thing_names])[0][0]+1]
        # else: class_names = thing_names[:np.where(["PN" in x for x in thing_names])[0][0]]  # Set class names if no PNs are present 
        if "vitrolife" in FLAGS.dataset_name.lower():                                       # If we are working on the vitrolife dataset sort the ...
            data_batch = sorted(data_batch, key=lambda x: x["image_custom_info"]["PN_image"])   # ... data_batch after the number of PN per found image
            img_ytrue_ypred = sort_dictionary_by_PN(data=img_ytrue_ypred)                   # And then also sort the data dictionary
        num_rows, num_cols = 3, len(data_batch)                                             # The figure will have three rows (input, y_pred, y_true) and one column per image
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(int(np.ceil(len(data_batch)*4)), 12))
        row = 0                                                                             # Initiate the row index counter (all manual indexing could have been avoided by having created img_ytrue_ypred as an OrderedDict)
        for key in img_ytrue_ypred.keys():                                                  # Loop through all the keys in the batch dictionary
            if key.lower() not in ['input', 'y_true', 'y_pred']: continue                   # If the key is not one of (input, y_pred, y_true), we simply skip to the next one
            for col, img in enumerate(img_ytrue_ypred[key]):                                # Loop through all available images in the dictionary
                if "vitrolife" in FLAGS.dataset_name.lower():                               # If we are visualizing the vitrolife dataset
                    axes[row,col].set_title("{:s} with {:.0f} PN".format(key, img_ytrue_ypred["PN"][col]), fontdict=fontdict)   # Create the title for the plot with the number of PN
                else: axes[row,col].set_title("{:s}".format(key), fontdict=fontdict)        # If on ADE20K, then the title is simply the key value
                axes[row,col].imshow(img, cmap="gray")                                      # Display the image 
                axes[row,col].set_axis_off()                                                # Remove axes xticks and yticks 
            row += 1                                                                        # Increase the row counter by 1
        try: fig = move_figure_position(fig=fig, position=position)                         # Try and move the figure to the wanted position (only possible on home computer with a display)
        except: pass                                                                        # Except, simply just let the figure retain the current position
        fig_name_init = "Segmented_{:s}_data_samples_from_".format(data_split)              # Initialize the figure name
        if epoch_num is not None: fig_name = "{:s}epoch_{:d}.jpg".format(fig_name_init, epoch_num)                      # If an epoch number has been specified, the figure name will contain that
        else: fig_name = "{:s}{:s}_training.jpg".format(fig_name_init, "after" if model_done_training else "before")    # Otherwise the visualization happens before/after training
        fig.tight_layout()                                                                  # Assures the subplots are plotted tight around each other
        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=thing_color_cmap), # Create a colorbar ... 
                ax=axes.ravel().tolist(), orientation='vertical')                           # ... shared by all axes
        cbar.set_ticks(np.add(np.arange(0,len(class_names),1), 0.5))                        # Set the ticks in the middle of the discretized area
        cbar.set_ticklabels(class_names)                                                    # Let the ticklabels be the class names of the current color 
        fig.savefig(os.path.join(get_save_dirs(config=config, dataset_split=data_split), fig_name), bbox_inches="tight")    # Save the figure in the correct output directory
        fig_list.append(fig)                                                                # Append the current figure to the list of figures
        data_batches_final.append(data_batch)                                               # Append the current data_batch to the list of data_batches
        fig.show() if FLAGS.display_images==True else plt.close(fig)                        # Display the figure if that is the chosen option
    return fig_list, data_batches_final, config, FLAGS                                      # Return the figure, the dictionary with the used images, the config and the FLAGS arguments

