import os
import pickle
import cv2
import time 
import pycocotools
import pandas as pd
import numpy as np
from natsort import natsorted
from copy import deepcopy
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer

# Create dictionary to store the class names and IDs 
class_labels = {kk: val for kk,val in enumerate(["Well", "Zona", "Perivitelline space", "Cell", "PN"])}
class_labels = {kk: val for kk, val in enumerate(["PN"])}

# Function to select sample dictionaries with unique PN's
def pickSamplesWithUniquePN(dataset_dict):
    PNs_found = np.zeros((1,8), dtype=bool)                                                         # Create a [1,2,...,7] list filled with False values to track if a sample with a specified PN number has been found
    data_used = []                                                                                  # Initiate a new list of dictionaries
    for data in dataset_dict:                                                                       # Iterate over all dictionaries in the list of dictionaries
        PN = int(data["image_custom_info"]["PN_image"])                                             # Get the number of PN's in the current sample
        if PNs_found[0,PN] == False:                                                                # If no sample with the current PN_number has been found ...
            PNs_found[0,PN] = True                                                                  # ... the corresponding entry in the PNs_found array are set to true ...
            data_used.append(data)                                                                  # ... and the data_used array is appended with the current sample
    data_used = sorted(data_used, key=lambda x: x["image_custom_info"]["PN_image"])                 # Sort the data_used list by the number of dictionaries
    return data_used


# run_mode = "val"
# debugging = False 
# visualize = False 


# Define the function to return the list of dictionaries with information regarding all images available in the vitrolife dataset
def vitrolife_dataset_function(run_mode="train", debugging=False, visualize=False):
    # Find the folder containing the vitrolife dataset  
    vitrolife_dataset_filepath = os.path.join(os.getenv("DETECTRON2_DATASETS"), "Vitrolife_dataset")    # Get the path to the vitrolife dataset
    
    # Find the metadata file
    metadata_file = os.path.join(vitrolife_dataset_filepath, "metadata.csv")                        # Get the csv file with the metadata for all images
    df_data = pd.read_csv(metadata_file)                                                            # Read the csv file 
    df_data = df_data.set_index(["HashKey","Well"])                                                 # Set the two columns HashKey and Well as index columns

    # Create the list of dictionaries with information about all images
    img_mask_pair_list = []                                                                         # Initiate the list to store the information about all images
    total_files = len(os.listdir(os.path.join(vitrolife_dataset_filepath, "raw_images")))           # List all the image files in the raw_images directory
    iteration_counter = 0                                                                           # Initiate a iteration counter 
    count = 0                                                                                       # Initiate a counter to count the number of images inserted to the dataset
    available_image_files = natsorted([x for x in os.listdir(os.path.join(vitrolife_dataset_filepath, "raw_images")) if x.endswith(".jpg")])    # Read a list of all available images 
    for img_filename in tqdm(available_image_files, total=total_files, unit="img",                  # Loop through all files in the raw_images folder
            postfix="Read the Vitrolife {:s} dataset dictionaries".format(run_mode), leave=True,
            bar_format="{desc}  | {percentage:3.0f}% | {bar:45}| {n_fmt}/{total_fmt} | Spent: {elapsed}. Remaining: {remaining} | {postfix}]"):  
        iteration_counter += 1                                                                      # Increase the counter that counts the number of iterations in the for-loop
        img_filename_wo_ext = os.path.splitext(os.path.basename(img_filename))[0]                   # Get the image filename without .jpg extension
        img_filename_wo_ext_parts = img_filename_wo_ext.split("_")                                  # Split the filename where the _ is
        hashkey = img_filename_wo_ext_parts[0]                                                      # Extract the hashkey from the filename
        well = int(img_filename_wo_ext_parts[1][1:])                                                # Extract the well from the filename
        row = deepcopy(df_data.loc[hashkey,well])                                                   # Find the row of the corresponding file in the dataframe
        data_split = row["split"]                                                                   # Find the split for the current image, i.e. either train, val or test
        if data_split != run_mode: continue                                                         # If the current image is supposed to be in another split, then continue to the next image
        row["img_file"] = os.path.join(vitrolife_dataset_filepath, "raw_images", img_filename)      # Add the current filename for the input image to the row-variable
        annotation_dict_filenames = [x for x in os.listdir(os.path.join(vitrolife_dataset_filepath, 'annotations_masks')) if img_filename_wo_ext in x and x.endswith(".pkl")]   # Find the corresponding annotation mask filename
        if len(annotation_dict_filenames) != 1: continue                                            # Continue only if we find only one dict filename
        annotation_dict_filename = os.path.join(vitrolife_dataset_filepath, 'annotations_masks', annotation_dict_filenames[0])  # Read the annotation mask filename
        with open(annotation_dict_filename, "rb") as anno_file:                                     # Open a file handler for the current annotation pickle file
            annotation_dict_file = pickle.load(anno_file)                                           # Read the current annotation pickle file 
        
        if visualize:                                                                                                   # If we are debugging and the images must be visualized ...
            orig_im = cv2.imread(os.path.join(vitrolife_dataset_filepath, "raw_images", img_filename)).astype(np.uint8) # Read the original image
            border = np.multiply(np.ones((500, 25, 3)).astype(np.uint8), (150, 0, 255)).astype(np.uint8)                # Create a border between subplots
        # Iterate over all instances in the image
        annotations = list()                                                                        # The annotations must be a list of dictionaries
        mask = np.zeros_like(annotation_dict_file[list(annotation_dict_file.keys())[0]])            # Initially, the mask will be empty 
        positive_pixels_list = []                                                                   # Initiate a list to store the number of positive pixels in each mask 
        for key in annotation_dict_file.keys():                                                     # Loop through each key (=object/instance) in the current image
            if "pn" not in key.lower():                                                             # If the current key is not the PN ... 
                continue                                                                            # ... continue, as only PN's should be "things" classes 
            mask = deepcopy(annotation_dict_file[key])                                              # Get the current mask as the value of the given key 
            if np.sum(mask) < 2:                                                                    # We need at least two positive pixels ...
                continue                                                                            # ... to create a mask with the given instance 
            mask_pixel_coordinates = np.asarray(np.where(mask))                                     # Get all pixel coordinates for the true pixels in the mask
            x1, y1 = np.amin(mask_pixel_coordinates, axis=1)                                        # Extract the minimum x and y true pixel values
            x2, y2 = np.amax(mask_pixel_coordinates, axis=1)                                        # Extract the maximum x and y true pixel values
            bbox = [float(val) for val in [y1, x1, y2, x2]]                                         # Convert the bounding box to float values
            positive_pixels_list.append(np.sum(mask))                                               # Append the approved amount of positive pixels to the list 

            if len(positive_pixels_list) > 1:                                                       # If more than one object instance has been added ...
                if all([np.sum(mask) == positive_pixels_list[-1],                                   # ... and the current instance has the exact same amount of positive pixels ...
                                        obj["bbox"]==bbox,                                          # ... and the exact same bounding box ...
                                        "PN" not in key]):                                          # ... and is not a PN, then something is wrong ...
                    continue                                                                        # ... thus the current instance is simply skipped 
            # Create the dictionary for the current object instance
            obj = dict()                                                                            # Each instance on the image must be in the format of a dictionary
            obj["bbox"] = bbox                                                                      # Get the bounding box for the current object
            obj["bbox_mode"] = BoxMode.XYXY_ABS                                                     # The mode of the bounding box
            obj["category_id"] = np.asarray(list(class_labels.keys()))[[bool(x in key) for x in list(class_labels.values())]].item()    # Get the category_ID from the key-value pair of the class_labels dictionary
            obj["segmentation"] = pycocotools.mask.encode(np.asarray(mask, order="F"))              # Convert the mask into a COCO compressed RLE dictionary
            obj["iscrowd"] = 0                                                                      # No object instances are labeled as COCO crowd-regions 
            annotations.append(obj)                                                                 # Append the current object instance to the annotations list

            # If we are debugging and the images and masks must be visualized             
            if visualize:
                mask_to_show = np.repeat(np.reshape(np.multiply(mask.astype(float), 255), (500,500,1)), 3, 2).astype(np.uint8)      # Make the mask image 3D
                mask_to_show = cv2.rectangle(mask_to_show, (y1,x1), (y2,x2), (0,0,255), 5)                                          # Add the current bounding box to the mask image
                mask_to_show = cv2.putText(mask_to_show, key, (25,35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,0,0), 2, cv2.LINE_AA)   # Add the key (instance label) as a text
                mask_to_show = cv2.putText(mask_to_show, "Positive pixels: {}".format(np.sum(mask)), (25,70), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,0,0), 2, cv2.LINE_AA)    # Add the amount of positive pixels as a text
                im_to_show = cv2.hconcat([orig_im, border, mask_to_show])                           # Concatenate the original image and the mask into a single image because opencv can't operate with subplots
                winname = img_filename_wo_ext + "with {} PNs".format(int(row["PN_image"]))          # Create a name for the figure window
                cv2.namedWindow(winname)                                                            # Create a figure with the chosen name 
                cv2.moveWindow(winname, 50,50)                                                      # Move the upper left corner to this pixel position
                cv2.imshow(winname, im_to_show)                                                     # Display the image
                cv2.waitKey(0)
                time.sleep(0.5)
        if visualize:
            cv2.destroyAllWindows() 
        # Create the current image/mask pair 
        current_pair = {"file_name": row["img_file"],                                               # Initiate the dict of the current image with the full filepath + filename
                        "height": mask.shape[0],                                                    # Write the image height
                        "width": mask.shape[1],                                                     # Write the image width
                        "image_id": img_filename_wo_ext,                                            # A unique key for the current image
                        "annotations": annotations,                                                 # The list containing the annotations for the current image 
                        "annotation_mask_file_name": annotation_dict_filename,                      # Add the instance annotation mask image to the dataset list of dictionaries 
                        "image_custom_info": row}                                                   # Add all the info from the current row to the dataset
        img_mask_pair_list.append(current_pair)                                                     # Append the dictionary for the current pair to the list of images for the given dataset
        count += 1                                                                                  # Increase the sample counter 
        # if "nico" in vitrolife_dataset_filepath.lower():                                            # If we are working on my local computer ...
        #     if count > 25:                                                                          # ... and 25 images have already been loaded ...
        #         break                                                                               # ... then that is enough, thus quit reading the rest of the images 
    assert len(img_mask_pair_list) >= 1, print("No image/mask pairs found in {:s} subfolders 'raw_image' and 'masks'".format(vitrolife_dataset_filepath))
    img_mask_pair_list = natsorted(img_mask_pair_list)                                              # Sorting the list assures the same every time this function runs
    if debugging==True: img_mask_pair_list=pickSamplesWithUniquePN(img_mask_pair_list)              # If we are debugging, we'll only get one sample with each number of PN's 
    return img_mask_pair_list                                                                       # Return the found list of dictionaries


# Function to register the dataset and the meta dataset for each of the three splitshuffleshuffles, [train, val, test]
def register_vitrolife_data_and_metadata_func(debugging=False):
    # thing_colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0),                                   # Set random colors for the Well, Zona, PV Space and Cell classes
    #     (185,220,255), (255,185,220), (220,255,185), (185,255,0), (0,185,220), (220,0,185), (115,45,115), (45,115,45)]  # Set similar colors for the PN classes
    thing_colors = [(185,220,255), (255,185,220), (220,255,185), (185,255,0),                       # Set colors for the ...
                    (0,185,220), (220,0,185), (115,45,115), (45,115,45)]                            # ... different numbers of PNs 
    thing_id = {kk: key for kk,key in enumerate(list(class_labels.keys()))}                         # Get a dictionary of continuous keys
    for split_mode in ["train", "val", "test"]:                                                     # Iterate over the three dataset splits ... 
        DatasetCatalog.register("vitrolife_dataset_{:s}".format(split_mode), lambda split_mode=split_mode: vitrolife_dataset_function(run_mode=split_mode, debugging=debugging))    # Register the dataset
        MetadataCatalog.get("vitrolife_dataset_{:s}".format(split_mode)).set(thing_classes=list(class_labels.values()),     # Name the thing classes
                                                                            thing_colors=thing_colors,                      # Color the thing classes
                                                                            thing_dataset_id_to_contiguous_id=thing_id,     # Give ID's to the thing classes
                                                                            num_files_in_dataset=len(DatasetCatalog["vitrolife_dataset_{:}".format(split_mode)]())) # Write the length of the dataset
    assert any(["vitrolife" in x for x in list(MetadataCatalog)]), "Datasets have not been registered correctly"    # Assuring the dataset has been registered correctly


# Test that the function will actually return a list of dicts
train_dataset = img_mask_list_train = vitrolife_dataset_function(run_mode="train", visualize=False)
val_dataset = img_mask_list_val = vitrolife_dataset_function(run_mode="val", visualize=False)
test_dataset = img_mask_list_test = vitrolife_dataset_function(run_mode="test", debugging=False, visualize=True)


# Visualize some random samples using the Detectron2 visualizer 
try: register_vitrolife_data_and_metadata_func(debugging=True)
except Exception as ex:
    error_string = "An exception of type {0} occured while trying to register the datasets. Arguments:\n{1!r}".format(type(ex).__name__, ex.args)
vitro_metadata = MetadataCatalog.get("vitrolife_dataset_test")
for kk, d in enumerate(test_dataset):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=vitro_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    mask_im = out.get_image()[:, :, ::-1]
    img = cv2.resize(img, mask_im.shape[:-1], cv2.INTER_LINEAR)
    border = np.multiply(np.ones((250, 25, 3)).astype(np.uint8), (150, 0, 255)).astype(np.uint8)                # Create a border between subplots
    concated_image = cv2.hconcat([img, border, mask_im])
    window_name = "Concatenated image with {:.0f} PN".format(d["image_custom_info"]["PN_image"])
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, concated_image)
    cv2.moveWindow(window_name, 40,30)
    cv2.resizeWindow(window_name,1100,800)
    cv2.waitKey(0)
    time.sleep(1)
    cv2.destroyAllWindows() 
    time.sleep(0.01)
    if kk >= 15:
        break 
