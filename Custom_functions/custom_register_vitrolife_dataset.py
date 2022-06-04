import os
import pickle
import cv2
import time 
import torch 
import shutil 
import json 
import pycocotools
import pandas as pd
import numpy as np
from PIL import Image 
from natsort import natsorted
from copy import deepcopy
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from panopticapi.utils import id2rgb, rgb2id 
from detectron2.data.datasets import register_coco_panoptic

# Create dictionary to store the class names and IDs 
stuff_class_labels = {kk: val for kk,val in enumerate(["Background", "Well", "Zona", "Perivitelline space", "Cell"])}
thing_class_labels = {kk: val for kk, val in enumerate(["PN"])}
panoptic_class_labels = {kk+1: val for kk, val in enumerate(["Background", "Well", "Zona", "Perivitelline space", "Cell", "PN"])}


# Define a function to convert a string of both letters and integers into all integers
def convert_string_and_int_func(string, convert_to="integers"):
    assert isinstance(string, str), "Only strings are accepted"
    assert any([convert_to in ["strings", "integers"]]), "Output has to be either strings or integers"
    output_list = list()                                                                            # Initiate a list of the outputs 
    for item in string:                                                                             # Iterate through all elements in the input string 
        if convert_to == "strings":                                                                 # if we want the output to be a string ... 
            if item.isnumeric(): output_list.append(chr(int(item)+97))                              # 97 is the first character, its an 'a', the elements before is weird unicode characters ... 
            else: output_list.append(item)                                                          # if the item is already a string, simply append it to the list 
        elif convert_to == "integers":                                                              # Else, if we want the output to be an integer ...
            if item.isnumeric(): output_list.append(int(item))                                      # ... if the current element is already an integer, simply append it to the list 
            else: output_list.append(ord(item))                                                     # ... else, the ord function will convert letters to unicode characters 
    output = "".join([str(x) for x in output_list])                                                 # Join the list into a string 
    if convert_to=="integers": output=int(output)                                                   # If we want integers, convert the output string into a long integer 
    return output                                                                                   # Return the final converted output 


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


# run_mode = "test"
# debugging = False 
# visualize = False 
# vitrolife_dataset_filepath = os.path.join(os.getenv("DETECTRON2_DATASETS"), "Vitrolife_dataset")
# img_filename = natsorted([x for x in os.listdir(os.path.join(vitrolife_dataset_filepath, "raw_images")) if x.endswith(".jpg")])[0]


# Define the function to return the list of dictionaries with information regarding all images available in the vitrolife dataset
def vitrolife_dataset_function(run_mode="train", debugging=False, visualize=False):
    # Find the folder containing the vitrolife dataset  
    vitrolife_dataset_filepath = os.path.join(os.getenv("DETECTRON2_DATASETS"), "Vitrolife_dataset")    # Get the path to the vitrolife dataset
    dataset_dir_end_number = ""                                                                     # This is how it is normally, 
    dataset_dir_end_number = "3"                                                                    # This is how it is temporarily while changing dataset folders while other scripts are running 
    
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

        sem_seg_mask_filename_list = [x for x in os.listdir(os.path.join(vitrolife_dataset_filepath,# Find the corresponding ...
                'annotations_semantic_masks'+dataset_dir_end_number)) if img_filename_wo_ext in x and x.endswith(".png")]  # ... semantic mask filename
        if len(sem_seg_mask_filename_list) != 1:                                                    # If either zero masks or more than one mask is found ...
            continue                                                                                # ... skip the current image 
        sem_seg_mask_filename = os.path.join(vitrolife_dataset_filepath,                            # Get the mask filename used for ...
                "annotations_semantic_masks"+dataset_dir_end_number, sem_seg_mask_filename_list[0]) # ... semantic segmentation as a string 
        panoptic_mask_filename_list = [x for x in os.listdir(os.path.join(vitrolife_dataset_filepath,   # Find the corresponding ...
                'annotations_panoptic_masks'+dataset_dir_end_number)) if img_filename_wo_ext in x and x.endswith(".png")]   # ... panoptic mask filename
        if len(panoptic_mask_filename_list) != 1:                                                   # If we haven't found one and only one panoptic image file ...
            continue                                                                                # ... then skip this image ... 
        panoptic_mask_filename = os.path.join(vitrolife_dataset_filepath,                           # Read the filename ...
                "annotations_panoptic_masks"+dataset_dir_end_number, panoptic_mask_filename_list[0])    # ... for the panoptic mask 
        annotation_dict_filenames = [x for x in os.listdir(os.path.join(vitrolife_dataset_filepath, # Find the corresponding ...
                'annotations_instance_dicts'+dataset_dir_end_number)) if img_filename_wo_ext in x and x.endswith(".pkl")]   # ... annotation dictionary filename
        if len(annotation_dict_filenames) != 1:                                                     # Continue only if we find only one dict filename
            continue 
        annotation_dict_filename = os.path.join(vitrolife_dataset_filepath,                         # Read the annotation ...
                'annotations_instance_dicts'+dataset_dir_end_number, annotation_dict_filenames[0])  # ... dictionary filename
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
            if "PN" not in key.upper():                                                             # If the current key is not the PN ... 
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
            obj["category_id"] = np.asarray(list(thing_class_labels.keys()))[[bool(x in key) for x in list(thing_class_labels.values())]].item()  # Get the category_ID from the key-value pair of the class_labels dictionary
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
                cv2.waitKey(0)                                                                      # Set the waitkey to 0 => necessary in order to display images 
                time.sleep(0.5)                                                                     # Sleep for 0.5 seconds 
        if visualize:                                                                               # If the user chose to visualize the images ...
            cv2.destroyAllWindows()                                                                 # ... when all images have been visualized, the windows will be closed 
        
        # Get the panoptic segmentation information
        segments_info = list()                                                                      # Initiate the list to store the panoptic segmentation information dictionaries 
        panoptic_mask = np.asarray(Image.open(panoptic_mask_filename))                              # Load the panoptic mask
        unique_colors = np.unique(panoptic_mask.reshape(-1,3),axis=0).tolist()                      # Read the unique values of the mask
        PN_value = {v:k for k,v in panoptic_class_labels.items()}["PN"]                             # Read the index value of where the PN is in the panoptic label dictionary
        PN_panoptic_count = 1
        for unique_color in unique_colors:                                                          # Iterate through all unique mask values 
            panoptic_obj = dict()                                                                   # Initiate a dict for each of the unqiue values in the panoptic mask 
            # panoptic_obj["id"] = rgb2id(np.stack((unique_value,)*3, axis=-1))                       # The object id calculated as ID = R + G*256 + B*256**2
            panoptic_obj["id"] = rgb2id(unique_color)                                               # The panoptic mask is made as ID masks with values [segment_value*label_divisor+instance_value]
            # panoptic_obj["category_id"] = np.min([unique_value, PN_value])                          # The category_id (class_id) is either the unique value or the value of a PN
            # panoptic_obj["category_id"] = int(np.floor(np.divide(unique_value, 1000)))              # The category_id (class_id) is either the unique value or the value of a PN
            panoptic_obj["category_id"] = int(np.floor(np.divide(panoptic_obj["id"], 1000)))        # The category_id (class_id) is either the unique value or the value of a PN
            panoptic_obj["iscrowd"] = 0                                                             # No classes are set to be "iscrowd" in the vitrolife dataset
            panoptic_obj["isthing"] = True if panoptic_obj["category_id"] == PN_value else False    # If the category ID represents a PN, then 'isthing' is true 
            panoptic_obj["label_name"] = panoptic_class_labels[panoptic_obj["category_id"]],
            panoptic_obj["label_name"] = str(panoptic_obj["label_name"][0])
            if "PN" in panoptic_obj["label_name"].upper():
                panoptic_obj["label_name"] = "{}{}".format(panoptic_obj["label_name"], PN_panoptic_count)
                PN_panoptic_count += 1
            segments_info.append(panoptic_obj)                                                      # Append the dictionary to the segments_info list 
        
        # Create the current image/mask pair 
        current_pair = {"file_name": row["img_file"],                                               # Initiate the dict of the current image with the full filepath + filename
                        "height": mask.shape[0],                                                    # Write the image height
                        "width": mask.shape[1],                                                     # Write the image width
                        "image_id": convert_string_and_int_func(img_filename_wo_ext),               # A unique key for the current image
                        "sem_seg_file_name": sem_seg_mask_filename,                                 # Add the filename for the semantic segmentation mask 
                        "annotations": annotations,                                                 # The list containing the annotations for the current image 
                        "pan_seg_file_name": panoptic_mask_filename,                                # The full path for the panoptic ground truth mask file 
                        "segments_info": segments_info,                                             # The list of dicts for defining the meaning of each id in the panoptic segmentation ground truth image 
                        "image_custom_info": row}                                                   # Add all the info from the current row to the dataset
        img_mask_pair_list.append(current_pair)                                                     # Append the dictionary for the current pair to the list of images for the given dataset
        count += 1                                                                                  # Increase the sample counter 
        if "nico" in vitrolife_dataset_filepath.lower():                                            # If we are working on my local computer ...
            if count >= 20000:                                                                         # ... and 20 images have already been loaded ...
                break                                                                               # ... then that is enough, thus quit reading the rest of the images 
    assert len(img_mask_pair_list) >= 1, print("No image/mask pairs found in {:s} subfolders 'raw_image' and 'masks'".format(vitrolife_dataset_filepath))
    if debugging==True: img_mask_pair_list=pickSamplesWithUniquePN(img_mask_pair_list)              # If we are debugging, we'll only get one sample with each number of PN's 
    return img_mask_pair_list                                                                       # Return the found list of dictionaries


# Function to register the dataset and the meta dataset for each of the three splitshuffleshuffles, [train, val, test]
def register_vitrolife_data_and_metadata_func(debugging=False, panoptic=False):
    # Common values for all types of datasets
    vitrolife_dataset_filepath = os.path.join(os.getenv("DETECTRON2_DATASETS"), "Vitrolife_dataset")    # Get the path to the vitrolife dataset
    thing_colors = [(185,220,255), (255,185,220), (220,255,185), (185,255,0),                       # Set colors for the ...
                    (0,185,220), (220,0,185), (115,45,115), (45,115,45)]                            # ... different numbers of PNs 
    thing_id = {kk+1: kk for kk in list(thing_class_labels.keys())}                                 # Get a dictionary of continuous keys
    stuff_colors = [(0,0,0), (255,0,0), (0,255,0), (0,0,255), (255,255,0), (185,220,255)]           # Set random colors for when the images will be visualized
    stuff_id = {kk: key for kk,key in enumerate(range(len(stuff_class_labels.keys())))}             # Create a dictionary with the class_id's as both keys and values
    panoptic_colors = stuff_colors[:-1] + thing_colors 
    panoptic_PN_count = 0
    panoptic_id = dict()
    for class_idx, class_label in enumerate(list(panoptic_class_labels.values())):
        if "PN" in class_label.upper():
            panoptic_PN_count += 1
        panoptic_class_id = (class_idx+1) * 1000 + panoptic_PN_count 
        panoptic_id[class_idx] = panoptic_class_id
    # For panoptic registration
    image_root = os.path.join(vitrolife_dataset_filepath, "raw_images")
    panoptic_root = os.path.join(vitrolife_dataset_filepath, "annotations_panoptic_masks")
    reduced_json_files = os.path.join(vitrolife_dataset_filepath, "reduced_json_files")
    if not os.path.isdir(reduced_json_files):
        os.makedirs(reduced_json_files)
    for split_mode in ["train", "val", "test"]:                                                     # Iterate over the three dataset splits ... 
        name = "vitrolife_dataset_{}".format(split_mode)
        panoptic_json = os.path.join(vitrolife_dataset_filepath, "dataset_json_files", "panoptic_json_file_{}.json".format(split_mode))
        instance_json = os.path.join(vitrolife_dataset_filepath, "dataset_json_files", "instance_json_file_{}.json".format(split_mode))
        # if panoptic == False:
        DatasetCatalog.register(name, lambda split_mode=split_mode: vitrolife_dataset_function(run_mode=split_mode, debugging=debugging))    # Register the dataset
        # else:
        #     register_coco_panoptic(name=name, metadata=dict(), image_root=image_root, panoptic_root=panoptic_root, panoptic_json=panoptic_json, instances_json=instance_json)
        dataset_copy = vitrolife_dataset_function(run_mode=split_mode)

        # Edit the json files by removing files that aren't used (because on my local computer only a subset of the entire dataset is used)
        image_id_used = list()
        for image_element in dataset_copy:
            image_id_used.append(str(image_element["image_id"]))
        kept_pan_images, kept_pan_annotations, kept_inst_images, kept_inst_annotations = list(), list(), list(), list()
        with open(panoptic_json) as ps_json:
            pan_seg_json = json.load(ps_json)
        with open(instance_json) as in_json:
            inst_seg_json = json.load(in_json)
        for pan_img_element, pan_ann_element in zip(pan_seg_json["images"], pan_seg_json["annotations"]):
            if pan_img_element["id"] in image_id_used:
                pan_img_element["id"] = int(pan_img_element["id"])
                kept_pan_images.append(pan_img_element)
            if pan_ann_element["image_id"] in image_id_used:
                pan_ann_element["image_id"] = int(pan_ann_element["image_id"])
                kept_pan_annotations.append(pan_ann_element)
        for ins_img_element, inst_ann_element in zip(inst_seg_json["images"], inst_seg_json["annotations"]):
            if ins_img_element["id"] in image_id_used:
                ins_img_element["id"] = int(ins_img_element["id"])
                kept_inst_images.append(ins_img_element)
            if inst_ann_element["image_id"] in image_id_used:
                inst_ann_element["image_id"] = int(inst_ann_element["image_id"])
                kept_inst_annotations.append(inst_ann_element)
        pan_seg_json["images"] = kept_pan_images
        pan_seg_json["annotations"] = kept_pan_annotations
        inst_seg_json["images"] = kept_inst_images
        inst_seg_json["annotations"] = kept_inst_annotations

        # Save the edited json files again 
        reduced_panoptic_json_file_path = os.path.join(reduced_json_files, "reduced_panoptic_json_file_{}.json".format(split_mode))
        with open(reduced_panoptic_json_file_path, "w") as fb:
            json.dump(pan_seg_json, fb, sort_keys=True, indent=4)
        reduced_instance_json_file_path = os.path.join(reduced_json_files, "reduced_instance_json_file_{}.json".format(split_mode))
        with open(reduced_instance_json_file_path, "w") as fb:
            json.dump(inst_seg_json, fb, sort_keys=True, indent=4)

        # Set the metadata for the current dataset 
        MetadataCatalog.get("vitrolife_dataset_{:s}".format(split_mode)).set(thing_classes=list(thing_class_labels.values()),   # Name the thing classes
                                                                        thing_colors=thing_colors,                              # Color the thing classes
                                                                        thing_dataset_id_to_contiguous_id=panoptic_id if panoptic else thing_id,    # Give ID's to the thing classes
                                                                        stuff_classes=list(stuff_class_labels.values()),        # Get the semantic stuff classes
                                                                        stuff_colors = stuff_colors,                            # Set the metadata stuff_colors for visualization
                                                                        stuff_dataset_id_to_contiguous_id=stuff_id,             # Give the ID to the stuff classes 
                                                                        ignore_label=255,                                       # No labels will be ignored as 255 >> num_classes ...
                                                                        panoptic_classes = list(panoptic_class_labels.values()),# Get the panoptic class labels 
                                                                        panoptic_colors = panoptic_colors,                      # Get the panoptic colors for visualization
                                                                        panoptic_dataset_id_to_contiguous_id = panoptic_id,     # Get the panoptic continuous IDs 
                                                                        panoptic_root = panoptic_root,                          # The folder path to the root of the panoptic RGB masks
                                                                        panoptic_json = reduced_panoptic_json_file_path,        # The filepath to the panoptic json 
                                                                        json_file = reduced_instance_json_file_path,            # The filepath to the instance json 
                                                                        image_root = image_root,
                                                                        num_files_in_dataset=len(dataset_copy))                 # Write the length of the dataset
    assert any(["vitrolife" in x for x in list(MetadataCatalog)]), "Datasets have not been registered correctly"                # Assuring the dataset has been registered correctly




# Test that the function will actually return a list of dicts
# train_dataset = img_mask_list_train = vitrolife_dataset_function(run_mode="train", visualize=False)
# val_dataset = img_mask_list_val = vitrolife_dataset_function(run_mode="val", visualize=False)
# test_dataset = img_mask_list_test = vitrolife_dataset_function(run_mode="test", debugging=False, visualize=False)


# # Visualize some random samples using the Detectron2 visualizer 
# try: register_vitrolife_data_and_metadata_func(debugging=True)
# except Exception as ex:
#     error_string = "An exception of type {0} occured while trying to register the datasets. Arguments:\n{1!r}".format(type(ex).__name__, ex.args)
# vitro_metadata = MetadataCatalog.get("vitrolife_dataset_test")
# visualize_dataset = img_mask_list_test = vitrolife_dataset_function(run_mode="test", debugging=True, visualize=False)
# for kk, d in enumerate(visualize_dataset):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=vitro_metadata, scale=0.5)
#     # out = visualizer.draw_dataset_dict(d)
#     pan_seg = cv2.imread(d["pan_seg_file_name"])
#     out = visualizer.draw_panoptic_seg(torch.from_numpy(pan_seg), d["segments_info"])
#     mask_im = out.get_image()[:, :, ::-1]
#     img = cv2.resize(img, mask_im.shape[:-1], cv2.INTER_LINEAR)
#     border = np.multiply(np.ones((250, 25, 3)).astype(np.uint8), (150, 0, 255)).astype(np.uint8)                # Create a border between subplots
#     concated_image = cv2.hconcat([img, border, mask_im])
#     window_name = "Concatenated image with {:.0f} PN".format(d["image_custom_info"]["PN_image"])
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#     cv2.imshow(window_name, concated_image)
#     cv2.moveWindow(window_name, 40,30)
#     cv2.resizeWindow(window_name,1100,800)
#     cv2.waitKey(0)
#     time.sleep(1)
#     cv2.destroyAllWindows() 
#     time.sleep(0.01)
#     if kk >= 10:
#         break 

