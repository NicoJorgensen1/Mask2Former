import numpy as np 
import os 
from PIL import Image 
import cv2 
import copy 
import json
from matplotlib import pyplot as plt 
from glob import glob 
from panopticapi.utils import id2rgb, rgb2id
from detectron2.data import DatasetCatalog, MetadataCatalog

# The path for the datasets
datasets_path = os.path.join("/mnt", "c", "Users", "Nico-", "OneDrive - Aarhus Universitet", "Alting", "Biomedicinsk Teknologi", "5. semester", "Speciale", "Datasets")
cityscapes_path = os.path.join(datasets_path, "Cityscapes")#, "gtFine_trainvaltest", "gtFine")
vitrolife_path = os.path.join(datasets_path, "Vitrolife_dataset")
ADE20K_path = os.path.join(datasets_path, "ADEChallengeData2016")
coco_path = os.path.join(datasets_path, "coco_2017")


# Define a function to check an image is grayscale or not 
def isgray(img):
    if len(img.shape) < 3: return True
    if img.shape[2]  == 1: return True
    b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    if (b==g).all() and (b==r).all(): return True
    return False

# Mask out an image using a predefined mask 
def mask_lbl_image(lbl_image, idx_mask):
    lbl_final_mask = np.zeros(shape=(lbl_image.shape[0], lbl_image.shape[1]), dtype=np.uint8)
    if not isgray(lbl_image):
        idx_mask = np.stack((idx_mask,)*lbl_image.shape[-1], axis=-1)
    lbl_final_mask = np.multiply(lbl_image, idx_mask)
    return lbl_final_mask

################################################ CITYSCAPES ################################################

# Convert cityscapes training images into panoptic format
# from cityscapesscripts.preparation import createPanopticImgs
# createPanopticImgs.convert2panoptic(cityscapesPath=cityscapes_path, outputFolder=None)

# Read the cityscapes metadata
metadata_cityscapes = copy.deepcopy(MetadataCatalog.get("cityscapes_fine_panoptic_train"))
stuff_classes_cityscapes = metadata_cityscapes.stuff_classes
thing_classes_cityscapes = metadata_cityscapes.thing_classes
label_divisor_cityscapes = metadata_cityscapes.label_divisor
thing_id_to_contiguous_id_cityscapes = metadata_cityscapes.thing_dataset_id_to_contiguous_id
stuff_id_to_contiguous_id_cityscapes = metadata_cityscapes.stuff_dataset_id_to_contiguous_id
thing_id_keys_cityscapes = list(thing_id_to_contiguous_id_cityscapes.keys())
thing_id_values_cityscapes = list(thing_id_to_contiguous_id_cityscapes.values())
stuff_id_keys_cityscapes = list(stuff_id_to_contiguous_id_cityscapes.keys())
stuff_id_values_cityscapes = list(stuff_id_to_contiguous_id_cityscapes.values())
thing_classes_id_contiguous_cityscapes = np.asarray(thing_classes_cityscapes)[thing_id_values_cityscapes].tolist()
stuff_classes_id_contiguous_cityscapes = np.asarray(stuff_classes_cityscapes)[stuff_id_values_cityscapes].tolist()

# Read the ground truth images 
img_name_cityscape = "bremen_000000_000019"
files_used_cityscape = [str(x) for x in Path(cityscapes_path).rglob("*"+img_name_cityscape+"*")]                                # Get file names using pathlib.Path => way smarter than using os !! 
gt_color = np.asarray(Image.open([x for x in files_used_cityscape if "gtfine_color" in x.lower()][0]).convert("RGB"))           # Colored semantic ground truth
gt_fine_instance = np.asarray(Image.open([x for x in files_used_cityscape if "gtfine_instanceid" in x.lower()][0]))             # Instance segmentation ground truth. Almost equal to panoptic GT
gt_fine_labels = np.asarray(Image.open([x for x in files_used_cityscape if "gtfine_labelids" in x.lower()][0]))                 # Gray scale semantic segmentation ground truth
gt_panoptic = np.asarray(Image.open([str(x) for x in files_used_cityscape if "panoptic" in x.lower()][-1]))                     # Panoptic GT
orig_img_cityscapes = np.asarray(Image.open([str(x) for x in files_used_cityscape if "leftimg" in x.lower()][-1]))

panoptic_id = rgb2id(gt_panoptic)           # Create a ID image from the panoptic ground truth 
instance_rgb = id2rgb(gt_fine_instance)     # Create an RGB color image from the instance ground truth. 

# Read the panoptic json file 
with open(os.path.join(cityscapes_path, "cityscapes_panoptic_train.json"), "rb") as json_cityscapes_panoptic_train:
    pan_json_cityscapes = json.load(json_cityscapes_panoptic_train)
annotations_cityscapes = [x for x in pan_json_cityscapes["annotations"] if img_name_cityscape in x["file_name"]][-1]
annotated_ids_cityscapes = sorted([x["id"] for x in annotations_cityscapes["segments_info"]])
annotated_category_ids_cityscapes = sorted([x["category_id"] for x in annotations_cityscapes["segments_info"]])
categories_cityscape = pan_json_cityscapes["categories"]
for idx, item in enumerate(categories_cityscape):
    print("This is the {:>2}. class.\tIt has name: {:<13}\tIt has id: {:>2}.\tIt has supercategory: {:<12}\t It is a thing: {:<5}\t It has color {:<16}".format(
            idx+1,item["name"], item["id"], item["supercategory"], "True" if item["isthing"]==1 else "False", str(item["color"])))


cat_id, class_id = list(), list()
for item in pan_json_cityscapes["annotations"]:
    for item2 in item["segments_info"]:
        cat_id.append(item2["category_id"])
        class_id.append(item2["id"])
cat_id = sorted(np.unique(cat_id).tolist())
class_id = sorted(np.unique(class_id).tolist())


# Check the unique colors of the instance RGB and panoptic GT images. They are almost similar 
instance_rgb_colors = np.unique(instance_rgb.reshape(-1,3),axis=0)          # Instance_rgb has [1,0,0], [3,0,0], [4,0,0] colors that panoptic GT doesnt have 
gt_panoptic_colors = np.unique(gt_panoptic.reshape(-1,3),axis=0)            # Panoptic GT has [0,0,0] that instance_rgb doesnt have

# Create the mask to label out classes with instance ID labels. Stuff classes have values "semantic label", while thing classes have "semantic*1000+instance_label" => for this instance segmentation ground truth
unique_ids = sorted(np.unique(gt_fine_instance).tolist())                   # Unique instance IDs 
unique_pan_ids = sorted(np.unique(panoptic_id).tolist())
mask_id = gt_fine_instance < 50


# Plot the images 
plt.subplot(2,2,1)
plt.imshow(orig_img_cityscapes, cmap="gray")
plt.subplot(2,2,2)
plt.imshow(gt_fine_labels, cmap="gray")
plt.subplot(2,2,3)
plt.imshow(gt_fine_instance, cmap="gray")
plt.subplot(2,2,4)
plt.imshow(gt_panoptic, cmap="gray")
plt.show(block=False)


################################################ ADE20K ################################################
img_name_ADE20K = "ADE_train_00000002"
#  = glob(os.path.join(ADE20K_path, "**/*", img_name_ADE20K),  recursive=True)
from pathlib import Path 
files_used_ADE20K = [str(x) for x in Path(ADE20K_path).rglob(img_name_ADE20K+"*")]

# Reading the ADE20K metadata
metadata_ade20k = copy.deepcopy(MetadataCatalog.get("ade20k_sem_seg_val"))

_ = [print(x) for x in list(MetadataCatalog)]





################################################ COCO ################################################

panoptic_coco_path = os.path.join(coco_path, "panoptic_annotations_trainval2017", "annotations", "panoptic_val2017")
pan_img_name = "000000000872"
pan_img_coco = np.asarray(Image.open([str(x) for x in Path(panoptic_coco_path).rglob(pan_img_name+"*")][-1]))
pan_id_img_coco = rgb2id(pan_img_coco)
with open(os.path.join(os.path.dirname(panoptic_coco_path), "panoptic_val2017.json"), "rb") as json_file:
    pan_coco_json = json.load(json_file)

# Read unique pan_img_coco values and ids 
unique_colors_coco_pan_img = np.unique(pan_img_coco.reshape(-1,3),axis=0)
unique_ids_coco_pan = sorted(np.unique(pan_id_img_coco).tolist())

annotations = [x for x in pan_coco_json["annotations"] if pan_img_name in x["file_name"]][-1]
img_ids_coco = sorted([x["id"] for x in annotations["segments_info"]])


# Read coco metadata 
metadata_coco = copy.deepcopy(MetadataCatalog.get("coco_2017_val_panoptic"))
stuff_classes_coco = metadata_coco.stuff_classes
thing_classes_coco = metadata_coco.thing_classes
label_divisor_coco = metadata_coco.label_divisor
thing_data_id_to_cont_id_coco = metadata_coco.thing_dataset_id_to_contiguous_id
thing_data_id_to_cont_id_keys_coco = thing_data_id_to_cont_id_coco.keys()
thing_data_id_to_cont_id_values_coco = thing_data_id_to_cont_id_coco.values()
stuff_data_id_to_cont_id_coco = metadata_coco.stuff_dataset_id_to_contiguous_id
stuff_data_id_to_cont_id_keys_coco = stuff_data_id_to_cont_id_coco.keys()
stuff_data_id_to_cont_id_values_coco = stuff_data_id_to_cont_id_coco.values()

vars(metadata_coco).keys()
## Read the thing and stuff dataset id contiguous id to and see how that should be
## Then the Vitrolife dataset should be formed the same way 



