import os
import cv2 
import numpy as np
import copy 
from PIL import Image 
import matplotlib.pyplot as plt 
import pickle

def isgray(img):
    if len(img.shape) < 3: return True
    if img.shape[2]  == 1: return True
    b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    if (b==g).all() and (b==r).all(): return True
    return False

def apply_colormap(mask_img, segmentation_type="panoptic", use_vitrolife=True, colors_used=None):
    if colors_used is None and use_vitrolife:
        thing_colors = [(185,220,255), (255,185,220), (220,255,185), (185,255,0),                       # Set colors for the ...
                    (0,185,220), (220,0,185), (115,45,115), (45,115,45)]                                # ... different numbers of PNs 
        stuff_colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (185,220,255)]                    # Set random colors for when the images will be visualized
        panoptic_colors = stuff_colors[:-1] + thing_colors                                              # Set random colors for the panoptic classes 
        
    unique_values = np.unique(mask_img)
    final_mask = np.zeros(shape=(mask_img.shape[0], mask_img.shape[1], 3))
    if use_vitrolife:
        if "panoptic" in segmentation_type.lower(): colors_used = panoptic_colors
        if "instance" in segmentation_type.lower(): colors_used = thing_colors
        if "semantic" in segmentation_type.lower(): colors_used = stuff_colors
    if use_vitrolife:
        colors_used.insert(0, (0,0,0))                                                                  # All kind of segmentations will have their first unique value = 0, which should be black background 
    for unique_value in unique_values:
        col_idx = np.where(unique_value == unique_values)[0][0]
        final_mask[mask_img==unique_value] = colors_used[col_idx]
        if "instance" in segmentation_type.lower() and unique_value != 0:
            bbox_coordinates = np.asarray(np.where((mask_img==unique_value).astype(bool)))              # Get all pixel coordinates for the white pixels in the mask
            x1, y1 = np.amin(bbox_coordinates, axis=1)                                                  # Extract the minimum x and y white pixel values
            x2, y2 = np.amax(bbox_coordinates, axis=1)                                                  # Extract the maximum x and y white pixel values
            # final_mask = cv2.rectangle(final_mask, (y1,x1), (y2,x2), colors_used[col_idx], 2)           # Overlay the bounding box for the current object on the current image 
    return final_mask.astype(np.uint8)


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
dataset_dir = os.path.join("/mnt", "c", "Users", "Nico-", "OneDrive - Aarhus Universitet", "Alting", "Biomedicinsk Teknologi", "5. semester", "Speciale", "Datasets")                 # Home WSL
if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("C:\\", dataset_dir.split(os.path.sep,1)[1])                                                                  # Home windows computer
if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("/mnt", "c", "Users", "wd974261", "Documents", "Python", "Datasets")                                          # Work WSL
if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("C:\\", dataset_dir.split(os.path.sep,1)[1])                                                                  # Work windows computer
if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("/home", "neal", "Panoptic_segmentation_using_deep_neural_networks", "Datasets")                              # Larac server
if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("/mnt", "home_shared", dataset_dir.split(os.path.sep, 2)[2])                                                  # Balder server
assert os.path.isdir(dataset_dir), "The dataset directory doesn't exist in the chosen location"


testing_dir = os.path.join(Mask2Former_dir, "ade20k_outputs")
Using_Vitrolife = False 
using_predicted_images = False  
if using_predicted_images:
    assert using_predicted_images != Using_Vitrolife, "We can't use predicted images when using Vitrolife images"
    endings = "out.jpg"
    image_endings = "prediction"
else:
    endings = "gt.png"
    image_endings = "gt"
img_string = "Vitrolife_" if Using_Vitrolife else ""


if Using_Vitrolife:
    Vitrolife_dataset_dir = os.path.join(dataset_dir, "Vitrolife_dataset")
    used_im = "30a73399b12b9c964815e1f71b0f27554b1b9203492fc792640078522426b1fe_W02"
    orig_im_path = [os.path.join(Vitrolife_dataset_dir, "raw_images", x) for x in os.listdir(os.path.join(Vitrolife_dataset_dir, "raw_images")) if used_im in x][0]
    sem_im_path = [os.path.join(Vitrolife_dataset_dir, "annotations_semantic_masks", x) for x in os.listdir(os.path.join(Vitrolife_dataset_dir, "annotations_semantic_masks")) if used_im in x][0]
    pan_im_path = [os.path.join(Vitrolife_dataset_dir, "annotations_panoptic_masks", x) for x in os.listdir(os.path.join(Vitrolife_dataset_dir, "annotations_panoptic_masks")) if used_im in x][0]
    inst_im_path = [os.path.join(Vitrolife_dataset_dir, "annotations_instance_dicts", x) for x in os.listdir(os.path.join(Vitrolife_dataset_dir, "annotations_instance_dicts")) if used_im in x][0]
else:
    orig_im_path = os.path.join(testing_dir, "jena_0026.png")
    inst_im_path = os.path.join(testing_dir, "jena_0026_inst_seg_"+endings)
    inst_im_path_pred = os.path.join(testing_dir, "jena_0026_inst_seg_"+"out.jpg")
    sem_im_path = os.path.join(testing_dir, "jena_0026_sem_seg_"+endings)
    pan_im_path = os.path.join(testing_dir, "jena_0026_pan_seg_"+endings)
    pan_im_path_pred = os.path.join(testing_dir, "jena_0026_pan_seg_"+"out.jpg")

orig_im = np.asarray(Image.open(orig_im_path))
sem_im =  np.asarray(Image.open(sem_im_path))
pan_im =  np.asarray(Image.open(pan_im_path))
inst_im_pred = np.asarray(Image.open(inst_im_path_pred))
pan_im_pred = np.asarray(Image.open(pan_im_path_pred))
if Using_Vitrolife:
    with open(inst_im_path, "rb") as fb:
        inst_dict = pickle.load(fb)
    inst_im = np.zeros_like(inst_dict[list(inst_dict.keys())[0]]).astype(np.uint8)
    PN_count = 1
    for kk, class_key in enumerate(inst_dict.keys()):
        if "PN" in class_key.upper():
            inst_im[inst_dict[class_key]] = PN_count
            PN_count += 1
    pan_im = pan_im[:,:,0]
elif not using_predicted_images and not Using_Vitrolife:
    inst_img = np.asarray(Image.open(inst_im_path))
    inst_im = np.zeros_like(inst_img).astype(np.uint8)
    color_count = 1
    for unique_value in np.unique(inst_img).tolist():
        if unique_value > 25:
            inst_im[inst_img==unique_value] = color_count
            color_count += 1
    if sem_im.shape[-1] > 3:
        sem_im = sem_im[:,:,:3]
    unique_colors_sem_im = np.unique(sem_im.reshape(-1,3), axis=0)
    sem_img = np.zeros(shape=(sem_im.shape[0], sem_im.shape[1]), dtype=np.uint8)
    for kk, unique_color in enumerate(unique_colors_sem_im):
        sem_img[np.all(sem_im==unique_color, axis=-1)] = kk
    sem_im = copy.deepcopy(sem_img)
    thing_colors, stuff_colors = list(), list()
    while True:
        new_col = tuple(np.random.randint(low=0, high=255, size=(1,3), dtype=int).squeeze())
        if new_col not in stuff_colors and np.sum(new_col) > 50:
            stuff_colors.append(new_col)
        if len(stuff_colors) == len(np.unique(sem_im).tolist()):
            break 
    while True:
        new_col = tuple(np.random.randint(low=0, high=255, size=(1,3), dtype=int).squeeze())
        if new_col not in stuff_colors and new_col not in thing_colors and np.sum(new_col) > 50:
            thing_colors.append(new_col)
        if len(thing_colors) == len(np.unique(inst_im).tolist()):
            break 
else:
    inst_im =  np.asarray(Image.open(inst_im_path))

if not using_predicted_images:
    if not Using_Vitrolife:
        stuff_col_numb, thing_col_numb = int(0), int(0)
        panop_colors = list()
        del pan_im
        stuff_colors.insert(0, (0,0,0))
        thing_colors.insert(0, (0,0,0))
    if isgray(inst_im):
        inst_im = apply_colormap(mask_img=inst_im, segmentation_type="instance", use_vitrolife=Using_Vitrolife, colors_used=None if Using_Vitrolife else thing_colors)
    if isgray(sem_im):
        sem_im = apply_colormap(mask_img=sem_im, segmentation_type="semantic", use_vitrolife=Using_Vitrolife, colors_used=None if Using_Vitrolife else stuff_colors)
    if Using_Vitrolife:
        if isgray(pan_im):
            pan_im = apply_colormap(mask_img=pan_im, segmentation_type="panoptic", use_vitrolife=Using_Vitrolife, colors_used=None if Using_Vitrolife else panop_colors)

    if not Using_Vitrolife:
        pan_im = copy.deepcopy(sem_im)
        unique_values = np.unique(inst_im.reshape(-1,3),axis=0)
        for kk, unique_value in enumerate(unique_values):
            if np.sum(unique_value) == 0:
                continue 
            idx = inst_im==unique_value
            new_col = tuple(unique_value.squeeze())
            pan_im = np.multiply(pan_im, ~idx)
            pan_im = np.add(pan_im, np.multiply(idx, inst_im))
    plt.imshow(inst_im, cmap="gray")
    plt.show(block=False)


border_size = 25 if Using_Vitrolife else 50
border_color = (0, 255, 3)

border_vertical = np.multiply(np.ones((orig_im.shape[0], border_size, 3)).astype(np.uint8), border_color).astype(np.uint8)
border_flat = np.multiply(np.ones((border_size, orig_im.shape[1], 3)).astype(np.uint8), border_color).astype(np.uint8)

orig_and_inst = cv2.hconcat(src=[orig_im,border_vertical,inst_im])
orig_and_sem  = cv2.hconcat(src=[orig_im,border_vertical,sem_im])
orig_and_pan =  cv2.hconcat(src=[orig_im,border_vertical,pan_im])

border_horizontal = np.multiply(np.ones((border_size, orig_and_pan.shape[1], 3)).astype(np.uint8), border_color).astype(np.uint8)

if not Using_Vitrolife:
    inst_im_gt = copy.deepcopy(inst_im)
    inst_im_gt_mask = np.any(a=(inst_im_gt > 0), axis=-1)
    inst_im_gt_mask_3D = np.transpose(np.stack((inst_im_gt_mask,)*3), [1,2,0])
    inst_im_gt = np.multiply(orig_im, ~inst_im_gt_mask_3D) + np.multiply(inst_im_gt, 0.7) + np.multiply(orig_im, inst_im_gt_mask_3D)*0.3
    inst_gt_and_pred_sideways = cv2.hconcat(src=[np.asarray(inst_im_gt).astype(np.uint8), border_vertical, inst_im_pred])
    plt.imshow(inst_gt_and_pred_sideways, cmap="gray")
    plt.show(block=False)
    Image.fromarray(inst_gt_and_pred_sideways).save(fp=os.path.join(testing_dir, img_string+"Instance_GT_and_prediction_{}.jpg".format(image_endings)))

    pan_im_gt = copy.deepcopy(pan_im)
    pan_im_gt = np.add(np.multiply(orig_im, 0.5), np.multiply(pan_im_gt, 0.6)).astype(np.uint8)
    pan_gt_and_pred_sideways = cv2.hconcat(src=[np.asarray(pan_im_gt).astype(np.uint8), border_vertical, pan_im_pred])
    plt.imshow(pan_gt_and_pred_sideways, cmap="gray")
    plt.show(block=False)
    Image.fromarray(pan_gt_and_pred_sideways).save(fp=os.path.join(testing_dir, img_string+"Panoptic_GT_and_prediction_{}.jpg".format(image_endings)))


inst_and_pan_sideways = cv2.hconcat(src=[inst_im,border_vertical,pan_im])
inst_and_pan_ontop = cv2.vconcat(src=[inst_im,border_flat,pan_im])
sem_and_pan_sideways = cv2.hconcat(src=[sem_im,border_vertical,pan_im])


orig_inst_sem_pan = cv2.vconcat(src=[orig_and_inst, border_horizontal, sem_and_pan_sideways])



plt.imshow(orig_inst_sem_pan)
plt.axis("off")
plt.show(block=False)

Image.fromarray(orig_and_inst).save(fp=os.path.join(testing_dir, img_string+"Original_and_instance_segmentation_{}.jpg".format(image_endings)))
Image.fromarray(orig_and_sem).save(fp=os.path.join(testing_dir, img_string+"Original_and_semantic_segmentation_{}.jpg".format(image_endings)))
Image.fromarray(orig_and_pan).save(fp=os.path.join(testing_dir, img_string+"Original_and_panoptic_segmentation_{}.jpg".format(image_endings)))
Image.fromarray(sem_and_pan_sideways).save(fp=os.path.join(testing_dir, img_string+"Semantic_and_panoptic_segmentation_{}_sideways.jpg".format(image_endings)))
Image.fromarray(inst_and_pan_sideways).save(fp=os.path.join(testing_dir, img_string+"Instance_and_panoptic_segmentation_{}_sideways.jpg".format(image_endings)))
Image.fromarray(inst_and_pan_ontop).save(fp=os.path.join(testing_dir, img_string+"Instance_and_panoptic_segmentation_{}_ontop.jpg".format(image_endings)))
Image.fromarray(orig_inst_sem_pan).save(fp=os.path.join(testing_dir, img_string+"Original_instance_semantic_panoptic_{}.jpg".format(image_endings)))


alpha_val = 0.5
title_font = 27
overlaid_title_string = " overlaid" if Using_Vitrolife else ""
fig=plt.figure(figsize=(15,15) if Using_Vitrolife else (15,9))
plt.subplot(2,2,1)
plt.imshow(orig_im)
plt.axis("off")
plt.title("Input image", fontsize=title_font)
plt.subplot(2,2,2)
plt.imshow(orig_im if Using_Vitrolife else sem_im)
if Using_Vitrolife:
    plt.imshow(sem_im, alpha=alpha_val)
plt.axis("off")
plt.title("Semantic segmentation"+overlaid_title_string, fontsize=title_font)
plt.subplot(2,2,3)
plt.imshow(orig_im if Using_Vitrolife else inst_im)
if Using_Vitrolife:
    plt.imshow(inst_im, alpha=alpha_val)
plt.axis("off")
plt.title("Instance segmentation"+overlaid_title_string, fontsize=title_font)
plt.subplot(2,2,4)
plt.imshow(orig_im if Using_Vitrolife else pan_im)
if Using_Vitrolife:
    plt.imshow(pan_im, alpha=alpha_val)
plt.axis("off")
plt.title("Panoptic segmentation"+overlaid_title_string, fontsize=title_font)
plt.tight_layout()
plt.show(block=False)
fig.savefig(os.path.join(testing_dir, img_string+"orig_sem_inst_pan_overlaid.jpg"), bbox_inches="tight")

