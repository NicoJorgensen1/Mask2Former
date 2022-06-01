import os
from cv2 import COLOR_HSV2RGB, COLOR_RGB2HSV 
import numpy as np 
import shutil 
import pickle 
import sys 
import re 
import torch 
import copy 
import torchvision
import detectron2
from detectron2.config import get_cfg 
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from matplotlib import pyplot as plt 
from PIL import Image
import PIL 
import cv2
from scipy import ndimage 

# Save dictionary
def save_dictionary(dictObject, save_folder, dictName):                                                         # Function to save a dict in the specified folder 
    dict_file = open(os.path.join(save_folder, dictName+".pkl"), "wb")                                          # Opens a pickle for saving the dictionary 
    pickle.dump(dictObject, dict_file)                                                                          # Saves the dictionary 
    dict_file.close()                                                                                           # Close the pickle again  

def scale_img(img):
    img = img.astype(float)
    img = np.subtract(img, np.min(img))
    img = np.divide(img, np.max(img))
    img = np.multiply(img, 255)
    img = img.astype(np.uint8)
    return img


def fancy_pca(img, alpha_std=0.1):
    '''
    INPUTS:
    img:  numpy array with (h, w, rgb) shape, as ints between 0-255)
    alpha_std:  how much to perturb/scale the eigen vecs and vals
                the paper used std=0.1
    RETURNS:
    numpy image-like array as float range(0, 1)
    NOTE: Depending on what is originating the image data and what is receiving
    the image data returning the values in the expected form is very important
    in having this work correctly. If you receive the image values as UINT 0-255
    then it's probably best to return in the same format. (As this
    implementation does). If the image comes in as float values ranging from
    0.0 to 1.0 then this function should be modified to return the same.
    Otherwise this can lead to very frustrating and difficult to troubleshoot
    problems in the image processing pipeline.
    This is 'Fancy PCA' from:
    # http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    #######################
    #### FROM THE PAPER ###
    #######################
    "The second form of data augmentation consists of altering the intensities
    of the RGB channels in training images. Specifically, we perform PCA on the
    set of RGB pixel values throughout the ImageNet training set. To each
    training image, we add multiples of the found principal components, with
    magnitudes proportional to the corresponding eigenvalues times a random
    variable drawn from a Gaussian with mean zero and standard deviation 0.1.
    Therefore to each RGB image pixel Ixy = [I_R_xy, I_G_xy, I_B_xy].T
    we add the following quantity:
    [p1, p2, p3][α1λ1, α2λ2, α3λ3].T
    Where pi and λi are ith eigenvector and eigenvalue of the 3 × 3 covariance
    matrix of RGB pixel values, respectively, and αi is the aforementioned
    random variable. Each αi is drawn only once for all the pixels of a
    particular training image until that image is used for training again, at
    which point it is re-drawn. This scheme approximately captures an important
    property of natural images, namely, that object identity is invariant to
    change."
    ### END ###############
    Other useful resources for getting this working:
    # https://groups.google.com/forum/#!topic/lasagne-users/meCDNeA9Ud4
    # https://gist.github.com/akemisetti/ecf156af292cd2a0e4eb330757f415d2
    '''

    orig_img = img.astype(float).copy()

    img = img / 255.0  # rescale to 0 to 1 range

    # flatten image to columns of RGB
    img_rs = img.reshape(-1, 3)
    # img_rs shape (640000, 3)

    # center mean
    img_centered = img_rs - np.mean(img_rs, axis=0)

    # paper says 3x3 covariance matrix
    img_cov = np.cov(img_centered, rowvar=False)

    # eigen values and eigen vectors
    eig_vals, eig_vecs = np.linalg.eigh(img_cov)

#     eig_vals [0.00154689 0.00448816 0.18438678]

#     eig_vecs [[ 0.35799106 -0.74045435 -0.56883192]
#      [-0.81323938  0.05207541 -0.57959456]
#      [ 0.45878547  0.67008619 -0.58352411]]

    # sort values and vector
    sort_perm = eig_vals[::-1].argsort()
    eig_vals[::-1].sort()
    eig_vecs = eig_vecs[:, sort_perm]

    # get [p1, p2, p3]
    m1 = np.column_stack((eig_vecs))
    

    # get 3x1 matrix of eigen values multiplied by random variable draw from normal
    # distribution with mean of 0 and standard deviation of 0.1
    m2 = np.zeros((3, 1))
    # according to the paper alpha should only be draw once per augmentation (not once per channel)
    if alpha_std is None:
        alpha = np.random.normal(0, alpha_std)
    else:
        alpha = alpha_std

    # broad cast to speed things up
    m2[:, 0] = alpha * eig_vals[:]

    # this is the vector that we're going to add to each pixel in a moment
    add_vect = np.matrix(m1) * np.matrix(m2)

    for idx in range(3):   # RGB
        orig_img[..., idx] += add_vect[idx]

    # for image processing it was found that working with float 0.0 to 1.0
    # was easier than integers between 0-255
    # orig_img /= 255.0
    orig_img = np.clip(orig_img, 0.0, 255.0)

    # orig_img *= 255
    orig_img = orig_img.astype(np.uint8)

    # about 100x faster after vectorizing the numpy, it will be even faster later
    # since currently it's working on full size images and not small, square
    # images that will be fed in later as part of the post processing before being
    # sent into the model
#     print("elapsed time: {:2.2f}".format(time.time() - start_time), "\n")

    return orig_img



# Find parent directory 
Mask2Former_dir = os.path.join("/mnt", "c", "Users", "Nico-", "Documents", "Python_Projects", "Mask2Former")                                                                # Home WSL
if not os.path.isdir(Mask2Former_dir):
    Mask2Former_dir = os.path.join("C:\\", Mask2Former_dir.split(os.path.sep, 1)[1])                                                     # Home windows computer
if not os.path.isdir(Mask2Former_dir):
    Mask2Former_dir = os.path.join("/mnt", "c", "Users", "wd974261", "Documents", "Python", "Mask2Former")                               # Work WSL
if not os.path.isdir(Mask2Former_dir):
    Mask2Former_dir = os.path.join("C:\\", Mask2Former_dir.split(os.path.sep, 1)[1])                                                     # Work windows computer
if not os.path.isdir(Mask2Former_dir):
    Mask2Former_dir = os.path.join("/home", "neal", "Panoptic_segmentation_using_deep_neural_networks", "Repositories", "Mask2Former")   # Larac server
if not os.path.isdir(Mask2Former_dir):
    Mask2Former_dir = os.path.join("/mnt", "home_shared", Mask2Former_dir.split(os.path.sep, 2)[2])                                      # Balder server
assert os.path.isdir(Mask2Former_dir), "The Mask2Former directory doesn't exist in the chosen location"
sys.path.append(Mask2Former_dir)                                                                                # Add Mask2Former directory to PATH
sys.path.append(os.path.join(Mask2Former_dir, "Custom_functions"))                                              # Add Custom_functions directory to PATH
os.chdir(os.path.join(Mask2Former_dir, "Custom_functions"))                                                     # Switch the current directory to the Custom_functions directory
sys.path.append(os.path.join(Mask2Former_dir, "tools"))
ade20k_output_folder = os.path.join(Mask2Former_dir, "ade20k_outputs")
sys.path.append(ade20k_output_folder)
dataset_dir = os.path.join("/mnt", "c", "Users", "Nico-", "OneDrive - Aarhus Universitet", "Alting", "Biomedicinsk Teknologi", "5. semester", "Speciale", "Datasets")                 # Home WSL
if not os.path.isdir(dataset_dir):
    dataset_dir = os.path.join("C:\\", dataset_dir.split(os.path.sep,1)[1])                                                                  # Home windows computer
if not os.path.isdir(dataset_dir):
    dataset_dir = os.path.join("/mnt", "c", "Users", "wd974261", "Documents", "Python", "Datasets")                                          # Work WSL
if not os.path.isdir(dataset_dir):
    dataset_dir = os.path.join("C:\\", dataset_dir.split(os.path.sep,1)[1])                                                                  # Work windows computer
if not os.path.isdir(dataset_dir):
    dataset_dir = os.path.join("/home", "neal", "Panoptic_segmentation_using_deep_neural_networks", "Datasets")                              # Larac server
if not os.path.isdir(dataset_dir):
    dataset_dir = os.path.join("/mnt", "home_shared", dataset_dir.split(os.path.sep, 2)[2])                                                  # Balder server
assert os.path.isdir(dataset_dir), "The dataset directory doesn't exist in the chosen location"
os.environ["DETECTRON2_DATASETS"] = dataset_dir
os.chdir(ade20k_output_folder)

Vitrolife_dataset_dir = os.path.join(dataset_dir, "Vitrolife_dataset")
used_im_name = "30a73399b12b9c964815e1f71b0f27554b1b9203492fc792640078522426b1fe_W02"
used_im_path = [os.path.join(Vitrolife_dataset_dir, "raw_images", x) for x in os.listdir(os.path.join(Vitrolife_dataset_dir, "raw_images")) if used_im_name in x]
assert len(used_im_path) == 1, "Only one raw image with the chosen name can be accepted"
assert os.path.isfile(used_im_path[0]), "The chosen image file path has to exist"

# Original image 
vitrolife_img = np.asarray(Image.open(used_im_path[0]))

border_size = 15
border_col = (150, 0, 255)
border_horizontal = np.multiply(np.ones((vitrolife_img.shape[0], border_size, 3)).astype(np.uint8), border_col).astype(np.uint8) 
border_vertical = np.multiply(np.ones((border_size, vitrolife_img.shape[1]*2+border_size, 3)).astype(np.uint8), border_col).astype(np.uint8) 

# Brightness augmentation from [0.8, 1.5]
vitrolife_random_brightness_list = list()
brightness_factors = np.round(np.linspace(start=0.7, stop=1.5, num=4, endpoint=True),2).tolist()
for brightness_factor in brightness_factors:
    vitrolife_random_brightness_list.append(torch.reshape(torchvision.transforms.ColorJitter(brightness=(brightness_factor,brightness_factor)).forward(
            torch.reshape(torch.from_numpy(copy.deepcopy(vitrolife_img)), shape=vitrolife_img.shape[::-1])), shape=vitrolife_img.shape).numpy())
vitrolife_brightness_imgs = cv2.vconcat([cv2.hconcat([vitrolife_random_brightness_list[0], border_horizontal, vitrolife_random_brightness_list[1]]),
                            border_vertical, cv2.hconcat([vitrolife_random_brightness_list[2], border_horizontal, vitrolife_random_brightness_list[3]])])

# Random lightning of [0.7]
vitrolife_random_light_list = list()
random_PCA_scaling_values = np.round(np.linspace(start=-25, stop=25, num=4, endpoint=True),2).tolist()
for kk, random_value in enumerate(random_PCA_scaling_values):
    vitrolife_random_light_list.append(fancy_pca(img=copy.deepcopy(vitrolife_random_brightness_list[kk]), alpha_std=random_value))
vitrolife_random_light_imgs = cv2.vconcat([cv2.hconcat([vitrolife_random_light_list[0], border_horizontal, vitrolife_random_light_list[1]]),
                            border_vertical, cv2.hconcat([vitrolife_random_light_list[2], border_horizontal, vitrolife_random_light_list[3]])])


# Random contrast of [0.7, 1.3]
vitrolife_random_contrast_list = list()
random_contrast_values = np.round(np.linspace(start=0.7, stop=1.3, num=4, endpoint=True),2).tolist()
for kk, contrast_value in enumerate(random_contrast_values):
    vitrolife_random_contrast_list.append(torch.reshape(torchvision.transforms.ColorJitter(contrast=(contrast_value,contrast_value)).forward(
            torch.reshape(torch.from_numpy(copy.deepcopy(vitrolife_random_light_list[kk])), shape=vitrolife_img.shape[::-1])), shape=vitrolife_img.shape).numpy())
vitrolife_contrast_imgs = cv2.vconcat([cv2.hconcat([vitrolife_random_contrast_list[0], border_horizontal, vitrolife_random_contrast_list[1]]),
                            border_vertical, cv2.hconcat([vitrolife_random_contrast_list[2], border_horizontal, vitrolife_random_contrast_list[3]])])


# Random saturation of [0.85, 1.15]
vitrolife_random_saturation_list = list()
random_saturation_values = np.round(np.linspace(start=0.85, stop=1.15, num=4, endpoint=True),2).tolist()
for kk, saturation_value in enumerate(random_saturation_values):
    img = copy.deepcopy(vitrolife_random_contrast_list[kk])
    HSV = cv2.cvtColor(src=img, code=COLOR_RGB2HSV)
    H,S,V = cv2.split(m=HSV)
    S2 = np.multiply(S, saturation_value)
    S2 = np.clip(S2, a_min=0, a_max=255).astype(np.uint8)
    HSV2 = cv2.merge((H,S2,V))
    img2 = cv2.cvtColor(src=HSV2, code=COLOR_HSV2RGB)
    vitrolife_random_saturation_list.append(img2)
vitrolife_saturation_imgs = cv2.vconcat([cv2.hconcat([vitrolife_random_saturation_list[0], border_horizontal, vitrolife_random_saturation_list[1]]),
                            border_vertical, cv2.hconcat([vitrolife_random_saturation_list[2], border_horizontal, vitrolife_random_saturation_list[3]])])


# first_pixel = vitrolife_img[0,0,:]
# print("The mean value is {:.3f} with a std of {:.3f}. The first pixel has values {}\n".format(np.mean(vitrolife_img), np.std(vitrolife_img), vitrolife_img[0,0,:]))
# for item, saturation_value in zip(vitrolife_random_saturation_list, random_saturation_values):
#     print("The mean value is {:.3f} with a std of {:.3f}. The first pixel has values {}.".format(np.mean(item), np.std(item), item[0,0,:]))
#     print("")


# Random flipping of horizontal and/or vertical with 0.25 pct probability
vitrolife_random_flipping_list = [vitrolife_random_saturation_list[0], np.fliplr(vitrolife_random_saturation_list[1]), np.flipud(vitrolife_random_saturation_list[2]), np.fliplr(np.flipud(vitrolife_random_saturation_list[3]))]
vitrolife_flipping_imgs = cv2.vconcat([cv2.hconcat([vitrolife_random_flipping_list[0], border_horizontal, vitrolife_random_flipping_list[1]]),
                            border_vertical, cv2.hconcat([vitrolife_random_flipping_list[2], border_horizontal, vitrolife_random_flipping_list[3]])])




# Random rotation in the range [-45, 45] degrees 
vitrolife_random_rotation_list = list()
random_rotation_values = np.round(np.linspace(start=-45, stop=45, num=4, endpoint=True),2).tolist()
for kk, rotation_value in enumerate(random_rotation_values):
    vitrolife_random_rotation_list.append(cv2.resize(ndimage.rotate(copy.deepcopy(vitrolife_random_flipping_list[kk]), rotation_value), interpolation=cv2.INTER_LINEAR, dsize=(vitrolife_img.shape[0],vitrolife_img.shape[1])))
vitrolife_rotation_imgs = cv2.vconcat([cv2.hconcat([vitrolife_random_rotation_list[0], border_horizontal, vitrolife_random_rotation_list[1]]),
                            border_vertical, cv2.hconcat([vitrolife_random_rotation_list[2], border_horizontal, vitrolife_random_rotation_list[3]])])



# Random cropping of [0.80H, 0.80W]
vitrolife_random_cropping_list = list()
random_cropping_values = np.divide(np.floor(np.random.uniform(low=0, high=20, size=(4,2))),100)
for kk, item in enumerate(random_cropping_values):
    used_img = copy.deepcopy(vitrolife_random_flipping_list[kk])
    start_bottom = int(used_img.shape[0] * item[1])
    stop_top = int(used_img.shape[0] * (item[1] + 0.8))
    start_left = int(used_img.shape[1] * item[0])
    stop_right = int(used_img.shape[1] * (item[0] + 0.8))
    cropped_image = copy.deepcopy(used_img)[start_bottom:stop_top, start_left:stop_right]
    vitrolife_random_cropping_list.append(cv2.resize(src=cropped_image, dsize=vitrolife_img.shape[:-1], interpolation=cv2.INTER_LINEAR))
vitrolife_cropping_imgs = cv2.vconcat([cv2.hconcat([vitrolife_random_cropping_list[0], border_horizontal, vitrolife_random_cropping_list[1]]),
                            border_vertical, cv2.hconcat([vitrolife_random_cropping_list[2], border_horizontal, vitrolife_random_cropping_list[3]])])



img_size, n_rows, n_cols, ax_count = 4, 2, 4, 0
fig = plt.figure(figsize=(n_cols*img_size, n_rows*img_size))
titles = ["Original image", "Brightness enhanced with\na factor {}".format(brightness_factors), "PCA color augmentation\nwith values {}".format(np.round(np.divide(random_PCA_scaling_values,12.5),2).tolist()),
            "Contrast enhanced with\na factor {}".format(random_contrast_values), "Saturation enhanced with\na factor {}".format(random_saturation_values),
            "Horizontal + vertical flips", "Rotation of\n{} degrees".format([int(x) for x in random_rotation_values]), "Cropping of [0.80H, 0.80W] pixels"]
img_list = [vitrolife_img, vitrolife_brightness_imgs, vitrolife_random_light_imgs, vitrolife_contrast_imgs, vitrolife_saturation_imgs, vitrolife_flipping_imgs, vitrolife_rotation_imgs, vitrolife_cropping_imgs]

for row in range(n_rows):
    for col in range(n_cols):
        if ax_count >= len(img_list):
            break 
        img_used = copy.deepcopy(img_list[ax_count])
        if ax_count >= 1:
            img_used = cv2.putText(img_used, "{}A)".format(ax_count+1), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0),3)
            img_used = cv2.putText(img_used, "{}B)".format(ax_count+1), (10+border_size+vitrolife_img.shape[0],60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0),3)
            img_used = cv2.putText(img_used, "{}C)".format(ax_count+1), (10, border_size+vitrolife_img.shape[0]+60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0),3)
            img_used = cv2.putText(img_used, "{}D)".format(ax_count+1), (10+border_size+vitrolife_img.shape[0],border_size+vitrolife_img.shape[0]+60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0),3)
        else:
            img_used = cv2.putText(img_used, "{}A,B,C,D)".format(ax_count+1), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2)
        plt.subplot(n_rows, n_cols, ax_count+1)
        plt.imshow(scale_img(img_used), cmap="gray")
        plt.axis("off")
        plt.title(label=titles[ax_count], fontsize=16)
        ax_count += 1
fig.tight_layout()
fig.show()
fig.savefig(os.path.join(ade20k_output_folder, "Data_augmentation_used.jpg"), bbox_inches="tight")


# fig = plt.figure(figsize=(2*img_size, 2*img_size))
# ax_count = 0
# for row in range(2):
#     for col in range(2):
#         if ax_count >= 4:
#             break 
#         plt.subplot(2, 2, ax_count+1)
#         plt.imshow(vitrolife_random_saturation_list[ax_count], cmap="gray")
#         plt.axis("off")
#         plt.title("Saturation enhanced\nwith a factor {}".format(random_saturation_values[ax_count]), fontsize=16)
#         ax_count += 1
# fig.tight_layout()
# fig.show()



