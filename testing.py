import os 
import numpy as np 
from matplotlib import pyplot as plt 
import cv2 
from detectron2.data import DatasetCatalog, MetadataCatalog
from PIL import Image 
import sys 

Mask2Former_dir = os.path.join("/mnt", "c", "Users", "Nico-", "Documents", "Python_Projects", "Mask2Former")
dataset_dir = os.path.join("/mnt", "c", "Users", "Nico-", "OneDrive - Aarhus Universitet", "Biomedicinsk Teknologi", "5. semester", "Speciale", "Datasets")
mapillary_vistas_train_dataset = os.path.join(dataset_dir, "mapillary_vistas", "training", "panoptic")


pan_img = Image.open(os.path.join(mapillary_vistas_train_dataset, "__CRyFzoDOXn6unQ6a3DnQ.png"))
pan_img_numpy = np.asarray(pan_img)


plt.imshow(pan_img_numpy)
plt.show(block=False)



