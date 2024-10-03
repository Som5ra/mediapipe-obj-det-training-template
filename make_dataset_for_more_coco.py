

import json
import os
from PIL import Image
import shutil

# 0 for background
categories = []
# Define the COCO dataset dictionary

input_dir = "/media/sombrali/HDD1/3d_object_detection/mediapipe/dataset/v3/coco_test2017_false_positive/val"
images_base_dir = os.path.join(input_dir, 'images')
# Loop through the images in the input directory


coco_dataset = {
    "info": {},
    "licenses": [],
    "categories": categories,
    "images": [],
    "annotations": []
}
images_dir = images_base_dir

for image_i, image_file in enumerate(os.listdir(images_dir)):
    
    # Load the image and get its dimensions
    image_path = os.path.join(images_dir, image_file)

    image = Image.open(image_path)
    width, height = image.size
    
    # Add the image to the COCO dataset
    image_dict = {
        "id": image_i,
        "width": width,
        "height": height,
        "file_name": image_file
    }
    coco_dataset["images"].append(image_dict)
    
    # Load the bounding box annotations for the image
    
    # Loop through the annotations and add them to the COCO dataset

# Save the COCO dataset to a JSON file
with open(os.path.join(input_dir, f'labels.json'), 'w') as f:
    json.dump(coco_dataset, f)
    