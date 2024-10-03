# This Python code converts a dataset in YOLO format into the COCO format. 
# The YOLO dataset contains images of bottles and the bounding box annotations in the 
# YOLO format. The COCO format is a widely used format for object detection datasets.

# The input and output directories are specified in the code. The categories for 
# the COCO dataset are also defined, with only one category for "bottle". A dictionary for the COCO dataset is initialized with empty values for "info", "licenses", "images", and "annotations".

# The code then loops through each image in the input directory. The dimensions 
# of the image are extracted and added to the COCO dataset as an "image" dictionary, 
# including the file name and an ID. The bounding box annotations for each image are 
# read from a text file with the same name as the image file, and the coordinates are 
# converted to the COCO format. The annotations are added to the COCO dataset as an 
# "annotation" dictionary, including an ID, image ID, category ID, bounding box coordinates,
# area, and an "iscrowd" flag.

# The COCO dataset is saved as a JSON file in the output directory.

import json
import os
from PIL import Image
import shutil

# Set the paths for the input and output directories
input_dir = '/media/sombrali/HDD1/dataset_generator/20240930-yolo'
output_dir = 'dataset/v4/headband-autogen-dataset-coco-faceside-20240930'

# Define the categories for the COCO dataset
# 0 for background
categories = [{"id": 1, "name": "Bruni-woband"}, {"id": 2, "name": "Elsa-woband"}, {"id": 3, "name": "WOFAnnaV2"}]
# Define the COCO dataset dictionary


images_base_dir = os.path.join(input_dir, 'images')
annotations_base_dir = os.path.join(input_dir, 'labels')
# Loop through the images in the input directory



for train_val in ['train', 'val']:
    os.makedirs(os.path.join(output_dir, train_val, 'images'), exist_ok=True)

    coco_dataset = {
        "info": {},
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": []
    }
    images_dir = os.path.join(images_base_dir, train_val)
    annotations_dir = os.path.join(annotations_base_dir, train_val)

    for image_i, image_file in enumerate(os.listdir(images_dir)):
        
        # Load the image and get its dimensions
        image_path = os.path.join(images_dir, image_file)
        shutil.copy2(image_path, os.path.join(output_dir, train_val, 'images'))

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
        with open(os.path.join(annotations_dir, f'{image_file.split(".")[0]}.txt')) as f:
            annotations = f.readlines()
        
        # Loop through the annotations and add them to the COCO dataset
        for ann_i, ann in enumerate(annotations):
            x, y, w, h = map(float, ann.strip().split()[1:])
            x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
            x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)
            ann_dict = {
                "id": ann_i,
                "image_id": image_i,
                "category_id": int(ann[0]) + 1,
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "area": (x_max - x_min) * (y_max - y_min),
                # "iscrowd": 0
            }
            coco_dataset["annotations"].append(ann_dict)

    # Save the COCO dataset to a JSON file
    with open(os.path.join(output_dir, train_val, f'labels.json'), 'w') as f:
        json.dump(coco_dataset, f)
        