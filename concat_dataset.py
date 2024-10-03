import json
import os
import shutil

datasets = [
    "/media/sombrali/HDD1/3d_object_detection/mediapipe/dataset/v4/headband-autogen-dataset-coco-faceside-20240930",
    "/media/sombrali/HDD1/3d_object_detection/mediapipe/dataset/v3/coco_test2017_false_positive"
]


concated_dataset = "/media/sombrali/HDD1/3d_object_detection/mediapipe/dataset/v4/concated_coco_v4"

train_val = ['train', 'val']

for tv in train_val:
    labels = []

    os.makedirs(f"{concated_dataset}/{tv}", exist_ok=True)
    os.makedirs(f"{concated_dataset}/{tv}/images", exist_ok=True)

    for dataset in datasets:
        with open(f"{dataset}/{tv}/labels.json", "r") as f:
            # labels.extend(json.load(f))
            labels.append(json.load(f))
        shutil.copytree(f"{dataset}/{tv}/images", f"{concated_dataset}/{tv}/images", dirs_exist_ok=True)


    current_label = labels[0]
    global_image_id = len(current_label['images'])
    global_annotation_id = len(current_label['annotations'])



    for label in labels[1: ]:

        cat_id_mapper = {}
        img_id_mapper = {}

        for category in label['categories']:
            cat_id_mapper[category['id']] = global_annotation_id + 1
            category['id'] = global_annotation_id + 1
            current_label['categories'].append(category)
            global_annotation_id += 1

        for image in label['images']:
            img_id_mapper[image['id']] = global_image_id + 1
            image['id'] = global_image_id
            global_image_id += 1


        for annotation in label['annotations']:
            annotation['image_id'] = img_id_mapper[annotation['image_id']]
            annotation['category_id'] = cat_id_mapper[annotation['category_id']]
        

        current_label['categories'].extend(label['categories'])
        current_label['images'].extend(label['images'])
        current_label['annotations'].extend(label['annotations'])
    
    with open(f"{concated_dataset}/{tv}/labels.json", "w") as f:
        json.dump(current_label, f)