from mediapipe_model_maker import object_detector

spec = object_detector.SupportedModels.MOBILENET_MULTI_AVG_I384

epochs = 120

cache_dataset_name = 'v4-syndata-gdisney-coco-20241006'
train_dataset_path = '/mnt/HDD_1/mediapipe-obj-det-training-template/dataset/v4/v4-syndata-gdisney-coco-20241006/train'
validation_dataset_path = '/mnt/HDD_1/mediapipe-obj-det-training-template/dataset/v4/v4-syndata-gdisney-coco-20241006/val'



# save path would be {weights_save_dir}/{config_name}
weights_save_dir = '/mnt/HDD_1/mediapipe-obj-det-training-template/weights'

export_fp16 = True



### training 
batch_size=32
learning_rate = 0.01
cosine_decay_epochs = epochs
cosine_decay_alpha = 0.00001
