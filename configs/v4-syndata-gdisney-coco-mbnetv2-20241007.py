from mediapipe_model_maker import object_detector

model_spec = object_detector.SupportedModels.MOBILENET_V2_I320

epochs = 120

cache_dataset_name = 'v4-syndata-gdisney-coco-20241006'
train_dataset_path = '/media/sombrali/HDD1/3d_object_detection/mediapipe/dataset/v4/v4-syndata-gdisney-coco-20241006/train'
validation_dataset_path = '/media/sombrali/HDD1/3d_object_detection/mediapipe/dataset/v4/v4-syndata-gdisney-coco-20241006/val'



# save path would be {weights_save_dir}/{config_name}
weights_save_dir = '/media/sombrali/HDD1/3d_object_detection/mediapipe/weights'

export_fp16 = True



### training 
batch_size = 32
learning_rate = 0.01
cosine_decay_epochs = epochs
cosine_decay_alpha = 0.00001
