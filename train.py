import os
import tensorflow as tf
from tensorflow.keras import layers
assert tf.__version__.startswith('2')

from mediapipe_model_maker import object_detector
from mediapipe_model_maker import quantization


# my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
# tf.config.experimental.set_visible_devices(devices = my_devices, device_type='CPU')



def prepare_dataset(train_dataset_path, validation_dataset_path):

    train_data = object_detector.Dataset.from_coco_folder(train_dataset_path, cache_dir="/tmp/od_data/v4_concated_fp/train")
    validation_data = object_detector.Dataset.from_coco_folder(validation_dataset_path, cache_dir="/tmp/od_data/v4_concated_fp/validation")
    print("train_data size: ", train_data.size)
    print("validation_data size: ", validation_data.size)

    return train_data, validation_data

def train(train_data, validation_data, epochs):

    spec = object_detector.SupportedModels.MOBILENET_V2_I320


    hparams = object_detector.HParams(batch_size=32, 
                                      learning_rate = 0.01, 
                                      cosine_decay_epochs = epochs, 
                                      cosine_decay_alpha = 0.00001,
                                      epochs = epochs, 
                                    #   export_dir=f'weights/exported_model_v3_{epochs}_epoch@cocofp')
                                      export_dir=f'weights/exported_model_v4_{epochs}_epoch@cocofp@aug')
    
    options = object_detector.ObjectDetectorOptions(
        supported_model=spec,
        hparams=hparams
    )

    model = object_detector.ObjectDetector.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options,
    )

    loss, coco_metrics = model.evaluate(validation_data, batch_size=4)
    print(f"Validation loss: {loss}")
    print(f"Validation coco metrics: {coco_metrics}")

    model.export_model()
    return model

def quantize_fp16(model):
    quantization_config = quantization.QuantizationConfig.for_float16()
    model.restore_float_ckpt()
    model.export_model(model_name="model_fp16.tflite", quantization_config=quantization_config)


def quantize_int8(model, train_data, validation_data, epochs):
# qat_hparams = object_detector.QATHParams(learning_rate = 0.2, batch_size = 16, epochs = 20, decay_steps=6, decay_rate=0.96)
    qat_hparams = object_detector.QATHParams(batch_size=32, learning_rate = 0.01, decay_steps = 20, decay_rate = 0.96, epochs = epochs)
    model.restore_float_ckpt()
    model.quantization_aware_training(train_data, validation_data, qat_hparams=qat_hparams)
    qat_loss, qat_coco_metrics = model.evaluate(validation_data)
    print(f"QAT validation loss: {qat_loss}")
    print(f"QAT validation coco metrics: {qat_coco_metrics}")
    model.export_model('model_int8_qat.tflite')

if __name__ == '__main__':


    epochs_for_training = 120

    train_data, validation_data = prepare_dataset(
        train_dataset_path="/media/sombrali/HDD1/3d_object_detection/mediapipe/dataset/v4/concated_coco_v4/train",
        validation_dataset_path="/media/sombrali/HDD1/3d_object_detection/mediapipe/dataset/v4/concated_coco_v4/val"
    )
    model = train(train_data, validation_data, epochs = epochs_for_training)
    quantize_fp16(model)
    # quantize_int8(model, train_data, validation_data, epochs = epochs_for_training)