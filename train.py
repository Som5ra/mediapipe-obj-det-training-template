import os
import tensorflow as tf
import time
from datetime import datetime

from tensorflow.keras import layers
assert tf.__version__.startswith('2')

from mediapipe_model_maker import object_detector
from mediapipe_model_maker import quantization

from utils.util_logging import create_logger
from utils.util_config import load_config

import argparse

# my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
# tf.config.experimental.set_visible_devices(devices = my_devices, device_type='CPU')



def prepare_dataset(train_dataset_path, validation_dataset_path, cache_name):

    train_data = object_detector.Dataset.from_coco_folder(train_dataset_path, cache_dir=f"/tmp/od_data/{cache_name}/train")
    validation_data = object_detector.Dataset.from_coco_folder(validation_dataset_path, cache_dir=f"/tmp/od_data/{cache_name}/validation")
    print("train_data size: ", train_data.size)
    print("validation_data size: ", validation_data.size)

    return train_data, validation_data

def train(train_data, validation_data, config, save_at):

    spec = config.model_spec


    hparams = object_detector.HParams(batch_size=config.batch_size, 
                                      learning_rate = config.learning_rate, 
                                      cosine_decay_epochs = config.cosine_decay_epochs, 
                                      cosine_decay_alpha = config.cosine_decay_alpha,
                                      epochs = epochs, 
                                      export_dir=f'{save_at}')
    
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
    qat_hparams = object_detector.QATHParams(learning_rate = 0.2, batch_size = 16, epochs = 20, decay_steps=6, decay_rate=0.96)
    # qat_hparams = object_detector.QATHParams(batch_size=32, learning_rate = 0.01, decay_steps = 20, decay_rate = 0.96, epochs = epochs)
    model.restore_float_ckpt()
    model.quantization_aware_training(train_data, validation_data, qat_hparams=qat_hparams)
    qat_loss, qat_coco_metrics = model.evaluate(validation_data)
    print(f"QAT validation loss: {qat_loss}")
    print(f"QAT validation coco metrics: {qat_coco_metrics}")
    model.export_model('model_int8_qat.tflite')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str, help="Path to the config .py file")
    args = parser.parse_args()

    config = load_config(args.config_file)

    print(config)
    # Now you can access the variables like config['epochs']
    epochs = config.epochs
    train_dataset_path = config.train_dataset_path
    validation_dataset_path = config.validation_dataset_path
    export_fp16 = config.export_fp16

    weights_save_dir = os.path.join(config.weights_save_dir, os.path.basename(args.config_file).split('.')[0] + '-' + datetime.now().strftime("%Y-%m-%d"))

    logger = create_logger(name='[training_logger]')

    logger.info(f"Training started with epochs: {epochs}")
    logger.info(f"Train dataset path: {train_dataset_path}")
    logger.info(f"Validation dataset path: {validation_dataset_path}")
    logger.info(f"Exporting fp16: {export_fp16}")

    start_time = time.time()

    train_data, validation_data = prepare_dataset(
        train_dataset_path=train_dataset_path,
        validation_dataset_path=validation_dataset_path,
        cache_name = config.cache_dataset_name
    )
    
    dataset_prepare_time = time.time()
    logger.info(f"Dataset preparation time: {dataset_prepare_time - start_time} seconds")

    model = train(train_data, validation_data, config = config, save_at = weights_save_dir)
    training_time = time.time()
    logger.info(f"Training time: {training_time - dataset_prepare_time} seconds")

    if export_fp16:
        quantize_fp16(model)
        export_fp16_time = time.time()
        logger.info(f"Exporting fp16 time: {export_fp16_time - training_time} seconds")

    logger.info(f"Summary:")
    logger.info(f"Dataset preparation time: {dataset_prepare_time - start_time} seconds")
    logger.info(f"Training time: {training_time - dataset_prepare_time} seconds")
    if export_fp16:
        logger.info(f"Exporting fp16 time: {export_fp16_time - training_time} seconds")
    logger.info(f"Total time: {time.time() - start_time} seconds")
    # quantize_int8(model, train_data, validation_data, epochs = epochs_for_training)