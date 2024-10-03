# STEP 1: Import the necessary modules.
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


IMAGE_FILE = 'image.jpg'
# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path='/media/sombrali/HDD1/3d_object_detection/mediapipe/weights/exported_model_v2_200_epoch@cocofp/model_fp16.tflite')

options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)

detector = vision.ObjectDetector.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file(IMAGE_FILE)

# STEP 4: Detect objects in the input image.
detection_result = detector.detect(image)
print(detection_result)