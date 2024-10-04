import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import cv2
import shutil
import tqdm

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image

if __name__ == '__main__':

    base_options = python.BaseOptions(model_asset_path='/media/sombrali/HDD1/3d_object_detection/mediapipe/weights/exported_model_v4_120_epoch@cocofp@aug/model_fp16.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                        score_threshold=0.4)
    detector = vision.ObjectDetector.create_from_options(options)


    cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)


    # cap = cv2.VideoCapture(0)

    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    #     detection_result = detector.detect(image)

    #     image_copy = np.copy(image.numpy_view())
    #     annotated_image = visualize(image_copy, detection_result)
    #     rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
    #     cv2.imshow("Object Detection", rgb_annotated_image)
    #     cv2.waitKey(1)
    # cv2.destroyAllWindows()


    coco_possible_fp = "/home/sombrali/coco_dataset/test2017"

    output_dir = "/media/sombrali/HDD1/3d_object_detection/mediapipe/dataset/v4/coco_test2017_false_positive/"
    os.makedirs(output_dir, exist_ok=True)
    for image in tqdm.tqdm(os.listdir(coco_possible_fp), total=len(os.listdir(coco_possible_fp))):
        fp_tag = False
        IMAGE_FILE = os.path.join(coco_possible_fp, image)

        image = mp.Image.create_from_file(IMAGE_FILE)

        detection_result = detector.detect(image)

        for detection in detection_result.detections:
            # bbox = detection.bounding_box
            for cat in detection.categories:
                cat_n = cat.category_name
                score = cat.score
                if score > 0.5:
                   fp_tag = True

        image_copy = np.copy(image.numpy_view())
        if fp_tag:
            shutil.copy2(IMAGE_FILE, output_dir)
        annotated_image = visualize(image_copy, detection_result)
        rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        cv2.imshow("Object Detection", rgb_annotated_image)
        cv2.waitKey(1)

    cv2.destroyAllWindows()