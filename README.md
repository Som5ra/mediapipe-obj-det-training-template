### mediapipe-obj-det-training-template

```
Weights Download: 

https://thisisgusto-my.sharepoint.com/my?id=%2Fpersonal%2Fsombra%5Fli%5Fthisisgusto%5Fcom%2FDocuments%2FDisney%2DModel%2DTraining 
```


### Modified Mediapipe Object Detection Part
Replace the mediapipe-model-maker with folder: mediapipr-model-maker-modified
It has features:
    1. Image Augmentation: refer to [`this is code`](https://github.com/Sombraa711/mediapipe-obj-det-training-template/blob/main/mediapipe-model-maker-modified/python/vision/object_detector/preprocessor.py#L100-L112)
    2. Enable Training with Negative Samples: refer to [`this is code`](https://github.com/Sombraa711/mediapipe-obj-det-training-template/blob/main/mediapipe-model-maker-modified/python/vision/object_detector/preprocessor.py#L100-L112)
    3. Save intermediate checkpoints: refer to [`this is code`](https://github.com/Sombraa711/mediapipe-obj-det-training-template/blob/main/mediapipe-model-maker-modified/python/vision/object_detector/object_detector.py#L116-L130)