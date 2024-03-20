from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

RIGHT_EYE = 'right_eye'
LEFT_EYE = 'left_eye'
MOUTH = 'mouth'

ROI_MAPPER = {
    RIGHT_EYE: 0,
    LEFT_EYE: 1,
    MOUTH: 3
}

ROI_SIZES = {
    0: ((30, 30), (30, 30)),  # Right eye
    1: ((30, 30), (30, 30)),  # Left eye
    3: ((50, 40), (20, 50)),  # Mouth
}

ROI_COLOR = (0, 255, 0)


# Detector
BASE_OPTIONS = python.BaseOptions(model_asset_path=os.path.join('models', 'blaze_face_short_range.tflite'))
OPTIONS = vision.FaceDetectorOptions(base_options=BASE_OPTIONS)
DETECTOR = vision.FaceDetector.create_from_options(OPTIONS)