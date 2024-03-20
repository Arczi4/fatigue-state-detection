import cv2
from constants import DETECTOR
import numpy as np
import mediapipe as mp

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red
ROI_COLOR = (0, 255, 0)


def get_faces(image):
    detection_result = DETECTOR.detect(image)
    image = np.copy(image.numpy_view())
    
    annotated_image = image.copy()
    height, width, _ = image.shape
    
    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = (bbox.origin_x, bbox.origin_y)
        end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)
        
        # Getting single face from an image
        image_copy_face = image[start_point[1]: end_point[1], start_point[0]: end_point[0]]
        
        # Resizing image
        resized_image = cv2.resize(image_copy_face, (256, 256))
        
        # Changing to gray scale
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        
        cv2.imshow('Image', gray_image)
        cv2.waitKey(0)  # Wait for any key to be pressed
        cv2.destroyAllWindows()  # Close all OpenCV windows

image = mp.Image.create_from_file(r'D:\Studia II\Master Thesis\Code\my_work\data\train\mouth\no_yawn\1.jpg')
get_faces(image)