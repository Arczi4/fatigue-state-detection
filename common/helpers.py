from typing import Tuple, Union
import math
import cv2
from typing import List, Union
from common.constants import LEFT_EYE, MOUTH, RIGHT_EYE, ROI_MAPPER, DETECTOR, ROI_SIZES
import numpy as np
import os
import mediapipe as mp

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red
ROI_COLOR = (0, 255, 0)


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def get_eyes(detection, image_width, image_height):
    roi_to_return = {}
    keyponits=[RIGHT_EYE, LEFT_EYE]
    for keypoint in keyponits:
        # Map keypoint to it's integer representation to match the output from detector
        keypoint_int = ROI_MAPPER[keypoint]
        
        region = ROI_SIZES[keypoint_int]
        point = detection.keypoints[keypoint_int]
        keypoint_px = _normalized_to_pixel_coordinates(point.x, point.y, image_width, image_height)
        start_point = (keypoint_px[0] - region[0][0], keypoint_px[1] - region[0][1])
        end_point = (keypoint_px[0] + region[1][0], keypoint_px[1] + region[1][1])
        roi_to_return[keypoint] = ((start_point, end_point))




def get_faces(image):
    detection_result = DETECTOR.detect(image)
    image = np.copy(image.numpy_view())
    
    annotated_image = image.copy()
    
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
        face_gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        
        return annotated_image, face_gray_image
        
        # cv2.imshow('Image', gray_image)
        # cv2.waitKey(0)  # Wait for any key to be pressed
        # cv2.destroyAllWindows()  # Close all OpenCV windows

# image = mp.Image.create_from_file(r'D:\Studia II\Master Thesis\Code\my_work\data\train\mouth\no_yawn\1.jpg')
# get_faces(image)
        

def get_roi_from_image(image, detection_result, keyponits=[RIGHT_EYE, LEFT_EYE, MOUTH]) -> List[Union[list, dict]]:
    """Get region of intrests specified in keypoints argument.
    Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
    keypoints: Default value set to right eye, left eye, mouth
    Returns:
    List of annotated image with boudning boxes and dict of ROI.
    """
    # annotated_image = image.copy()
    height, width, _ = image.shape
    roi_to_return = {}
    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        # cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        # Draw keypoints
        for keypoint in keyponits:
            # Map keypoint to it's integer representation to match the output from detector
            keypoint_int = ROI_MAPPER[keypoint]
            
            region = ROI_SIZES[keypoint_int]
            point = detection.keypoints[keypoint_int]
            keypoint_px = _normalized_to_pixel_coordinates(point.x, point.y, width, height)
            start_point = (keypoint_px[0] - region[0][0], keypoint_px[1] - region[0][1])
            end_point = (keypoint_px[0] + region[1][0], keypoint_px[1] + region[1][1])
            roi_to_return[keypoint] = ((start_point, end_point))
            # cv2.rectangle(annotated_image, start_point, end_point, ROI_COLOR, 3)

        # # Draw label and score
        # category = detection.categories[0]
        # category_name = category.category_name
        # category_name = '' if category_name is None else category_name
        # probability = round(category.score, 2)
        # result_text = category_name + ' (' + str(probability) + ')'
        # text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        # cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
        #             FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

        return roi_to_return


def get_detections(image):
    detection_result = DETECTOR.detect(image)
    
    return detection_result