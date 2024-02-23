from typing import Tuple, Union
import math
import cv2
import numpy as np

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

roi = {
    0: ((30, 40), (30, 20)), # Right eye
    1: ((30, 40), (30, 20)),
    3: ((50, 40), (20, 50))
}

def visualize(image, detection_result) -> np.ndarray:
    """Draws bounding boxes and keypoints on the input image and return it.
    Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
    Returns:
    Image with bounding boxes.
    """
    annotated_image = image.copy()
    height, width, _ = image.shape
    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        # Draw keypoints
        # right eye, left eye, mouth
        keyponits = [0, 1, 3]
        for keypoint in keyponits:
            region = roi[keypoint]
            
            point = detection.keypoints[keypoint]
            keypoint_px = _normalized_to_pixel_coordinates(point.x, point.y, width, height)
            start_point = (keypoint_px[0] - region[0][0], keypoint_px[1] - region[0][1])
            end_point = (keypoint_px[0] + region[1][0], keypoint_px[1] + region[1][1])
            cv2.rectangle(annotated_image, start_point, end_point, ROI_COLOR, 3)
            # cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        category_name = '' if category_name is None else category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return annotated_image, detection
