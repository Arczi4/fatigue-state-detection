import cv2
import os
from tensorflow.keras.models import load_model
from common.constants import LEFT_EYE, MOUTH, RIGHT_EYE, ROI_COLOR
from common.helpers import get_detections, get_roi_from_image
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Define the text, font, and position
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255) # White color in BGR
thickness = 2

mouth_state_classes = ['no_yawn', 'yawn']
eye_state_classes = ['closed', 'opened']

eye_model_path = os.path.join('models', 'eye_classification.keras')
mouth_model_path = os.path.join('models', 'mouth_classification.keras')

eye_model = load_model(eye_model_path, compile=False)
eye_model.compile(run_eagerly=True)

mouth_model = load_model(mouth_model_path, compile=False)
mouth_model.compile(run_eagerly=True)





def get_video():
    # Open the default camera (usually the first one available)
    cap = cv2.VideoCapture(0)
    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Loop to capture frames from the camera
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Getting gray image
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Convert frame to MediaPipe image format (tf.Tensor)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        detection_result = get_detections(mp_image)
        
        rois = get_roi_from_image(frame, detection_result)

        # Right eye

        right_eye_roi = gray_frame[rois[RIGHT_EYE][0][1]: rois[RIGHT_EYE][1][1], rois[RIGHT_EYE][0][0]: rois[RIGHT_EYE][1][0]]
        img_array = tf.expand_dims(right_eye_roi, 0) # Create a batch
        
        right_eye_prediction = eye_model.predict(img_array)
        right_eye_score = tf.nn.softmax(right_eye_prediction[0])
        
        cv2.rectangle(frame, rois[RIGHT_EYE][0], rois[RIGHT_EYE][1], ROI_COLOR, 3)
        
        right_eye_classification_position = (50, 50)
        right_eye_classification_text = f'right eye {eye_state_classes[np.argmax(right_eye_score)]}'
        cv2.putText(frame, right_eye_classification_text, right_eye_classification_position, font, font_scale, font_color, thickness)
        
        # Left eye
        left_eye_roi = gray_frame[rois[LEFT_EYE][0][1]: rois[LEFT_EYE][1][1], rois[LEFT_EYE][0][0]: rois[LEFT_EYE][1][0]]
        img_array = tf.expand_dims(left_eye_roi, 0) # Create a batch
        
        left_eye_prediction = eye_model.predict(img_array)
        left_eye_score = tf.nn.softmax(left_eye_prediction[0])
        
        cv2.rectangle(frame, rois[LEFT_EYE][0], rois[LEFT_EYE][1], ROI_COLOR, 3)

        left_eye_classification_position = (50, 100)
        left_eye_classification_text = f'left eye {eye_state_classes[np.argmax(left_eye_score)]}'
        cv2.putText(frame, left_eye_classification_text, left_eye_classification_position, font, font_scale, font_color, thickness)
        
        
        # Mouth
        mouth_roi = gray_frame[rois[MOUTH][0][1]: rois[MOUTH][1][1], rois[MOUTH][0][0]: rois[MOUTH][1][0]]
        img_array = tf.expand_dims(mouth_roi, 0) # Create a batch
        
        mouth_prediction = mouth_model.predict(img_array)
        mouth_score = tf.nn.softmax(mouth_prediction[0])
        
        cv2.rectangle(frame, rois[MOUTH][0], rois[MOUTH][1], ROI_COLOR, 3)
        
        mouth_classification_position = (50, 150)
        mouth_classification_text = f'mouth {mouth_state_classes[np.argmax(mouth_score)]}'
        cv2.putText(frame, mouth_classification_text, mouth_classification_position, font, font_scale, font_color, thickness)
        
        print(f'right_eye_score: {right_eye_score}')
        print(f'left_eye_score: {left_eye_score}')
        print(f'mouth_score: {mouth_score}')
        
        # If frame is read correctly ret is True
        if not ret:
            print("Error: Cannot receive frame.")
            break

        # Display the frame
        cv2.imshow('Frame', frame)
        
        # Check if the user pressed the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        input()
        
    # Release the capture
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    get_video()
    