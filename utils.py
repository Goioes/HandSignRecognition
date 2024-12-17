import os
import math
import numpy as np
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import GestureRecognizerOptions, GestureRecognizer, RunningMode

def load_gesture_recognizer(recognizer_name, running_mode=RunningMode.IMAGE, call_back=None):
    model_path = os.path.join(recognizer_name, 'gesture_recognizer.task')
    options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
            result_callback=call_back)
    
    return GestureRecognizer.create_from_options(options)

def determine_prediction(result):
    try:
        prediction = result.gestures[0][0].category_name
    except IndexError:
        return 'Fail', 'No hand detected'
    
    if prediction == 'None':
        return 'Fail', 'Gesture not recognized'
    else:
        return 'Success', prediction
    
def resize_frame(frame, desired_width=480, desired_height=480):
    h, w = frame.shape[:2]
    if h < w:
        frame = cv2.resize(frame, (desired_width, math.floor(h/(w/desired_width))))
    else:
        frame = cv2.resize(frame, (math.floor(w/(h/desired_height)), desired_height))
    return frame

def draw_landmark_connections(frame, hand_landmarks):
    hand_landmarks_list = hand_landmarks
    annotated_frame = np.copy(frame)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
        solutions.drawing_utils.draw_landmarks(
            annotated_frame,
            hand_landmarks_proto,
            solutions.hands_connections.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_frame

def put_central_text(frame, text, height, color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX

    h, w = frame.shape[:2]
    text_width = cv2.getTextSize(text, font, 1, 2)[0][0]
    text_x = int((w - text_width) / 2)
    if height == 'bottom':
        text_y = h -30
    else:
        text_y = 30

    frame = cv2.putText(frame, text, 
                    org=(text_x, text_y), 
                    fontFace=font, 
                    fontScale=1,
                    color=color,
                    thickness=2)  
    return frame
    
def annotate_frame(frame, state, prediction, hand_landmarks, desired_width=480, desired_height=480, waving=False):
    frame = resize_frame(frame, desired_height, desired_width)

    if state == 'Success':
        frame = draw_landmark_connections(frame, hand_landmarks)

    frame = put_central_text(frame, prediction, 'bottom')

    if waving:
        text = 'Waving motion detected!'
        frame = put_central_text(frame, text, 'top', color=(0, 255, 0))

    return frame