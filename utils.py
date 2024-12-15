import os
import math
import numpy as np
import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

VisionRunningMode = mp.tasks.vision.RunningMode

def load_gesture_recognizer(recognizer_name, running_mode=VisionRunningMode.IMAGE, call_back=None):
    model_path = os.path.join(recognizer_name, 'gesture_recognizer.task')
    options = vision.GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
            result_callback=call_back)
    
    return vision.GestureRecognizer.create_from_options(options)

def annotate_frame(frame, result):
    hand_landmarks_list = result.hand_landmarks
    annotated_frame = np.copy(frame)

    # Loop through the detected poses to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        # Draw the pose landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
        solutions.drawing_utils.draw_landmarks(
            annotated_frame,
            hand_landmarks_proto,
            solutions.hands_connections.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_frame

def resize_show_image(image, desired_width=480, desired_height=480):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (desired_width, math.floor(h/(w/desired_width))))
    else:
        img = cv2.resize(image, (math.floor(w/(h/desired_height)), desired_height))
    cv2.imshow('Show', img)
    cv2.waitKey()