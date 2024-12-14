import os
import random
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from handmark_visualization import annotate_frame


dataset_path = "test_data"
model_path = os.path.join('custom_rps_gesture_recognizer', 'gesture_recognizer.task')

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

#image_dir = os.path.join(dataset_path, 'surf.jpg')
#image_file_name = os.path.join(image_dir, random.choice(os.listdir(image_dir)))
image_file_name = os.path.join(dataset_path, 'rock.jpg')
image = mp.Image.create_from_file(image_file_name)

recognition_result = recognizer.recognize(image)
print(recognition_result.gestures, recognition_result.handedness, recognition_result.hand_landmarks)

annotated_image = annotate_frame(image.numpy_view(), recognition_result)
cv2.imshow('Show', annotated_image)
cv2.waitKey()
