import os
import random
import cv2
from matplotlib import pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
#print(recognition_result.gestures[0][0].category_name)

image = cv2.imread(image_file_name)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image, interpolation='nearest')
plt.show()