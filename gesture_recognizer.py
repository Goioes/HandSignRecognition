import os
import random
import cv2
import mediapipe as mp
from utils import load_gesture_recognizer, annotate_frame

dataset_path = "benchmark_data"

recognizer = load_gesture_recognizer('custom_gesture_recognizer')

random_folder = random.choice(os.listdir(dataset_path))
print(random_folder)
image_dir = os.path.join(dataset_path, random_folder)
image_file_name = os.path.join(image_dir, random.choice(os.listdir(image_dir)))
#image_file_name = os.path.join(dataset_path, 'folder', 'IMG_6681.jpg')
image = mp.Image.create_from_file(image_file_name)

    
recognition_result = recognizer.recognize(image)
print(recognition_result.gestures)

image, _ = annotate_frame(image.numpy_view(), recognition_result)
cv2.imshow('Show', image)
cv2.waitKey()

