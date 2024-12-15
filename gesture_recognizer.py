import os
import random
import mediapipe as mp
from utils import annotate_frame, load_gesture_recognizer, resize_show_image

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

annotated_image = annotate_frame(image.numpy_view(), recognition_result)
resize_show_image(annotated_image)


