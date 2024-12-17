import os
import requests
import base64
import random

url = 'http://127.0.0.1:5000/predict'

data_path = 'benchmark_data'
ground_truth = random.choice(os.listdir(data_path))

image_file_name = random.choice(os.listdir(os.path.join(data_path, ground_truth)))
image_path = os.path.join(data_path, ground_truth, image_file_name)
with open(image_path, "rb") as image_file:
    image = image_file.read()

encoded_image = base64.encodebytes(image).decode('utf-8')

    
myobj = {'image_data': str(encoded_image)}

result = requests.post(url, json = myobj).json()

print(f'Ground truth: {ground_truth}')
print(f'Prediction: {result["predicted_sign"]}')