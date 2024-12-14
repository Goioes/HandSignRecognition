# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:43:03 2024

@author: gielo
"""

from flask import Flask, request, jsonify
import numpy as np
import os, base64, cv2
import subprocess
from matplotlib import pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def load_model():
    model_path = os.path.join('custom_rps_gesture_recognizer', 'gesture_recognizer.task')
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(base_options=base_options)
    return vision.GestureRecognizer.create_from_options(options)

app = Flask(__name__)
model = load_model()


@app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = data.get('image_data')
        
        decoded_image = base64.b64decode(image_data)
        np_image = np.frombuffer(decoded_image, dtype=np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image, interpolation='nearest')
        plt.show()
        
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(image))
        result = model.recognize(image)
        
        try:
            prediction = result.gestures[0][0].category_name
            if prediction == 'None':
                prediction = 'Gesture not recognized'

        except IndexError:
            prediction = 'No hand detected'
        
        return jsonify({"predicted_sign": prediction})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run()