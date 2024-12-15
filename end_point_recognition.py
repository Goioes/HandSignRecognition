# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:43:03 2024

@author: gielo
"""

from flask import Flask, request, jsonify
import numpy as np
import base64, cv2
import mediapipe as mp
from utils import annotate_frame, load_gesture_recognizer, resize_show_image

app = Flask(__name__)
model = load_gesture_recognizer('custom_rps_gesture_recognizer')


@app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = data.get('image_data')
        
        decoded_image = base64.b64decode(image_data)
        np_image = np.frombuffer(decoded_image, dtype=np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(image))
        recognition_result = model.recognize(image)

        annotated_image = annotate_frame(image.numpy_view(), recognition_result)
        resize_show_image(annotated_image)
        
        try:
            prediction = recognition_result.gestures[0][0].category_name
            if prediction == 'None':
                prediction = 'Gesture not recognized'

        except IndexError:
            prediction = 'No hand detected'
        
        return jsonify({"predicted_sign": prediction})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run()