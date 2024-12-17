from flask import Flask, request, jsonify
import numpy as np
import base64, cv2
import mediapipe as mp
from utils import load_gesture_recognizer, determine_prediction

app = Flask(__name__)
model = load_gesture_recognizer('custom_gesture_recognizer')


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

        _, prediction = determine_prediction(recognition_result)
        
        return jsonify({"predicted_sign": prediction})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run()