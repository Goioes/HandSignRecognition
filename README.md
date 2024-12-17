# HandSignRecognition
Recognition of hand gestures using the MediaPipe framework.

This project implements real-time gesture recognition using the MediaPipe framework and OpenCV. It tracks hand landmarks, classifies basic gestures like "surf hand," "open palm,", "peace", and "three", and detects a waving hand using simple logic.

![alt text](https://github.com/Goioes/HandSignRecognition/blob/main/results/Benchmark_results.png?raw=true)

## Features
- Real-time hand tracking and gesture recognition using MediaPipe.
- Gesture recognition for:
  - Surf hand 
  - Open Palm 
  - Peace sign
  - Three sign
- Detection of a waving hand.
- API endpoint for hand gesture recognition in images (JSON format).

## Code
- Gesture_Recognition_Maker.ipynb: trains a custom gesture recognizer on a specified dataset.
- benchmark.py: benchmarks the gesture recognizer on a dataset of 10 images.
- endpoint_recognition.py: creates API endpoint to access the gesture recognizer for images.
- request_recognition.py: sends recognition request to API endpoint.
- live_recognition.py: performs gesture recognition on webcam livestream and detects waving motion.
- utils.py: contains specific helper functions such as image annotation.




