import os
import random
import numpy as np
import cv2
from matplotlib import pyplot as plt
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

class LiveGestureRecognition():
    def __init__(self):
        self.results = None
    
    def store_result(self, result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        self.results = result

    def annotate_frame(self, frame, result):
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
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style())
        return annotated_frame

    def main(self):
        video = cv2.VideoCapture(0)

        model_path = os.path.join('canned_gesture_recognizer', 'gesture_recognizer.task')
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.store_result)

        timestamp = 0
        with GestureRecognizer.create_from_options(options) as recognizer:
            while video.isOpened(): 
                # Capture frame-by-frame
                ret, frame = video.read()

                if not ret:
                    print("Ignoring empty frame")
                    break

                timestamp += 1
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                # Send live image data to perform gesture recognition
                # The results are accessible via the `result_callback` provided in
                # the `GestureRecognizerOptions` object.
                # The gesture recognizer must be created with the live stream mode.
                recognizer.recognize_async(mp_image, timestamp)

                try:
                    prediction = self.results.gestures[0][0].category_name
                    if prediction == 'None':
                        prediction = 'Gesture not recognized'
                        cv2.imshow('Show', frame)
                    else:
                        #annotated_frame = self.annotate_frame(mp_image.numpy_view(), self.results)
                        #cv2.imshow('Show',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                        cv2.imshow('Show', frame)
                except AttributeError:
                    prediction = 'Initializaton'
                except IndexError:
                    prediction = 'No hand detected'
                    cv2.imshow('Show', frame)
                print('gesture recognition result: {}'.format(prediction))

                if cv2.waitKey(5) & 0xFF == 27:
                    break

            video.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    live_gesture_recognition = LiveGestureRecognition()
    live_gesture_recognition.main()
