import numpy as np
import cv2
import time
import mediapipe as mp
from mediapipe.tasks.python.vision import GestureRecognizerResult, RunningMode
from mediapipe.python.solutions.hands import HandLandmark
from utils import load_gesture_recognizer, determine_prediction, annotate_frame

class LiveGestureRecognition():
    def __init__(self):
        self.results = None
        self.waving = False
        self.frame_number = 0
        self.waving_timestamp_ms = 0
        self.hand_positions = []
    
    def process_result(self, result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        self.results = result   # Store result gesture recognizer
        if self.results != None:
            self.state, self.prediction = determine_prediction(self.results)
            print('gesture recognition result: {}'.format(self.prediction))
            if (timestamp_ms - self.waving_timestamp_ms) > 250:
                self.waving = self.check_waving()
                if self.waving:
                    self.waving_timestamp_ms = timestamp_ms
                    print(f'Waving!!!')

    def check_waving(self):
        if self.prediction == 'open_palm':
            try:
                hand_landmark = self.results.hand_landmarks[0][HandLandmark.MIDDLE_FINGER_TIP]
                return self.check_waving_motion(hand_landmark)
            except IndexError:
                print(f'No hand detected!')
        return False

    def check_waving_motion(self, hand_landmark):
        self.hand_positions.append(hand_landmark.x)
        if len(self.hand_positions) > 10:   # Keep track of last 10 positions
            self.hand_positions.pop(0)

        if len(self.hand_positions) > 5:    # Ensure enough points to calculate
            hand_xs = np.asarray(self.hand_positions)
            hand_delta_xs = np.diff(hand_xs)
            direction_change = np.diff(np.sign(hand_delta_xs))
            direction_change_index = np.where(direction_change != 0)[0]
            extreme_xs = hand_xs[direction_change_index + 1]
            extreme_delta_xs = np.absolute(np.diff(extreme_xs))
            print(f'Info: {hand_xs}, {hand_landmark.x}, {np.mean(extreme_delta_xs)}, {extreme_delta_xs}')
            # Define as wave
            return (len(extreme_delta_xs) > 0 and np.mean(extreme_delta_xs) > 0.1)
        
    def main(self):
        video = cv2.VideoCapture(0)
        
        with load_gesture_recognizer('custom_gesture_recognizer',
                                     running_mode=RunningMode.LIVE_STREAM,
                                     call_back=self.process_result) as recognizer:
            while video.isOpened(): 
                # Capture frame-by-frame
                ret, frame = video.read()
                if not ret:
                    print("Ignoring empty frame")
                    break

                #self.frame_number += 1
                timestamp_ms = int(time.time() * 1000)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                recognizer.recognize_async(mp_image, timestamp_ms)

                if self.results != None:
                    frame = annotate_frame(frame, self.state, self.prediction, self.results.hand_landmarks, waving=self.waving)
                cv2.imshow('Show', frame)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

            video.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    live_gesture_recognition = LiveGestureRecognition()
    live_gesture_recognition.main()
