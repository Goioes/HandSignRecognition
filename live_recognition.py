import numpy as np
import cv2
import time
import mediapipe as mp
from mediapipe.tasks.python.vision import GestureRecognizerResult, RunningMode
from mediapipe.python.solutions.hands import HandLandmark
from utils import load_gesture_recognizer, determine_prediction, annotate_frame

MINIMUM_WAVING_TIME_MS = 200
MINIMUM_WAVE_DISPLACEMENT = 0.1

class LiveGestureRecognition():
    def __init__(self):
        self.results = None
        self.waving = False
        self.waving_timestamp_ms = 0
        self.hand_positions = []
    
    def process_result(self, result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        self.results = result   # Store result gesture recognizer
        if self.results != None:
            self.state, self.prediction = determine_prediction(self.results)
            print('gesture recognition result: {}'.format(self.prediction))

            # Check if minimum waving time passed
            if (timestamp_ms - self.waving_timestamp_ms) > MINIMUM_WAVING_TIME_MS:
                self.waving = self.check_waving()
                if self.waving:
                    self.waving_timestamp_ms = timestamp_ms # Update waving time
                    print(f'Waving!!!')

    def check_waving(self):
        """
        Waving if open palm and significant periodic displacement of hand position, i.e. waving motion.
        """
        
        if self.prediction == 'open_palm':
            try:
                hand_landmark = self.results.hand_landmarks[0][HandLandmark.MIDDLE_FINGER_TIP]
                return self.check_waving_motion(hand_landmark)
            except IndexError:
                print(f'No hand detected!')
        return False

    def check_waving_motion(self, hand_landmark):
        """
        Waving motion if at least 2 direction changes and significant displacement between extremes.
        """

        self.hand_positions.append(hand_landmark.x)
        if len(self.hand_positions) > 10:   # Keep track of last 10 positions along x direction
            self.hand_positions.pop(0)

        if len(self.hand_positions) > 5:    # Ensure enough points to calculate
            hand_xs = np.asarray(self.hand_positions)
            hand_delta_xs = np.diff(hand_xs)
            direction_change = np.diff(np.sign(hand_delta_xs))
            direction_change_index = np.where(direction_change != 0)[0]
            extreme_xs = hand_xs[direction_change_index + 1]    # Find local extreme values
            extreme_delta_xs = np.absolute(np.diff(extreme_xs))
            
            return (len(direction_change_index) > 2 and np.mean(extreme_delta_xs) > MINIMUM_WAVE_DISPLACEMENT)
        
    def main(self):
        video = cv2.VideoCapture(0)
        
        with load_gesture_recognizer('custom_gesture_recognizer',
                                     running_mode=RunningMode.LIVE_STREAM,
                                     call_back=self.process_result) as recognizer:
            while video.isOpened(): 
                ret, frame = video.read()   # Capture frame-by-frame
                if not ret:
                    print("Ignoring empty frame")
                    break

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
