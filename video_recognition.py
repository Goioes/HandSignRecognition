import cv2
import mediapipe as mp
from utils import annotate_frame, load_gesture_recognizer

GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

class LiveGestureRecognition():
    def __init__(self):
        self.results = None
    
    def store_result(self, result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        self.results = result

    def main(self):
        video = cv2.VideoCapture(0)

        timestamp = 0
        with load_gesture_recognizer('custom_gesture_recognizer',
                                     running_mode=VisionRunningMode.LIVE_STREAM,
                                     call_back=self.store_result) as recognizer:
            while video.isOpened(): 
                # Capture frame-by-frame
                ret, frame = video.read()

                if not ret:
                    print("Ignoring empty frame")
                    break

                timestamp += 1
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                recognizer.recognize_async(mp_image, timestamp)

                try:
                    prediction = self.results.gestures[0][0].category_name
                    if prediction == 'None':
                        prediction = 'Gesture not recognized'
                        cv2.imshow('Show', frame)
                    else:
                        annotated_frame = annotate_frame(mp_image.numpy_view(), self.results)
                        cv2.imshow('Show', annotated_frame)
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
