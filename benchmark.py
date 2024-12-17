import matplotlib.pyplot as plt
import mediapipe as mp
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from utils import load_gesture_recognizer, determine_prediction, annotate_frame

dataset_path = "benchmark_data"
data_set = tf.keras.preprocessing.image_dataset_from_directory(dataset_path,
    labels='inferred',
    shuffle=True)

recognizer = load_gesture_recognizer('custom_gesture_recognizer')

ground_truths = []
predictions = []

plt.figure(figsize=(10, 10))
for frames, labels in data_set.take(1):
    for i in range(len(frames)):       
        frame_np = frames[i].numpy().astype("uint8")
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_np)
        
        ground_truth = data_set.class_names[labels[i]]
        ground_truths.append(ground_truth)
        
        results = recognizer.recognize(frame)
        state, prediction = determine_prediction(results)
        predictions.append(prediction)

        if i < 10:
            ax = plt.subplot(2, 5, i + 1)
            frame_np = annotate_frame(frame_np, state, prediction, results.hand_landmarks)
            plt.imshow(frame_np)
            plt.title(f'GT: {ground_truth}, PR: {prediction}')
            plt.axis("off")
plt.show()

accuracy = accuracy_score(ground_truths, predictions)
print("Accuracy:", accuracy)
print("Detailed Report:")
print(classification_report(ground_truths, predictions))