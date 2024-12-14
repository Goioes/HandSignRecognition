import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

def annotate_frame(frame, result):
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
            solutions.hands_connections.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_frame