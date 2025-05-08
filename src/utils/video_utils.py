import cv2
import numpy as np
import mediapipe as mp
mp_pose = mp.solutions.pose
_pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

def extract_pose(frame: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = _pose.process(rgb)
    if not res.pose_landmarks:
        return np.zeros((17, 2), dtype=np.float32)

    lm = res.pose_landmarks.landmark
    coords = np.array([[l.x, l.y] for l in lm[:17]], dtype=np.float32)
    return coords
