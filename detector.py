"""Shared YOLO person detection helpers used by the app and CLI tools."""

from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO


PERSON_CLASS_ID = 0  # COCO class 0 is person.


def annotate_frame(frame: np.ndarray, model: YOLO, confidence: float) -> tuple[np.ndarray, int]:
    """Return a BGR frame annotated with person boxes and its person count."""
    result = model.predict(frame, classes=[PERSON_CLASS_ID], conf=confidence, verbose=False)[0]
    person_count = 0
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            person_count += 1
            color = (235, 99, 37)  # OpenCV uses BGR.
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"Person {person_count}"
            label_y = max(y1, 24)
            cv2.rectangle(frame, (x1, label_y - 23), (x1 + 112, label_y), color, -1)
            cv2.putText(frame, label, (x1 + 5, label_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.rectangle(frame, (0, 0), (300, 42), (235, 99, 37), -1)
    cv2.putText(frame, f"People detected: {person_count}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame, person_count
