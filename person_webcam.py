"""Run live YOLO person detection from a webcam. Press q to quit."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from detector import annotate_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parent / "yolov8n.pt", help="YOLO weights file")
    parser.add_argument("--confidence", type=float, default=0.50, help="Detection threshold from 0 to 1 (default: 0.50)")
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence must be between 0 and 1")
    if not args.model.is_file():
        raise SystemExit(f"Model weights do not exist: {args.model}")

    model = YOLO(str(args.model))
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Check the camera connection and permissions.")
    print("Live detection started. Press q in the video window to quit.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera stopped returning frames.")
                break
            annotated, _ = annotate_frame(frame, model, args.confidence)
            cv2.imshow("Trinetra Person Detection · press q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
