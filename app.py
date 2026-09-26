"""Streamlit interface for Trinetra's person detection feature."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from detector import annotate_frame


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.45

st.set_page_config(page_title="Trinetra | Person Detection", page_icon="👤", layout="centered")

st.markdown(
    """
    <style>
    .stApp { background: #fff; color: #172033; }
    #MainMenu, footer { visibility: hidden; }
    .hero { font-size: 2.6rem; font-weight: 700; letter-spacing: -.04em; }
    .accent { color: #2563eb; }
    .muted { color: #64748b; margin-bottom: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="hero">Trinetra <span class="accent">Person Detection</span></div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Detect and count people in images and video using YOLOv8.</div>', unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def get_model() -> YOLO:
    if not MODEL_PATH.is_file():
        st.error(f"Model weights not found: {MODEL_PATH}. Keep yolov8n.pt beside app.py.")
        st.stop()
    try:
        return load_model(str(MODEL_PATH))
    except Exception as exc:
        st.error(f"Could not load the YOLO model: {exc}")
        st.stop()


model = get_model()
mode = st.radio("Detection mode", ("Image", "Video"), horizontal=True)
confidence = st.slider("Confidence threshold", min_value=0.10, max_value=0.90, value=DEFAULT_CONFIDENCE, step=0.05)

if mode == "Image":
    uploaded_image = st.file_uploader("Choose an image", type=("jpg", "jpeg", "png", "bmp", "webp"), key="image")
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Uploaded image", use_container_width=True)
            if st.button("Run detection", type="primary", use_container_width=True):
                bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                annotated, count = annotate_frame(bgr, model, confidence)
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(rgb, caption=f"Detection result · {count} {'person' if count == 1 else 'people'}", use_container_width=True)
                output = io.BytesIO()
                Image.fromarray(rgb).save(output, format="PNG")
                st.download_button("Download annotated image", output.getvalue(), "trinetra_detection.png", "image/png", use_container_width=True)
        except (OSError, ValueError) as exc:
            st.error(f"Could not read this image: {exc}")
else:
    uploaded_video = st.file_uploader("Choose a video", type=("mp4", "avi", "mov", "mkv"), key="video")
    if uploaded_video is not None:
        st.video(uploaded_video)
        if st.button("Run detection", type="primary", use_container_width=True):
            suffix = Path(uploaded_video.name).suffix or ".mp4"
            input_path = None
            output_path = None
            capture = None
            writer = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as source:
                    source.write(uploaded_video.getvalue())
                    input_path = source.name
                capture = cv2.VideoCapture(input_path)
                if not capture.isOpened():
                    raise RuntimeError("OpenCV could not open this video. Try an MP4 encoded with H.264 or MPEG-4.")
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = capture.get(cv2.CAP_PROP_FPS)
                total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if width <= 0 or height <= 0:
                    raise RuntimeError("The video has invalid frame dimensions.")
                if not np.isfinite(fps) or fps <= 0:
                    fps = 25.0
                out_file = tempfile.NamedTemporaryFile(delete=False, suffix="_detected.mp4")
                output_path = out_file.name
                out_file.close()
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError("OpenCV could not create the output video on this system.")

                bar = st.progress(0, text="Processing video…")
                preview = st.empty()
                frame_count = 0
                sum_people = 0
                peak_people = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    annotated, count = annotate_frame(frame, model, confidence)
                    writer.write(annotated)
                    frame_count += 1
                    sum_people += count
                    peak_people = max(peak_people, count)
                    if frame_count % 8 == 0:
                        preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                    if total_frames > 0 and frame_count % 4 == 0:
                        bar.progress(min(frame_count / total_frames, 1.0), text=f"Processed {frame_count} of {total_frames} frames")
                if frame_count == 0:
                    raise RuntimeError("No readable frames were found in this video.")
                bar.progress(1.0, text="Detection complete")
                preview.empty()
                c1, c2, c3 = st.columns(3)
                c1.metric("Frames processed", frame_count)
                c2.metric("Peak people", peak_people)
                c3.metric("Average per frame", f"{sum_people / frame_count:.1f}")
                capture.release()
                capture = None
                writer.release()
                writer = None
                video_bytes = Path(output_path).read_bytes()
                st.video(video_bytes)
                st.download_button("Download detected video", video_bytes, "trinetra_detected.mp4", "video/mp4", use_container_width=True)
            except (OSError, RuntimeError, cv2.error) as exc:
                st.error(str(exc))
            finally:
                if capture is not None:
                    capture.release()
                if writer is not None:
                    writer.release()
                for temporary_path in (input_path, output_path):
                    if temporary_path:
                        Path(temporary_path).unlink(missing_ok=True)

st.caption("Trinetra · YOLOv8 · Ultralytics")
