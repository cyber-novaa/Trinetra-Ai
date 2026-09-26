# Trinetra person detection feature

This feature detects people (COCO class `person`) in still images, uploaded videos, saved video files, or a live webcam feed. It uses the included `yolov8n.pt` YOLOv8 nano weights. The Streamlit app can display and download annotated image/video results; the two command-line scripts cover local video and webcam use.

## Setup (Windows PowerShell)

Open PowerShell in this folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Keep `yolov8n.pt`, `detector.py`, and the Python scripts in this same folder. If PowerShell blocks activation, run the commands below instead of activating the environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the standalone tools

Process a video, save an annotated MP4 beside the input, and show a preview window:

```powershell
python person_video.py .\input_video.mp4
```

Choose a different output or disable the preview window:

```powershell
python person_video.py .\input_video.mp4 --output .\result.mp4 --no-preview
```

Run live webcam detection (press `q` in the preview window to quit):

```powershell
python person_webcam.py
```

For a different camera index or detection threshold, use `--camera 1` or `--confidence 0.6`. If the model is stored elsewhere, pass `--model C:\path\to\yolov8n.pt`.

## Notes

- On first run, PyTorch/Ultralytics may take time to initialize. The app loads the local model once and reuses it while the Streamlit process is running.
- Detection runs on CPU by default when no supported CUDA GPU is available.
- OpenCV video codecs vary by computer. If a video cannot open or an MP4 cannot be written, try an MP4 input and check that OpenCV is installed in the active Python environment.
- `person_video.py` and `person_webcam.py` open desktop preview windows, so run them in a normal local desktop session. The Streamlit app works through a browser.
