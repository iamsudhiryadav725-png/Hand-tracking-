# HandVision Live

An enhanced real-time hand-tracking project built for **Sudhir Yadav**. It uses OpenCV and MediaPipe to track up to two hands, label left/right hands, recognize simple finger gestures, and render a smooth live dashboard.

## Features

- Real-time 21-point hand landmark detection
- Left/right labels and finger-count gestures: fist, one, peace, three, four, open palm
- Smoothed FPS display and live status panel
- Mirror, landmark, and help overlays that can be toggled while running
- Screenshot capture and MP4 recording
- Camera selection and confidence options through command-line arguments

## Setup

Use Python 3.10 or 3.11 for the smoothest MediaPipe installation.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

If your webcam is not camera 0, run `python app.py --camera 1`.

## Controls

| Key | Action |
| --- | --- |
| `Q` / `Esc` | Quit safely |
| `M` | Toggle mirror mode |
| `L` | Toggle hand tracking overlay |
| `R` | Start/stop MP4 recording |
| `S` | Save a screenshot |
| `H` | Show/hide controls |

Screenshots and recordings are saved to `captures/`. Good, even lighting and keeping the full hand within the camera frame gives the most stable results.
