"""HandVision Live — an enhanced, real-time MediaPipe hand tracker.

Run: python app.py
Keys: Q/Esc quit | M mirror | L landmarks | R record | S screenshot | H help
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp

APP_NAME = "HandVision Live | Sudhir Yadav"
TIP_IDS = (4, 8, 12, 16, 20)


class HandVision:
    """Draw and classify hands detected by MediaPipe."""

    def __init__(self, max_hands: int, detection: float, tracking: float) -> None:
        self.hands_api = mp.solutions.hands
        self.drawer = mp.solutions.drawing_utils
        self.styles = mp.solutions.drawing_styles
        self.detector = self.hands_api.Hands(
            static_image_mode=False, max_num_hands=max_hands, model_complexity=1,
            min_detection_confidence=detection, min_tracking_confidence=tracking,
        )

    @staticmethod
    def finger_count(points: list[tuple[int, int]], label: str) -> int:
        """Return raised fingers. Thumb direction is adjusted for handedness."""
        raised = 0
        if label == "Right":
            raised += points[4][0] < points[3][0]
        else:
            raised += points[4][0] > points[3][0]
        for tip in TIP_IDS[1:]:
            raised += points[tip][1] < points[tip - 2][1]
        return int(raised)

    @staticmethod
    def gesture(count: int) -> str:
        return {0: "FIST", 1: "ONE", 2: "PEACE", 3: "THREE", 4: "FOUR", 5: "OPEN PALM"}[count]

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.process(rgb)
        detected = []
        if not result.multi_hand_landmarks:
            return detected
        height, width = frame.shape[:2]
        for landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = handedness.classification[0].label
            points = [(int(p.x * width), int(p.y * height)) for p in landmarks.landmark]
            count = self.finger_count(points, label)
            self.drawer.draw_landmarks(
                frame, landmarks, self.hands_api.HAND_CONNECTIONS,
                self.styles.get_default_hand_landmarks_style(),
                self.styles.get_default_hand_connections_style(),
            )
            for tip in TIP_IDS:
                cv2.circle(frame, points[tip], 6, (0, 238, 255), cv2.FILLED)
            detected.append((label, count, points[0]))
        return detected

    def close(self) -> None:
        self.detector.close()


def panel(frame, fps: float, hands, recording: bool, show_help: bool) -> None:
    """Render an unobtrusive status panel and per-hand labels."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (315, 112), (18, 24, 38), cv2.FILLED)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.putText(frame, "HANDVISION LIVE", (26, 42), cv2.FONT_HERSHEY_DUPLEX, 0.63, (0, 238, 255), 1)
    cv2.putText(frame, f"FPS  {fps:4.1f}     HANDS  {len(hands)}", (26, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (242, 242, 242), 1)
    cv2.putText(frame, "REC" if recording else "READY", (26, 97), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 55, 255) if recording else (120, 235, 130), 2)
    for label, count, (x, y) in hands:
        text = f"{label}: {HandVision.gesture(count)} ({count})"
        cv2.putText(frame, text, (max(8, x - 45), max(30, y - 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 238, 255), 2)
    if show_help:
        cv2.rectangle(frame, (12, frame.shape[0] - 52), (frame.shape[1] - 12, frame.shape[0] - 12), (18, 24, 38), cv2.FILLED)
        cv2.putText(frame, "Q quit   M mirror   L landmarks   R recording   S screenshot   H help", (25, frame.shape[0] - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)


def open_camera(index: int):
    """Open the selected webcam, preferring the low-latency Windows backend."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}. Check its connection or try --camera 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced real-time hand tracking for Sudhir Yadav")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--max-hands", type=int, default=2, choices=range(1, 5), help="Maximum hands to track")
    parser.add_argument("--detection", type=float, default=0.65, help="Detection confidence, 0–1")
    parser.add_argument("--tracking", type=float, default=0.60, help="Tracking confidence, 0–1")
    args = parser.parse_args()
    if not 0 <= args.detection <= 1 or not 0 <= args.tracking <= 1:
        parser.error("--detection and --tracking must be between 0 and 1")

    output_dir = Path("captures")
    output_dir.mkdir(exist_ok=True)
    cap = open_camera(args.camera)
    tracker = HandVision(args.max_hands, args.detection, args.tracking)
    cv2.namedWindow(APP_NAME, cv2.WINDOW_NORMAL)
    mirrored, draw_landmarks, show_help, writer = True, True, True, None
    last_time, fps = time.perf_counter(), 0.0
    print("HandVision Live started. Press H to toggle controls; Q or Esc to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame was unavailable; stopping.")
                break
            if mirrored:
                frame = cv2.flip(frame, 1)
            now = time.perf_counter()
            instant = 1 / max(now - last_time, 0.0001)
            fps = instant if fps == 0 else fps * 0.9 + instant * 0.1
            last_time = now
            hands = tracker.process(frame) if draw_landmarks else []
            panel(frame, fps, hands, writer is not None, show_help)
            if writer is not None:
                writer.write(frame)
            cv2.imshow(APP_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                mirrored = not mirrored
            elif key == ord("l"):
                draw_landmarks = not draw_landmarks
            elif key == ord("h"):
                show_help = not show_help
            elif key == ord("s"):
                path = output_dir / f"handvision_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(path), frame)
                print(f"Screenshot saved: {path}")
            elif key == ord("r"):
                if writer is None:
                    path = output_dir / f"handvision_{datetime.now():%Y%m%d_%H%M%S}.mp4"
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (frame.shape[1], frame.shape[0]))
                    print(f"Recording started: {path}")
                else:
                    writer.release()
                    writer = None
                    print("Recording stopped.")
    finally:
        if writer is not None:
            writer.release()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
