# ---------- BEGIN: camera_processor.py ----------
"""
Direct camera/phone processor for billiard ball tracking.

Supports:
    - Phone as IP camera (IP Webcam, RTSP streams)
    - USB webcam / DroidCam
    - Local webcam (built-in)

Shows a live preview window with ball tracking overlay.
Optionally records the processed output to file.
"""

import cv2
import time
import numpy as np
from pathlib import Path
from billiard_tracker import BilliardTracker


class CameraProcessor:
    """
    Captures video from a camera source (IP cam, USB, webcam),
    runs billiard tracking, shows live preview, and optionally records.
    """

    def __init__(
        self,
        source,
        record_output=None,
        target_fps=30,
        tracker_settings=None,
        progress_callback=None,
        cancel_flag=None,
        preview_scale=1.0,
    ):
        """
        Args:
            source: Camera source. Can be:
                - int (0, 1, ...) for local/USB webcam
                - str URL for IP camera (e.g. "http://192.168.1.5:8080/video")
                - str RTSP URL (e.g. "rtsp://192.168.1.5:8554/stream")
            record_output: Path to save recorded video (None = no recording)
            target_fps: Target framerate
            tracker_settings: Dict with billiard tracker parameters
            progress_callback: function(message, percentage)
            cancel_flag: threading.Event to signal stop
            preview_scale: Scale factor for preview window (0.5 = half size)
        """
        self.source = source
        self.record_output = record_output
        self.target_fps = target_fps
        self.progress_callback = progress_callback or (lambda msg, pct: None)
        self.cancel_flag = cancel_flag
        self.preview_scale = preview_scale

        ts = tracker_settings or {}
        fps = target_fps
        self.tracker = BilliardTracker(
            min_ball_radius=ts.get("min_ball_radius", 8),
            max_ball_radius=ts.get("max_ball_radius", 25),
            shot_start_speed=ts.get("shot_start_speed", 8.0),
            shot_end_speed=ts.get("shot_end_speed", 2.0),
            path_fade_frames=int(ts.get("path_fade_seconds", 3.0) * fps),
            path_color=(0, 255, 255),
            path_thickness=2,
            label_balls=ts.get("label_balls", True),
        )

        self._running = False

    def _parse_source(self):
        """Convert source string to OpenCV-compatible input."""
        if isinstance(self.source, int):
            return self.source
        s = str(self.source).strip()
        # Numeric string -> device index
        if s.isdigit():
            return int(s)
        # URL-based sources
        return s

    def start(self):
        """Start capturing, processing, and displaying."""
        self._running = True
        source = self._parse_source()

        self.progress_callback(f"Opening camera: {source}", -1)
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            self.progress_callback(f"Cannot open camera source: {source}", 0)
            raise RuntimeError(
                f"Cannot open camera: {source}\n\n"
                "For IP Webcam (Android):\n"
                "  1. Install 'IP Webcam' from Play Store\n"
                "  2. Start server in the app\n"
                "  3. Use URL: http://<phone-ip>:8080/video\n\n"
                "For DroidCam:\n"
                "  1. Install DroidCam on phone + PC client\n"
                "  2. Connect via USB or WiFi\n"
                "  3. Use device index: 0 or 1\n\n"
                "For RTSP camera:\n"
                "  Use URL: rtsp://<ip>:<port>/stream"
            )

        # Get actual camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if actual_fps <= 0:
            actual_fps = self.target_fps

        self.progress_callback(
            f"Camera opened: {width}x{height} @ {actual_fps:.0f}fps", 0
        )

        # Setup recorder if requested
        writer = None
        if self.record_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(self.record_output), fourcc, actual_fps, (width, height)
            )
            self.progress_callback(f"Recording to: {self.record_output}", -1)

        frame_count = 0
        fps_timer = time.time()
        fps_frame_count = 0
        measured_fps = 0.0

        window_name = "Billiard Tracker - Live Preview (Q to quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        if self.preview_scale != 1.0:
            pw = int(width * self.preview_scale)
            ph = int(height * self.preview_scale)
            cv2.resizeWindow(window_name, pw, ph)

        try:
            while self._running:
                if self.cancel_flag and self.cancel_flag.is_set():
                    break

                ret, frame = cap.read()
                if not ret:
                    # For IP cameras, retry on frame drop
                    time.sleep(0.01)
                    continue

                # Process frame
                processed, balls = self.tracker.process_frame(frame)
                frame_count += 1
                fps_frame_count += 1

                # Draw HUD overlay
                self._draw_hud(processed, balls, measured_fps, frame_count, writer is not None)

                # Show preview
                cv2.imshow(window_name, processed)

                # Record if enabled
                if writer:
                    writer.write(processed)

                # FPS measurement
                elapsed = time.time() - fps_timer
                if elapsed >= 2.0:
                    measured_fps = fps_frame_count / elapsed
                    fps_frame_count = 0
                    fps_timer = time.time()
                    self.progress_callback(
                        f"Live: {measured_fps:.1f} fps | {len(balls)} balls | Frame #{frame_count}",
                        -1,
                    )

                # Check for quit key (Q or Esc)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break

        finally:
            self._running = False
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self.progress_callback(
                f"Camera processing stopped. {frame_count} frames processed.", 0
            )

    def _draw_hud(self, frame, balls, fps, frame_count, recording):
        """Draw a heads-up display with stats on the frame."""
        h, w = frame.shape[:2]

        # Background bar
        cv2.rectangle(frame, (0, 0), (w, 32), (0, 0, 0), -1)

        # Stats text
        ball_count = len(balls)
        cue_found = any(b.color_name == "white" for b in balls)
        cue_status = "CUE: TRACKING" if cue_found else "CUE: SEARCHING"

        info = f"FPS: {fps:.1f} | Balls: {ball_count} | {cue_status} | Frame: {frame_count}"
        cv2.putText(frame, info, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        # Recording indicator
        if recording:
            cv2.circle(frame, (w - 20, 16), 8, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 55, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Shot indicator
        if self.tracker.in_shot:
            cv2.putText(frame, "SHOT!", (w // 2 - 30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    def stop(self):
        """Signal the processor to stop."""
        self._running = False


def run_camera_processor(
    source,
    record_output=None,
    target_fps=30,
    tracker_settings=None,
    progress_callback=None,
    cancel_flag=None,
    preview_scale=1.0,
):
    """
    Convenience function. Blocks until cancelled, stream ends, or Q pressed.
    """
    processor = CameraProcessor(
        source=source,
        record_output=record_output,
        target_fps=target_fps,
        tracker_settings=tracker_settings,
        progress_callback=progress_callback,
        cancel_flag=cancel_flag,
        preview_scale=preview_scale,
    )
    processor.start()
    return processor


# ---------- END: camera_processor.py ----------
