# ---------- BEGIN: stream_processor.py ----------
"""
Real-time Twitch/stream processor for billiard ball tracking.

Captures a live stream via streamlink, processes frames through
BilliardTracker, and outputs to a virtual camera for OBS.

Requirements:
    pip install streamlink pyvirtualcam

OBS Setup:
    1. Install OBS-VirtualCam plugin (or OBS 28+ has it built-in)
    2. In OBS: Add Source -> Video Capture Device -> "AI Billiard Tracker"
"""

import subprocess
import threading
import time
import cv2
import numpy as np
from billiard_tracker import BilliardTracker

try:
    import pyvirtualcam
    HAS_VIRTUAL_CAM = True
except ImportError:
    HAS_VIRTUAL_CAM = False

try:
    import streamlink
    HAS_STREAMLINK = True
except ImportError:
    HAS_STREAMLINK = False


class StreamProcessor:
    """
    Captures a live video stream, runs billiard tracking,
    and outputs to a virtual camera visible in OBS.
    """

    def __init__(
        self,
        stream_url=None,
        quality="best",
        target_fps=30,
        output_width=1280,
        output_height=720,
        tracker_settings=None,
        progress_callback=None,
        cancel_flag=None,
    ):
        self.stream_url = stream_url
        self.quality = quality
        self.target_fps = target_fps
        self.output_width = output_width
        self.output_height = output_height
        self.progress_callback = progress_callback or (lambda msg, pct: None)
        self.cancel_flag = cancel_flag

        # Tracker setup
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
        self._frame_count = 0
        self._fps_actual = 0.0
        self._balls_detected = 0

    def _resolve_stream_url(self):
        """Use streamlink to get the direct stream URL from Twitch/YouTube."""
        if not HAS_STREAMLINK:
            raise RuntimeError(
                "streamlink is not installed. Run: pip install streamlink"
            )

        self.progress_callback("Resolving stream URL...", -1)
        try:
            streams = streamlink.streams(self.stream_url)
        except Exception as e:
            raise RuntimeError(f"Cannot access stream: {e}")

        if not streams:
            raise RuntimeError(
                f"No streams found at {self.stream_url}. "
                "Check the URL and that the stream is live."
            )

        available = list(streams.keys())
        self.progress_callback(f"Available qualities: {', '.join(available)}", -1)

        quality = self.quality
        if quality not in streams:
            # Fallback: try best, then first available
            quality = "best" if "best" in streams else available[0]
            self.progress_callback(f"Quality '{self.quality}' not found, using '{quality}'", -1)

        stream = streams[quality]
        return stream.url

    def _open_ffmpeg_reader(self, input_url):
        """Open an FFmpeg subprocess that reads from a stream URL and outputs raw frames."""
        cmd = [
            "ffmpeg",
            "-i", input_url,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-vf", f"scale={self.output_width}:{self.output_height}",
            "-an",  # No audio
            "-sn",  # No subtitles
            "-loglevel", "warning",
            "-",
        ]
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self.output_width * self.output_height * 3 * 2,
        )

    def start(self):
        """Start the stream processing pipeline."""
        if not HAS_VIRTUAL_CAM:
            raise RuntimeError(
                "pyvirtualcam is not installed. Run: pip install pyvirtualcam\n"
                "On Windows, also install OBS Virtual Camera."
            )

        self._running = True

        # Resolve the actual stream URL
        direct_url = self._resolve_stream_url()
        self.progress_callback("Opening stream with FFmpeg...", -1)

        # Start FFmpeg reader
        ffmpeg_proc = self._open_ffmpeg_reader(direct_url)
        frame_size = self.output_width * self.output_height * 3
        frame_interval = 1.0 / self.target_fps

        try:
            with pyvirtualcam.Camera(
                width=self.output_width,
                height=self.output_height,
                fps=self.target_fps,
                fmt=pyvirtualcam.PixelFormat.BGR,
                device="AI Billiard Tracker",
            ) as cam:
                self.progress_callback(
                    f"Virtual camera active: {cam.device} ({self.output_width}x{self.output_height}@{self.target_fps}fps)",
                    0,
                )

                fps_timer = time.time()
                fps_frame_count = 0

                while self._running:
                    if self.cancel_flag and self.cancel_flag.is_set():
                        break

                    raw = ffmpeg_proc.stdout.read(frame_size)
                    if len(raw) < frame_size:
                        # Stream ended or interrupted
                        self.progress_callback("Stream ended or connection lost.", 0)
                        break

                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                        (self.output_height, self.output_width, 3)
                    )

                    # Process through billiard tracker
                    processed, balls = self.tracker.process_frame(frame)
                    self._frame_count += 1
                    self._balls_detected = len(balls)

                    # Send to virtual camera
                    cam.send(processed)

                    # FPS calculation
                    fps_frame_count += 1
                    elapsed = time.time() - fps_timer
                    if elapsed >= 2.0:
                        self._fps_actual = fps_frame_count / elapsed
                        fps_frame_count = 0
                        fps_timer = time.time()
                        self.progress_callback(
                            f"Live: {self._fps_actual:.1f} fps | "
                            f"{self._balls_detected} balls | "
                            f"Frame #{self._frame_count}",
                            -1,
                        )

                    cam.sleep_until_next_frame()

        finally:
            self._running = False
            ffmpeg_proc.terminate()
            ffmpeg_proc.wait()
            self.progress_callback("Stream processing stopped.", 0)

    def stop(self):
        """Signal the processor to stop."""
        self._running = False


def run_stream_processor(
    stream_url,
    quality="best",
    target_fps=30,
    output_width=1280,
    output_height=720,
    tracker_settings=None,
    progress_callback=None,
    cancel_flag=None,
):
    """
    Convenience function to run the stream processor.
    Blocks until cancelled or stream ends.
    """
    processor = StreamProcessor(
        stream_url=stream_url,
        quality=quality,
        target_fps=target_fps,
        output_width=output_width,
        output_height=output_height,
        tracker_settings=tracker_settings,
        progress_callback=progress_callback,
        cancel_flag=cancel_flag,
    )
    processor.start()
    return processor


# ---------- END: stream_processor.py ----------
