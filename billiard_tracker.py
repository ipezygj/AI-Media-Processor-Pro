# ---------- BEGIN: billiard_tracker.py ----------
"""
Billiard Ball Tracker - Detects pool balls, tracks the cue ball,
and draws its path after each shot.

Uses OpenCV with HSV color filtering and Hough circle detection.
"""

import cv2
import numpy as np
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field


# --- Ball color definitions in HSV ---
# Each entry: (name, display_color_bgr, hsv_lower, hsv_upper)
# Multiple ranges for colors that wrap around HSV hue
BALL_COLORS = {
    "white": {
        "display": (255, 255, 255),
        "ranges": [((0, 0, 180), (180, 50, 255))]
    },
    "yellow": {
        "display": (0, 255, 255),
        "ranges": [((20, 100, 100), (35, 255, 255))]
    },
    "blue": {
        "display": (255, 100, 0),
        "ranges": [((100, 100, 80), (130, 255, 255))]
    },
    "red": {
        "display": (0, 0, 255),
        "ranges": [((0, 120, 80), (8, 255, 255)), ((170, 120, 80), (180, 255, 255))]
    },
    "purple": {
        "display": (180, 0, 180),
        "ranges": [((130, 50, 50), (160, 255, 255))]
    },
    "orange": {
        "display": (0, 140, 255),
        "ranges": [((8, 150, 150), (20, 255, 255))]
    },
    "green": {
        "display": (0, 200, 0),
        "ranges": [((35, 80, 80), (75, 255, 255))]
    },
    "brown": {
        "display": (30, 60, 130),
        "ranges": [((5, 80, 40), (18, 200, 150))]
    },
    "black": {
        "display": (50, 50, 50),
        "ranges": [((0, 0, 0), (180, 80, 60))]
    },
}

# Table green (for exclusion when detecting balls on felt)
TABLE_GREEN_LOWER = np.array([35, 40, 40])
TABLE_GREEN_UPPER = np.array([85, 255, 200])


@dataclass
class TrackedBall:
    """Represents a detected ball with position and identity."""
    x: int
    y: int
    radius: int
    color_name: str
    display_color: tuple


@dataclass
class ShotEvent:
    """Represents a detected shot with the cue ball path."""
    start_frame: int
    end_frame: int
    path_points: list = field(default_factory=list)


class BilliardTracker:
    """
    Tracks billiard balls in video frames.

    Detects all balls, specifically tracks the cue ball (white),
    and draws its path after each shot is detected.
    """

    def __init__(
        self,
        min_ball_radius=8,
        max_ball_radius=25,
        motion_threshold=5.0,
        shot_start_speed=8.0,
        shot_end_speed=2.0,
        path_fade_frames=90,
        path_color=(0, 255, 255),
        path_thickness=2,
        label_balls=True,
    ):
        self.min_ball_radius = min_ball_radius
        self.max_ball_radius = max_ball_radius
        self.motion_threshold = motion_threshold
        self.shot_start_speed = shot_start_speed
        self.shot_end_speed = shot_end_speed
        self.path_fade_frames = path_fade_frames
        self.path_color = path_color
        self.path_thickness = path_thickness
        self.label_balls = label_balls

        # State
        self.prev_cue_pos = None
        self.cue_speed = 0.0
        self.in_shot = False
        self.current_path = []
        self.completed_paths = deque()  # (path_points, frame_completed)
        self.frame_count = 0

    def _detect_table_mask(self, hsv_frame):
        """Create a mask of the table felt area to constrain ball search."""
        table_mask = cv2.inRange(hsv_frame, TABLE_GREEN_LOWER, TABLE_GREEN_UPPER)
        # Dilate to fill gaps, then find largest contour as table boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel)
        table_mask = cv2.dilate(table_mask, kernel, iterations=2)
        return table_mask

    def _find_circles(self, gray_frame, mask):
        """Detect circular shapes (balls) using HoughCircles."""
        # Apply mask
        masked_gray = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(masked_gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=self.min_ball_radius * 2,
            param1=50,
            param2=30,
            minRadius=self.min_ball_radius,
            maxRadius=self.max_ball_radius,
        )
        if circles is not None:
            return np.round(circles[0]).astype(int)
        return []

    def _classify_ball(self, hsv_frame, x, y, radius):
        """Classify a detected circle by its dominant color."""
        # Sample the center region of the ball
        sample_r = max(radius // 2, 3)
        h, w = hsv_frame.shape[:2]
        y1 = max(0, y - sample_r)
        y2 = min(h, y + sample_r)
        x1 = max(0, x - sample_r)
        x2 = min(w, x + sample_r)
        roi = hsv_frame[y1:y2, x1:x2]

        if roi.size == 0:
            return "unknown", (128, 128, 128)

        best_match = "unknown"
        best_score = 0
        best_display = (128, 128, 128)

        for color_name, color_info in BALL_COLORS.items():
            total_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            for lower, upper in color_info["ranges"]:
                mask = cv2.inRange(roi, np.array(lower), np.array(upper))
                total_mask = cv2.bitwise_or(total_mask, mask)
            score = cv2.countNonZero(total_mask)
            if score > best_score:
                best_score = score
                best_match = color_name
                best_display = color_info["display"]

        return best_match, best_display

    def _update_shot_tracking(self, cue_ball):
        """Track cue ball motion to detect shots and record paths."""
        if cue_ball is None:
            return

        cx, cy = cue_ball.x, cue_ball.y

        if self.prev_cue_pos is not None:
            dx = cx - self.prev_cue_pos[0]
            dy = cy - self.prev_cue_pos[1]
            self.cue_speed = np.sqrt(dx * dx + dy * dy)
        else:
            self.cue_speed = 0.0

        # Shot start detection
        if not self.in_shot and self.cue_speed > self.shot_start_speed:
            self.in_shot = True
            self.current_path = []
            if self.prev_cue_pos:
                self.current_path.append(self.prev_cue_pos)

        # Record path during shot
        if self.in_shot:
            self.current_path.append((cx, cy))

        # Shot end detection
        if self.in_shot and self.cue_speed < self.shot_end_speed:
            self.in_shot = False
            if len(self.current_path) > 3:
                self.completed_paths.append(
                    (list(self.current_path), self.frame_count)
                )
            self.current_path = []

        self.prev_cue_pos = (cx, cy)

    def _draw_overlays(self, frame, balls):
        """Draw ball labels, cue ball highlight, and shot paths."""
        # Draw all detected balls
        for ball in balls:
            # Circle outline
            cv2.circle(frame, (ball.x, ball.y), ball.radius, ball.display_color, 2)
            if self.label_balls:
                label = ball.color_name
                font_scale = 0.4
                thickness = 1
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                tx = ball.x - tw // 2
                ty = ball.y - ball.radius - 6
                cv2.rectangle(frame, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
                cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, ball.display_color, thickness)

            # Extra highlight for cue ball
            if ball.color_name == "white":
                cv2.circle(frame, (ball.x, ball.y), ball.radius + 3, (0, 255, 255), 1)

        # Draw active shot path
        if self.in_shot and len(self.current_path) > 1:
            pts = np.array(self.current_path, dtype=np.int32)
            cv2.polylines(frame, [pts], False, self.path_color, self.path_thickness + 1)

        # Draw completed paths with fade-out
        paths_to_keep = deque()
        for path_points, completed_frame in self.completed_paths:
            age = self.frame_count - completed_frame
            if age < self.path_fade_frames:
                alpha = 1.0 - (age / self.path_fade_frames)
                color = tuple(int(c * alpha) for c in self.path_color)
                thickness = max(1, int(self.path_thickness * alpha))
                pts = np.array(path_points, dtype=np.int32)
                cv2.polylines(frame, [pts], False, color, thickness)
                paths_to_keep.append((path_points, completed_frame))
        self.completed_paths = paths_to_keep

        return frame

    def process_frame(self, frame):
        """Process a single video frame: detect balls, track cue, draw overlays."""
        self.frame_count += 1
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect table area and find circles within it
        table_mask = self._detect_table_mask(hsv)
        circles = self._find_circles(gray, table_mask)

        # Classify each detected circle
        balls = []
        cue_ball = None
        for (x, y, r) in circles:
            color_name, display_color = self._classify_ball(hsv, x, y, r)
            ball = TrackedBall(x, y, r, color_name, display_color)
            balls.append(ball)
            if color_name == "white":
                cue_ball = ball

        # Track shot and cue ball path
        self._update_shot_tracking(cue_ball)

        # Draw everything
        output = self._draw_overlays(frame.copy(), balls)
        return output, balls


def process_billiard_video(
    input_path,
    output_path,
    cancel_flag=None,
    progress_callback=None,
    min_ball_radius=8,
    max_ball_radius=25,
    shot_start_speed=8.0,
    shot_end_speed=2.0,
    path_fade_frames=90,
    path_color=(0, 255, 255),
    path_thickness=2,
    label_balls=True,
):
    """
    Process a billiard video file: detect balls, track cue ball,
    draw shot paths, and write output video.
    """
    if progress_callback is None:
        progress_callback = lambda msg, pct: None

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    tracker = BilliardTracker(
        min_ball_radius=min_ball_radius,
        max_ball_radius=max_ball_radius,
        shot_start_speed=shot_start_speed,
        shot_end_speed=shot_end_speed,
        path_fade_frames=path_fade_frames,
        path_color=path_color,
        path_thickness=path_thickness,
        label_balls=label_balls,
    )

    progress_callback("Billiard tracking started...", 0)
    frame_idx = 0

    while True:
        if cancel_flag and cancel_flag.is_set():
            break

        ret, frame = cap.read()
        if not ret:
            break

        processed, balls = tracker.process_frame(frame)
        out.write(processed)
        frame_idx += 1

        if total_frames > 0 and frame_idx % 30 == 0:
            pct = int((frame_idx / total_frames) * 100)
            ball_count = len(balls)
            progress_callback(f"Frame {frame_idx}/{total_frames} - {ball_count} balls detected", pct)

    cap.release()
    out.release()
    progress_callback(f"Billiard tracking complete: {output_path}", 100)
    return output_path


# ---------- END: billiard_tracker.py ----------
