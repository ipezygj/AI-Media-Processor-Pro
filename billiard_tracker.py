# ---------- BEGIN: billiard_tracker.py ----------
"""
Billiard Ball Tracker - Full featured pool/billiard analysis system.

Features:
    - Ball detection and classification (HSV + Hough circles)
    - Cue ball path tracking and visualization
    - Shot power meter
    - Collision detection with angle display
    - Pocket/pot detection
    - Ball counter
    - Aiming line prediction
    - Cue ball heatmap
    - Shot replay buffer
    - Scoreboard overlay
    - Game type detection (8-ball, 9-ball, snooker)
    - Game statistics tracking and export
"""

import cv2
import numpy as np
import time
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field

from game_stats import GameStats, ShotRecord, detect_game_type


# --- Ball color definitions in HSV ---
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

TABLE_GREEN_LOWER = np.array([35, 40, 40])
TABLE_GREEN_UPPER = np.array([85, 255, 200])

# Pocket positions as fractions of table bounding box (x_frac, y_frac)
POCKET_POSITIONS_FRAC = [
    (0.0, 0.0),   # top-left
    (0.5, 0.0),   # top-center
    (1.0, 0.0),   # top-right
    (0.0, 1.0),   # bottom-left
    (0.5, 1.0),   # bottom-center
    (1.0, 1.0),   # bottom-right
]


@dataclass
class TrackedBall:
    """Represents a detected ball with position and identity."""
    x: int
    y: int
    radius: int
    color_name: str
    display_color: tuple
    speed: float = 0.0


@dataclass
class CollisionEvent:
    """A detected collision between two balls."""
    x: int
    y: int
    ball1_color: str
    ball2_color: str
    angle: float  # degrees
    frame: int


@dataclass
class PotEvent:
    """A ball that was potted (disappeared near a pocket)."""
    ball_color: str
    pocket_idx: int
    frame: int


class BilliardTracker:
    """
    Full-featured billiard ball tracker with game analysis.
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
        enable_heatmap=True,
        enable_collisions=True,
        enable_pockets=True,
        enable_power_meter=True,
        enable_aiming_line=True,
        enable_scoreboard=True,
        replay_buffer_seconds=5,
        fps=30,
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
        self.fps = fps

        # Feature toggles
        self.enable_heatmap = enable_heatmap
        self.enable_collisions = enable_collisions
        self.enable_pockets = enable_pockets
        self.enable_power_meter = enable_power_meter
        self.enable_aiming_line = enable_aiming_line
        self.enable_scoreboard = enable_scoreboard

        # Core state
        self.prev_cue_pos = None
        self.cue_speed = 0.0
        self.cue_max_speed = 0.0  # max speed in current shot
        self.in_shot = False
        self.current_path = []
        self.completed_paths = deque()
        self.frame_count = 0

        # Previous frame ball positions for tracking
        self.prev_balls = {}  # color_name -> (x, y)
        self.prev_ball_set = set()  # color names seen last frame

        # Collision detection
        self.recent_collisions = deque()  # CollisionEvent list
        self.collision_cooldown = {}  # (color1, color2) -> frame of last collision

        # Pocket detection
        self.table_bbox = None  # (x, y, w, h) of table area
        self.pocket_positions = []  # actual pixel positions
        self.pocket_radius = 30  # pixel radius for pocket zone
        self.recent_pots = deque()  # PotEvent list
        self.potted_balls = []  # all potted ball colors
        self.stable_ball_count = 0
        self.ball_count_stable_frames = 0

        # Heatmap
        self.heatmap = None  # accumulated cue ball positions
        self.heatmap_overlay = None

        # Aiming line
        self.cue_stationary_frames = 0
        self.cue_velocity = (0.0, 0.0)

        # Shot replay buffer
        self.replay_buffer = deque(maxlen=int(replay_buffer_seconds * fps))
        self.last_shot_replay = None  # stored replay of last shot

        # Game stats
        self.stats = GameStats(start_time=time.time())
        self._shot_collisions = 0
        self._shot_start_frame = 0
        self._shot_start_pos = (0, 0)
        self._pre_shot_balls = set()

    # ==================== HOTKEY TOGGLES ====================

    # Map of feature name -> attribute name on self
    TOGGLE_MAP = {
        "heatmap": "enable_heatmap",
        "collisions": "enable_collisions",
        "pockets": "enable_pockets",
        "power_meter": "enable_power_meter",
        "aiming_line": "enable_aiming_line",
        "scoreboard": "enable_scoreboard",
        "labels": "label_balls",
        "paths": "_enable_paths",
    }

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def toggle_feature(self, feature_name):
        """Toggle a feature on/off by name. Returns (feature_name, new_state)."""
        attr = self.TOGGLE_MAP.get(feature_name)
        if attr is None:
            return feature_name, None

        if attr == "_enable_paths":
            # Special: paths toggle clears or enables path drawing
            if not hasattr(self, "_enable_paths"):
                self._enable_paths = True
            self._enable_paths = not self._enable_paths
            if not self._enable_paths:
                self.completed_paths.clear()
                self.current_path = []
            return feature_name, self._enable_paths

        current = getattr(self, attr, True)
        new_val = not current
        setattr(self, attr, new_val)
        return feature_name, new_val

    def get_feature_states(self):
        """Return dict of feature_name -> enabled bool."""
        states = {}
        for name, attr in self.TOGGLE_MAP.items():
            if attr == "_enable_paths":
                states[name] = getattr(self, "_enable_paths", True)
            else:
                states[name] = getattr(self, attr, True)
        return states

    # ==================== DETECTION ====================

    def _detect_table_mask(self, hsv_frame):
        """Create table mask using largest green contour with gentle padding."""
        table_mask = cv2.inRange(hsv_frame, TABLE_GREEN_LOWER, TABLE_GREEN_UPPER)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return table_mask

        largest = max(contours, key=cv2.contourArea)
        frame_area = hsv_frame.shape[0] * hsv_frame.shape[1]

        # Table should be at least 5% of the frame
        if cv2.contourArea(largest) < frame_area * 0.05:
            return table_mask

        # Use contour mask but DILATE slightly so balls at edges aren't clipped
        contour_mask = np.zeros(table_mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [largest], -1, 255, -1)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        contour_mask = cv2.dilate(contour_mask, dilate_kernel, iterations=1)

        self.table_bbox = cv2.boundingRect(largest)
        self._table_contour = largest
        self._update_pocket_positions()

        return contour_mask

    def _update_pocket_positions(self):
        """Calculate pocket positions from table bounding box."""
        if self.table_bbox is None:
            return
        tx, ty, tw, th = self.table_bbox
        self.pocket_positions = [
            (int(tx + fx * tw), int(ty + fy * th))
            for fx, fy in POCKET_POSITIONS_FRAC
        ]
        self.pocket_radius = max(20, int(min(tw, th) * 0.06))

    def _find_circles(self, gray_frame, mask):
        """Detect circular shapes using HoughCircles."""
        masked_gray = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)
        blurred = cv2.GaussianBlur(masked_gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=self.min_ball_radius * 2,
            param1=50,
            param2=30,
            minRadius=self.min_ball_radius,
            maxRadius=self.max_ball_radius,
        )
        if circles is None:
            return []

        candidates = np.round(circles[0]).astype(int)

        # Must be inside the table mask
        valid = []
        for (x, y, r) in candidates:
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                if mask[y, x] > 0:
                    valid.append((x, y, r))

        # Size consistency: reject extreme outliers only (>80% off median)
        if len(valid) >= 3:
            radii = [r for (_, _, r) in valid]
            median_r = np.median(radii)
            valid = [(x, y, r) for (x, y, r) in valid
                     if abs(r - median_r) <= median_r * 0.8]

        return valid

    def _classify_ball(self, hsv_frame, x, y, radius):
        """Classify a detected circle by its dominant color."""
        sample_r = max(radius // 2, 3)
        h, w = hsv_frame.shape[:2]
        roi = hsv_frame[max(0, y-sample_r):min(h, y+sample_r),
                        max(0, x-sample_r):min(w, x+sample_r)]
        if roi.size == 0:
            return "unknown", (128, 128, 128)

        best_match, best_score, best_display = "unknown", 0, (128, 128, 128)
        for color_name, info in BALL_COLORS.items():
            total = np.zeros(roi.shape[:2], dtype=np.uint8)
            for lower, upper in info["ranges"]:
                total = cv2.bitwise_or(total, cv2.inRange(roi, np.array(lower), np.array(upper)))
            score = cv2.countNonZero(total)
            if score > best_score:
                best_score, best_match, best_display = score, color_name, info["display"]

        return best_match, best_display

    # ==================== TRACKING ====================

    def _update_shot_tracking(self, cue_ball):
        """Track cue ball motion, detect shots, record paths."""
        if cue_ball is None:
            return

        cx, cy = cue_ball.x, cue_ball.y

        if self.prev_cue_pos is not None:
            dx = cx - self.prev_cue_pos[0]
            dy = cy - self.prev_cue_pos[1]
            self.cue_speed = np.sqrt(dx * dx + dy * dy)
            self.cue_velocity = (dx, dy)
        else:
            self.cue_speed = 0.0
            self.cue_velocity = (0.0, 0.0)

        # Stationary detection for aiming line
        if self.cue_speed < 1.5:
            self.cue_stationary_frames += 1
        else:
            self.cue_stationary_frames = 0

        # Shot start
        if not self.in_shot and self.cue_speed > self.shot_start_speed:
            self.in_shot = True
            self.current_path = []
            self.cue_max_speed = 0.0
            self._shot_collisions = 0
            self._shot_start_frame = self.frame_count
            self._shot_start_pos = self.prev_cue_pos or (cx, cy)
            self._pre_shot_balls = set(self.prev_balls.keys()) - {"white"}
            if self.prev_cue_pos:
                self.current_path.append(self.prev_cue_pos)

        # During shot
        if self.in_shot:
            self.current_path.append((cx, cy))
            self.cue_max_speed = max(self.cue_max_speed, self.cue_speed)

        # Shot end
        if self.in_shot and self.cue_speed < self.shot_end_speed:
            self.in_shot = False
            if len(self.current_path) > 3:
                self.completed_paths.append((list(self.current_path), self.frame_count))
                self.last_shot_replay = list(self.replay_buffer)
                self._record_shot_stats(cx, cy)
            self.current_path = []

        self.prev_cue_pos = (cx, cy)

    def _record_shot_stats(self, end_x, end_y):
        """Record completed shot statistics."""
        self.stats.total_shots += 1
        power = min(100.0, (self.cue_max_speed / 30.0) * 100.0)
        start = self._shot_start_pos
        distance = np.sqrt((end_x - start[0])**2 + (end_y - start[1])**2)

        # Check which balls were potted during this shot
        current_ball_colors = set(self.prev_balls.keys()) - {"white"}
        potted_this_shot = list(self._pre_shot_balls - current_ball_colors)

        shot = ShotRecord(
            shot_number=self.stats.total_shots,
            timestamp=time.time(),
            frame_start=self._shot_start_frame,
            frame_end=self.frame_count,
            power=power,
            cue_start=start,
            cue_end=(end_x, end_y),
            distance=distance,
            collisions=self._shot_collisions,
            balls_potted=potted_this_shot,
            successful=len(potted_this_shot) > 0,
        )
        self.stats.shots.append(shot)

        if potted_this_shot:
            self.stats.total_pots += len(potted_this_shot)
            self.stats.add_score(self.stats.current_player, len(potted_this_shot))
        else:
            self.stats.switch_player()

    # ==================== COLLISION DETECTION ====================

    def _detect_collisions(self, balls):
        """Detect when two balls are close enough to be colliding."""
        if not self.enable_collisions:
            return

        for i, b1 in enumerate(balls):
            for b2 in balls[i+1:]:
                dist = np.sqrt((b1.x - b2.x)**2 + (b1.y - b2.y)**2)
                touch_dist = (b1.radius + b2.radius) * 1.3

                if dist < touch_dist:
                    pair = tuple(sorted([b1.color_name, b2.color_name]))
                    last = self.collision_cooldown.get(pair, 0)
                    if self.frame_count - last > 15:  # cooldown frames
                        angle = np.degrees(np.arctan2(b2.y - b1.y, b2.x - b1.x))
                        collision = CollisionEvent(
                            x=(b1.x + b2.x) // 2,
                            y=(b1.y + b2.y) // 2,
                            ball1_color=b1.color_name,
                            ball2_color=b2.color_name,
                            angle=angle,
                            frame=self.frame_count,
                        )
                        self.recent_collisions.append(collision)
                        self.collision_cooldown[pair] = self.frame_count
                        self.stats.total_collisions += 1
                        if self.in_shot:
                            self._shot_collisions += 1

    # ==================== POCKET DETECTION ====================

    def _detect_pots(self, balls):
        """Detect when a ball disappears near a pocket."""
        if not self.enable_pockets or not self.pocket_positions:
            return

        current_colors = {b.color_name for b in balls}

        for color in self.prev_ball_set - current_colors:
            if color == "white":
                continue  # cue ball scratch handled separately
            prev_pos = self.prev_balls.get(color)
            if prev_pos is None:
                continue

            # Check if the ball was near a pocket when it vanished
            for idx, (px, py) in enumerate(self.pocket_positions):
                dist = np.sqrt((prev_pos[0] - px)**2 + (prev_pos[1] - py)**2)
                if dist < self.pocket_radius * 3:
                    pot = PotEvent(ball_color=color, pocket_idx=idx, frame=self.frame_count)
                    self.recent_pots.append(pot)
                    self.potted_balls.append(color)
                    break

    # ==================== HEATMAP ====================

    def _update_heatmap(self, cue_ball, frame_shape):
        """Accumulate cue ball position into heatmap."""
        if not self.enable_heatmap or cue_ball is None:
            return

        h, w = frame_shape[:2]
        if self.heatmap is None:
            self.heatmap = np.zeros((h, w), dtype=np.float32)

        cv2.circle(self.heatmap, (cue_ball.x, cue_ball.y), 15, 1.0, -1)

    def get_heatmap_overlay(self, frame_shape):
        """Generate a colored heatmap overlay image."""
        if self.heatmap is None:
            return None

        normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
        # Make areas with no data transparent (black)
        mask = normalized > 1
        overlay = np.zeros_like(colored)
        overlay[mask] = colored[mask]
        return overlay

    # ==================== DRAWING ====================

    def _draw_overlays(self, frame, balls, cue_ball):
        """Draw all visual overlays on the frame."""
        h, w = frame.shape[:2]

        # Heatmap (drawn first, semi-transparent under everything)
        if self.enable_heatmap and self.heatmap is not None:
            overlay = self.get_heatmap_overlay(frame.shape)
            if overlay is not None:
                mask = np.any(overlay > 0, axis=2)
                frame[mask] = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)[mask]

        # Pocket zones
        if self.enable_pockets:
            for px, py in self.pocket_positions:
                cv2.circle(frame, (px, py), self.pocket_radius, (0, 0, 100), 1)

        # Ball circles and labels
        for ball in balls:
            cv2.circle(frame, (ball.x, ball.y), ball.radius, ball.display_color, 2)
            if self.label_balls:
                label = ball.color_name
                fs, th = 0.4, 1
                (tw, txh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
                tx = ball.x - tw // 2
                ty = ball.y - ball.radius - 6
                cv2.rectangle(frame, (tx-2, ty-txh-2), (tx+tw+2, ty+2), (0, 0, 0), -1)
                cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, ball.display_color, th)

            if ball.color_name == "white":
                cv2.circle(frame, (ball.x, ball.y), ball.radius + 3, (0, 255, 255), 1)

        # Aiming line
        if self.enable_aiming_line and cue_ball and self.cue_stationary_frames > 15:
            self._draw_aiming_line(frame, cue_ball)

        # Active shot path
        paths_enabled = getattr(self, "_enable_paths", True)
        if paths_enabled and self.in_shot and len(self.current_path) > 1:
            pts = np.array(self.current_path, dtype=np.int32)
            cv2.polylines(frame, [pts], False, self.path_color, self.path_thickness + 1)

        # Completed paths with fade
        paths_to_keep = deque()
        for path_points, completed_frame in self.completed_paths:
            age = self.frame_count - completed_frame
            if age < self.path_fade_frames:
                if paths_enabled:
                    alpha = 1.0 - (age / self.path_fade_frames)
                    color = tuple(int(c * alpha) for c in self.path_color)
                    thickness = max(1, int(self.path_thickness * alpha))
                    pts = np.array(path_points, dtype=np.int32)
                    cv2.polylines(frame, [pts], False, color, thickness)
                paths_to_keep.append((path_points, completed_frame))
        self.completed_paths = paths_to_keep

        # Collision flash effects
        if self.enable_collisions:
            self._draw_collisions(frame)

        # Pot notifications
        if self.enable_pockets:
            self._draw_pot_notifications(frame)

        # Power meter
        if self.enable_power_meter:
            self._draw_power_meter(frame)

        # Ball counter
        self._draw_ball_counter(frame, balls)

        # Scoreboard
        if self.enable_scoreboard:
            self._draw_scoreboard(frame)

        return frame

    def _draw_aiming_line(self, frame, cue_ball):
        """Draw a predicted aiming line when cue ball is stationary."""
        if self.prev_cue_pos is None:
            return

        # Use average of last few velocity samples for direction
        vx, vy = self.cue_velocity
        if abs(vx) < 0.1 and abs(vy) < 0.1:
            return

        # Extend the line in the direction of last movement
        length = 200
        mag = max(np.sqrt(vx*vx + vy*vy), 0.01)
        dx, dy = vx / mag, vy / mag
        end_x = int(cue_ball.x + dx * length)
        end_y = int(cue_ball.y + dy * length)

        # Dashed line effect
        for i in range(0, length, 12):
            sx = int(cue_ball.x + dx * i)
            sy = int(cue_ball.y + dy * i)
            ex = int(cue_ball.x + dx * min(i + 6, length))
            ey = int(cue_ball.y + dy * min(i + 6, length))
            cv2.line(frame, (sx, sy), (ex, ey), (100, 255, 100), 1)

    def _draw_collisions(self, frame):
        """Draw collision flash effects."""
        collisions_to_keep = deque()
        for c in self.recent_collisions:
            age = self.frame_count - c.frame
            if age < 20:  # show for 20 frames
                alpha = 1.0 - (age / 20.0)
                radius = int(20 + age * 2)
                color = (0, int(255 * alpha), int(255 * alpha))
                cv2.circle(frame, (c.x, c.y), radius, color, 2)

                if age < 10:
                    angle_text = f"{abs(c.angle):.0f}"
                    cv2.putText(frame, angle_text, (c.x + 15, c.y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                collisions_to_keep.append(c)
        self.recent_collisions = collisions_to_keep

    def _draw_pot_notifications(self, frame):
        """Draw POTTED! notifications."""
        h, w = frame.shape[:2]
        pots_to_keep = deque()
        for i, pot in enumerate(self.recent_pots):
            age = self.frame_count - pot.frame
            if age < 60:  # show for 2 seconds at 30fps
                alpha = 1.0 - (age / 60.0)
                y_offset = int(h * 0.15 + i * 35 - age * 0.5)
                text = f"POTTED! {pot.ball_color.upper()}"
                color = BALL_COLORS.get(pot.ball_color, {}).get("display", (255, 255, 255))
                faded_color = tuple(int(c * alpha) for c in color)
                cv2.putText(frame, text, (w // 2 - 80, max(40, y_offset)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, faded_color, 2)
                pots_to_keep.append(pot)
        self.recent_pots = pots_to_keep

    def _draw_power_meter(self, frame):
        """Draw a power meter bar on the side."""
        h, w = frame.shape[:2]
        if not self.in_shot and self.cue_max_speed < 1:
            return

        # Power as percentage (max ~30px/frame movement = 100%)
        power = min(1.0, self.cue_speed / 30.0) if self.in_shot else 0.0
        max_power = min(1.0, self.cue_max_speed / 30.0)

        # Bar dimensions
        bar_x = w - 40
        bar_y = 80
        bar_w = 20
        bar_h = h - 160
        filled_h = int(bar_h * power)
        max_h = int(bar_h * max_power)

        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 1)

        # Max power line
        if max_power > 0:
            max_y = bar_y + bar_h - max_h
            cv2.line(frame, (bar_x - 5, max_y), (bar_x + bar_w + 5, max_y), (0, 200, 255), 2)

        # Filled bar (green -> yellow -> red)
        if filled_h > 0:
            fill_y = bar_y + bar_h - filled_h
            if power < 0.33:
                bar_color = (0, 200, 0)
            elif power < 0.66:
                bar_color = (0, 200, 200)
            else:
                bar_color = (0, 0, 255)
            cv2.rectangle(frame, (bar_x, fill_y), (bar_x + bar_w, bar_y + bar_h), bar_color, -1)

        # Label
        pct = int(max_power * 100) if not self.in_shot else int(power * 100)
        cv2.putText(frame, f"{pct}%", (bar_x - 10, bar_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, "PWR", (bar_x - 5, bar_y + bar_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    def _draw_ball_counter(self, frame, balls):
        """Draw ball count indicator."""
        h, w = frame.shape[:2]
        non_cue = [b for b in balls if b.color_name != "white"]
        count = len(non_cue)
        potted = len(self.potted_balls)

        text = f"On table: {count} | Potted: {potted}"
        cv2.rectangle(frame, (0, h - 30), (250, h), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def _draw_scoreboard(self, frame):
        """Draw player scoreboard overlay."""
        h, w = frame.shape[:2]
        sb_w, sb_h = 220, 80
        sx, sy = 10, 40

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (sx, sy), (sx + sb_w, sy + sb_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (sx, sy), (sx + sb_w, sy + sb_h), (80, 80, 80), 1)

        # Game type
        cv2.putText(frame, self.stats.game_type.upper(), (sx + 5, sy + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Player 1
        p1_color = (0, 255, 255) if self.stats.current_player == 0 else (150, 150, 150)
        marker = ">" if self.stats.current_player == 0 else " "
        cv2.putText(frame, f"{marker} {self.stats.player_names[0]}: {self.stats.player_scores[0]}",
                    (sx + 5, sy + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, p1_color, 1)

        # Player 2
        p2_color = (0, 255, 255) if self.stats.current_player == 1 else (150, 150, 150)
        marker = ">" if self.stats.current_player == 1 else " "
        cv2.putText(frame, f"{marker} {self.stats.player_names[1]}: {self.stats.player_scores[1]}",
                    (sx + 5, sy + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, p2_color, 1)

        # Shot count
        cv2.putText(frame, f"Shots: {self.stats.total_shots}",
                    (sx + 5, sy + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

    # ==================== MAIN PROCESSING ====================

    def process_frame(self, frame):
        """Process a single video frame with full analysis."""
        self.frame_count += 1
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Store frame in replay buffer
        self.replay_buffer.append(frame.copy())

        # Detect table and find circles within it
        table_mask = self._detect_table_mask(hsv)
        circles = self._find_circles(gray, table_mask)

        # Classify each circle
        balls = []
        cue_ball = None
        for (x, y, r) in circles:
            color_name, display_color = self._classify_ball(hsv, x, y, r)
            ball = TrackedBall(x, y, r, color_name, display_color)
            balls.append(ball)
            if color_name == "white":
                cue_ball = ball

        # Game type detection (first few frames)
        if self.frame_count < 30:
            ball_colors = {b.color_name for b in balls}
            self.stats.game_type = detect_game_type(ball_colors)
            self.stats.max_balls_seen = max(self.stats.max_balls_seen, len(balls))

        # Track max ball count
        self.stats.balls_remaining = len([b for b in balls if b.color_name != "white"])

        # Collision detection
        self._detect_collisions(balls)

        # Pot detection
        self._detect_pots(balls)

        # Update previous ball state
        self.prev_ball_set = {b.color_name for b in balls}
        self.prev_balls = {b.color_name: (b.x, b.y) for b in balls}

        # Shot tracking
        self._update_shot_tracking(cue_ball)

        # Heatmap
        self._update_heatmap(cue_ball, frame.shape)

        # Draw everything
        output = self._draw_overlays(frame.copy(), balls, cue_ball)
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
    stats_output_dir=None,
    tracker=None,
    show_preview=True,
):
    """
    Process a billiard video file with full analysis.
    Exports stats to JSON/CSV if stats_output_dir is provided.
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

    if tracker is None:
        tracker = BilliardTracker(
            min_ball_radius=min_ball_radius,
            max_ball_radius=max_ball_radius,
            shot_start_speed=shot_start_speed,
            shot_end_speed=shot_end_speed,
            path_fade_frames=path_fade_frames,
            path_color=path_color,
            path_thickness=path_thickness,
            label_balls=label_balls,
            fps=fps,
        )

    progress_callback("Billiard tracking started...", 0)
    frame_idx = 0

    window_name = "Billiard Tracker - Preview (Q to quit)"
    if show_preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Scale preview to reasonable size
        preview_w = min(width, 960)
        preview_h = int(height * (preview_w / width))
        cv2.resizeWindow(window_name, preview_w, preview_h)

    try:
        while True:
            if cancel_flag and cancel_flag.is_set():
                break

            ret, frame = cap.read()
            if not ret:
                break

            processed, balls = tracker.process_frame(frame)
            out.write(processed)
            frame_idx += 1

            if show_preview:
                cv2.imshow(window_name, processed)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break

            if total_frames > 0 and frame_idx % 30 == 0:
                pct = int((frame_idx / total_frames) * 100)
                ball_count = len(balls)
                progress_callback(
                    f"Frame {frame_idx}/{total_frames} - {ball_count} balls | "
                    f"Shots: {tracker.stats.total_shots} | Pots: {tracker.stats.total_pots}",
                    pct,
                )
    finally:
        if show_preview:
            cv2.destroyAllWindows()

    cap.release()
    out.release()

    # Export stats
    out_dir = stats_output_dir or str(Path(output_path).parent)
    base_name = Path(output_path).stem
    tracker.stats.export_json(str(Path(out_dir) / f"{base_name}_stats.json"))
    tracker.stats.export_csv(str(Path(out_dir) / f"{base_name}_stats.csv"))
    progress_callback(f"Stats exported to {out_dir}", 100)

    progress_callback(f"Billiard tracking complete: {output_path}", 100)
    return output_path


# ---------- END: billiard_tracker.py ----------
