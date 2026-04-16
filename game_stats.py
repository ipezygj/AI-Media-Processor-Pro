# ---------- BEGIN: game_stats.py ----------
"""
Game statistics tracker and exporter for billiard games.

Tracks shots, pots, collisions, power, accuracy, and game state.
Exports to JSON and CSV.
"""

import json
import csv
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ShotRecord:
    """Record of a single shot."""
    shot_number: int
    timestamp: float
    frame_start: int
    frame_end: int
    power: float  # 0-100 scale
    cue_start: tuple = (0, 0)
    cue_end: tuple = (0, 0)
    distance: float = 0.0
    collisions: int = 0
    balls_potted: list = field(default_factory=list)
    successful: bool = False  # True if at least one ball was potted


@dataclass
class GameStats:
    """Accumulated game statistics."""
    game_type: str = "unknown"
    start_time: float = 0.0
    total_shots: int = 0
    total_pots: int = 0
    total_collisions: int = 0
    balls_remaining: int = 0
    max_balls_seen: int = 0
    shots: list = field(default_factory=list)

    # Per-player stats (player 1 = index 0, player 2 = index 1)
    player_names: list = field(default_factory=lambda: ["Player 1", "Player 2"])
    player_scores: list = field(default_factory=lambda: [0, 0])
    current_player: int = 0  # 0 or 1

    # Derived stats
    @property
    def avg_power(self):
        if not self.shots:
            return 0.0
        return sum(s.power for s in self.shots) / len(self.shots)

    @property
    def pot_success_rate(self):
        if self.total_shots == 0:
            return 0.0
        return (self.total_pots / self.total_shots) * 100

    @property
    def avg_collisions_per_shot(self):
        if not self.shots:
            return 0.0
        return sum(s.collisions for s in self.shots) / len(self.shots)

    def switch_player(self):
        self.current_player = 1 - self.current_player

    def add_score(self, player_idx, points=1):
        if 0 <= player_idx < len(self.player_scores):
            self.player_scores[player_idx] += points

    def export_json(self, filepath):
        """Export all stats to JSON."""
        data = {
            "game_type": self.game_type,
            "start_time": self.start_time,
            "duration": time.time() - self.start_time if self.start_time else 0,
            "total_shots": self.total_shots,
            "total_pots": self.total_pots,
            "total_collisions": self.total_collisions,
            "avg_power": round(self.avg_power, 1),
            "pot_success_rate": round(self.pot_success_rate, 1),
            "player_names": self.player_names,
            "player_scores": self.player_scores,
            "shots": [
                {
                    "shot_number": s.shot_number,
                    "power": round(s.power, 1),
                    "collisions": s.collisions,
                    "balls_potted": s.balls_potted,
                    "successful": s.successful,
                    "distance": round(s.distance, 1),
                }
                for s in self.shots
            ],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def export_csv(self, filepath):
        """Export shot-by-shot data to CSV."""
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Shot#", "Power", "Collisions", "BallsPotted",
                "Successful", "Distance", "FrameStart", "FrameEnd"
            ])
            for s in self.shots:
                writer.writerow([
                    s.shot_number,
                    round(s.power, 1),
                    s.collisions,
                    ";".join(s.balls_potted),
                    s.successful,
                    round(s.distance, 1),
                    s.frame_start,
                    s.frame_end,
                ])


def detect_game_type(ball_colors):
    """
    Detect the game type based on which ball colors are present.

    Args:
        ball_colors: set of color name strings detected on the table

    Returns:
        str: "8-ball", "9-ball", "snooker", or "unknown"
    """
    color_count = len(ball_colors - {"white"})

    # Snooker: many reds + colored balls, typically 15+ non-white balls
    has_many_reds = ball_colors.issuperset({"red"})

    # 9-ball: yellow through to the 9 (yellow stripe), typically 9 balls + cue
    # 8-ball: full set of 15 + cue, solids and stripes
    if color_count >= 10:
        if has_many_reds:
            return "snooker"
        return "8-ball"
    elif 4 <= color_count <= 9:
        return "9-ball"
    elif color_count >= 2:
        return "8-ball"
    return "unknown"


# ---------- END: game_stats.py ----------
