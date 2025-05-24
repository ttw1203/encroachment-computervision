"""
Segment-based speed calibration for encZone.py
---------------------------------------------
• Expose the entry/exit-line logic as an importable class.
• *Off* unless encZone.py is started with --segment_speed.
"""
from __future__ import annotations
import math, os, json, yaml, csv
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

# ───────────────────────────────────────── helpers ──────────────────────────
def line_side(pt, a, b) -> int:
    """+1 if *pt* is left of vector AB, –1 if right, 0 if on the line."""
    return int(np.sign((b[0] - a[0]) * (pt[1] - a[1]) -
                       (b[1] - a[1]) * (pt[0] - a[0])))  # :contentReference[oaicite:0]{index=0}

def load_segments(path: str):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return (np.asarray(data["segment_entry"], np.int32),
            np.asarray(data["segment_exit"],  np.int32))          # :contentReference[oaicite:1]{index=1}

# ──────────────────────────────────────── class ─────────────────────────────
class SegmentSpeedCalibrator:
    ENTRY_COLOR = (0, 255, 255)   # BGR yellow
    EXIT_COLOR  = (0,   0, 255)   # BGR red
    THICKNESS   = 3

    def __init__(self, zones_file: str, fps: float):
        self.fps         = fps
        self.entry_line, self.exit_line = load_segments(zones_file)
        self.state: dict[int, dict] = {}      # per-ID memory
        self.results: list[list]    = []      # CSV rows

    # ───────── per-frame update ─────────
    def update(self, tid: int, p_cur: tuple[float, float],
               world_m: tuple[float, float], frame_idx: int):
        st = self.state.setdefault(
            tid, dict(p_prev=p_cur,
                      side_prev=line_side(p_cur, *self.entry_line)))

        # ENTRY
        if ('t0' not in st and
            st['side_prev'] * line_side(p_cur, *self.entry_line) < 0):
            st.update(t0=frame_idx, x0m=world_m[0], y0m=world_m[1])

        # EXIT
        if 't0' in st and 't_exit' not in st:
            if (line_side(st['p_prev'], *self.exit_line) *
                line_side(p_cur,        *self.exit_line) < 0):
                t1  = frame_idx
                dt  = (t1 - st['t0']) / self.fps
                if dt:                                                     # guard /0
                    dx = world_m[0] - st['x0m']
                    dy = world_m[1] - st['y0m']
                    v  = math.hypot(dx, dy) / dt
                    self.results.append([
                        tid, st['t0'], t1,
                        round(math.hypot(dx, dy), 2),
                        round(dt, 3),
                        round(v, 2),
                        round(v * 3.6, 2)
                    ])                                                     # :contentReference[oaicite:2]{index=2}
                st['t_exit'] = t1

        # bookkeeping for next frame
        st['p_prev']   = p_cur
        st['side_prev'] = line_side(p_cur, *self.entry_line)

    # ───────── overlays ─────────
    def draw(self, frame):
        cv2.line(frame, tuple(self.entry_line[0]), tuple(self.entry_line[1]),
                 self.ENTRY_COLOR, self.THICKNESS)
        cv2.arrowedLine(frame, tuple(self.exit_line[0]), tuple(self.exit_line[1]),
                        self.EXIT_COLOR, self.THICKNESS, tipLength=0.05)
        # centre labels
        mid = lambda a, b: (int((a[0]+b[0])*0.5), int((a[1]+b[1])*0.5))
        cv2.putText(frame, "ENTRY", mid(*self.entry_line),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.ENTRY_COLOR, 2, cv2.LINE_AA)
        cv2.putText(frame, "EXIT",  mid(*self.exit_line),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.EXIT_COLOR, 2, cv2.LINE_AA)

    # ───────── dump CSV at the end ─────────
    def dump_csv(self, out_dir="results"):
        if not self.results:
            return
        Path(out_dir).mkdir(exist_ok=True)
        ts = datetime.now().strftime("%d%m%Y_%H%M%S")
        fn = Path(out_dir) / f"segment_speeds_{ts}.csv"
        with open(fn, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["vehicle_id", "t_entry_f", "t_exit_f",
                        "distance_m", "time_s", "speed_m_s", "speed_km_h"])
            w.writerows(self.results)
        print(f"[segment-speed] {len(self.results)} rows → {fn}")
