"""
Scene 8: Mapping New Data

Uses trained SOM weights (identical pipeline to scene7_umatrix.py).
No retraining — only inference (BMU lookup) happens during animation.

Parts:
  1 -- Scene-7 end state: trained grid (cluster-coloured) + data cloud
  2 -- Introduce 1st new point → "New observation"
  3 -- BMU search: lines to 4 candidates, winner highlighted
  4 -- Grid connection: new point → grid node
  5 -- Repeat fast for 2 more new points
  6 -- Final view  → "High-dimensional data → 2D representation"
  7 -- Fade new points, keep grid

Usage:
    manim -ql --disable_caching scene8_mapping.py Scene8Mapping
    manim -qh --fps 60 scene8_mapping.py Scene8Mapping
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from manim import *

DEFAULT_FONT = "CMU Serif"
Text.set_default(font=DEFAULT_FONT)

# ====================================================================
# STEP 1 — DATA  (identical seed / params to scene7)
# ====================================================================
_pts, _raw_labels = make_blobs(
    n_samples=300,
    centers=4,
    n_features=3,
    cluster_std=0.65,
    random_state=42,
)
_scaler = MinMaxScaler(feature_range=(-2.0, 2.0))
DATA_PTS = _scaler.fit_transform(_pts)      # (300, 3) in [-2, 2]^3
DATA_PTS[:, 0] += 2.2                        # shift cloud to RIGHT

# ====================================================================
# STEP 2 — SOM TRAINING  (identical to scene7_umatrix.py)
# ====================================================================
GRID_ROWS = 4
GRID_COLS = 4
N_NODES   = GRID_ROWS * GRID_COLS
SOM_ITERS = 300
ALPHA0    = 0.5
SIGMA0    = 2.0

_rng_som = np.random.default_rng(42)
_grid_coords = np.array(
    [[r, c] for r in range(GRID_ROWS) for c in range(GRID_COLS)],
    dtype=float,
)  # (16, 2)

weights = _rng_som.uniform(-2.0, 2.0, size=(N_NODES, 3))

for _t in range(SOM_ITERS):
    _xi    = DATA_PTS[_rng_som.integers(0, len(DATA_PTS))]
    _bmu_t = int(np.argmin(np.linalg.norm(weights - _xi, axis=1)))
    _alpha = ALPHA0 * np.exp(-_t / SOM_ITERS)
    _sigma = max(
        SIGMA0 * np.exp(-_t / (SOM_ITERS / np.log(SIGMA0 + 1e-9))),
        0.3,
    )
    _gc_d2 = np.sum((_grid_coords - _grid_coords[_bmu_t]) ** 2, axis=1)
    _h     = np.exp(-_gc_d2 / (2.0 * _sigma ** 2))
    weights += _alpha * _h[:, None] * (_xi - weights)

weights[:, 0] += 2.2   # shift to RIGHT

# ====================================================================
# STEP 3 — NODE LABELS (majority-vote)
# ====================================================================
def find_bmu(x: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(weights - x, axis=1)))

_point_bmu = np.array([find_bmu(DATA_PTS[i]) for i in range(len(DATA_PTS))])

node_labels = np.zeros(N_NODES, dtype=int)
for _ni in range(N_NODES):
    _assigned = _raw_labels[_point_bmu == _ni]
    if len(_assigned) > 0:
        node_labels[_ni] = int(np.bincount(_assigned).argmax())

for _ in range(4):
    for _ni in range(N_NODES):
        if (_point_bmu == _ni).sum() == 0:
            _d = np.linalg.norm(weights - weights[_ni], axis=1)
            _d[_ni] = 999.0
            node_labels[_ni] = node_labels[int(np.argmin(_d))]

# ====================================================================
# STEP 4 — NEW TEST POINTS (one per cluster, deterministic)
# ====================================================================
_rng_new = np.random.default_rng(77)
new_test_pts = []
for _cid in range(4):
    _candidates = np.where(_raw_labels == _cid)[0]
    _idx = _candidates[_rng_new.integers(len(_candidates))]
    new_test_pts.append(DATA_PTS[_idx].copy())   # already x-shifted
new_test_pts = np.array(new_test_pts)            # (4, 3)

new_bmus = [find_bmu(new_test_pts[i]) for i in range(4)]

# ====================================================================
# LAYOUT CONSTANTS
# ====================================================================
GRID_SPACING = 0.55
GRID_CX      = -3.6
GRID_CY      =  0.0
CLUSTER_COLORS = [RED_B, GREEN_B, BLUE_C, YELLOW]
NEW_PT_COLOR   = WHITE


def grid_pos(r: int, c: int) -> np.ndarray:
    x = GRID_CX + (c - (GRID_COLS - 1) / 2.0) * GRID_SPACING
    y = GRID_CY + (r - (GRID_ROWS - 1) / 2.0) * GRID_SPACING
    return np.array([x, y, 0.0])


GRID_POSITIONS = [grid_pos(r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS)]


def w_pos(ni: int) -> np.ndarray:
    return np.array([weights[ni, 0], weights[ni, 1], weights[ni, 2]])


# ====================================================================
class Scene8Mapping(ThreeDScene):
    """Mapping new data to a trained SOM grid."""

    def construct(self):
        self.set_camera_orientation(phi=55 * DEGREES, theta=-60 * DEGREES)
        self.camera.frame.scale(1 / 0.9)

        # ----------------------------------------------------------------
        # PART 1 — SCENE-7 END STATE
        # ----------------------------------------------------------------
        h_lines = VGroup(*[
            Line(
                GRID_POSITIONS[r * GRID_COLS + c],
                GRID_POSITIONS[r * GRID_COLS + c + 1],
                color=BLUE_E, stroke_width=1.2,
            )
            for r in range(GRID_ROWS) for c in range(GRID_COLS - 1)
        ])
        v_lines = VGroup(*[
            Line(
                GRID_POSITIONS[r * GRID_COLS + c],
                GRID_POSITIONS[(r + 1) * GRID_COLS + c],
                color=BLUE_E, stroke_width=1.2,
            )
            for r in range(GRID_ROWS - 1) for c in range(GRID_COLS)
        ])
        grid_nodes = VGroup(*[
            Dot3D(
                point=GRID_POSITIONS[i],
                radius=0.08,
                color=CLUSTER_COLORS[node_labels[i]],
            )
            for i in range(N_NODES)
        ])
        w_dots = VGroup(*[
            Dot3D(point=w_pos(i), radius=0.09, color=CLUSTER_COLORS[node_labels[i]])
            for i in range(N_NODES)
        ])
        data_dots = VGroup(*[
            Dot3D(
                point=np.array([DATA_PTS[i, 0], DATA_PTS[i, 1], DATA_PTS[i, 2]]),
                radius=0.03,
                color=CLUSTER_COLORS[_raw_labels[i]],
            )
            for i in range(len(DATA_PTS))
        ])

        data_axes = ThreeDAxes(
            x_range=[-2.2, 2.2, 1], y_range=[-2.2, 2.2, 1], z_range=[-2.2, 2.2, 1],
            x_length=4.4, y_length=4.4, z_length=4.4,
            tips=False,
            axis_config={"stroke_color": GREY_D, "stroke_width": 0.6,
                         "include_ticks": False},
        ).shift(RIGHT * 2.2)

        lbl_using = Text("Using the map", font_size=30, color=WHITE)
        lbl_using.to_edge(UL, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl_using)

        self.play(
            Create(h_lines), Create(v_lines),
            LaggedStart(*[FadeIn(d) for d in grid_nodes], lag_ratio=0.04),
            LaggedStart(*[FadeIn(d) for d in w_dots],    lag_ratio=0.04),
            LaggedStart(*[FadeIn(d) for d in data_dots], lag_ratio=0.005),
            FadeIn(data_axes),
            FadeIn(lbl_using),
            run_time=2.5,
        )
        self.wait(1.0)

        # ----------------------------------------------------------------
        # PART 2 — INTRODUCE FIRST NEW POINT
        # ----------------------------------------------------------------
        pt0 = new_test_pts[0]
        new_dot = Dot3D(point=pt0, radius=0.14, color=NEW_PT_COLOR)
        new_dot_lbl = Text("x  (new)", font_size=18, color=WHITE)
        new_dot_lbl.move_to(pt0 + np.array([0.0, 0.35, 0.0]))

        lbl_new_obs = Text("New observation", font_size=28, color=YELLOW)
        lbl_new_obs.to_edge(UR, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl_new_obs)

        self.play(
            FadeIn(new_dot),
            FadeIn(new_dot_lbl),
            FadeIn(lbl_new_obs),
            run_time=1.0,
        )
        # Pulse
        self.play(new_dot.animate.scale(1.5), run_time=0.3)
        self.play(new_dot.animate.scale(1 / 1.5), run_time=0.3)
        self.wait(0.5)

        # ----------------------------------------------------------------
        # PART 3 — FIND BMU  (4 nearest candidate lines)
        # ----------------------------------------------------------------
        bmu0 = new_bmus[0]

        # 4 closest weight nodes to the new point
        _dists_to_new = np.linalg.norm(weights - pt0, axis=1)
        _closest4 = np.argsort(_dists_to_new)[:4].tolist()

        cand_lines = VGroup(*[
            Line(
                pt0,
                w_pos(ni),
                color=GREY_B,
                stroke_width=1.5,
            )
            for ni in _closest4
        ])

        lbl_find = Text("Find closest node", font_size=28, color=WHITE)
        lbl_find.to_edge(UR, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl_find)

        self.play(
            FadeOut(lbl_new_obs),
            FadeIn(lbl_find),
            LaggedStart(*[Create(ln) for ln in cand_lines], lag_ratio=0.15),
            run_time=1.2,
        )
        self.wait(0.4)

        # Highlight winner line, fade others
        winner_idx_in4 = _closest4.index(bmu0)
        non_winner_lines = VGroup(*[cand_lines[k] for k in range(4) if k != winner_idx_in4])
        winner_line = cand_lines[winner_idx_in4]

        self.play(
            non_winner_lines.animate.set_opacity(0.12),
            winner_line.animate.set_stroke(color=GREEN_B, width=3.0),
            w_dots[bmu0].animate.set_color(WHITE).scale(1.6),
            run_time=0.8,
        )
        self.wait(0.5)

        # ----------------------------------------------------------------
        # PART 4 — DRAW CONNECTION TO GRID NODE
        # ----------------------------------------------------------------
        grid_bmu_pos = GRID_POSITIONS[bmu0]
        w_bmu_pos    = w_pos(bmu0)

        map_line = Line(pt0, grid_bmu_pos, color=YELLOW, stroke_width=2.0)

        lbl_mapped = Text("Mapped to grid", font_size=28, color=WHITE)
        lbl_mapped.to_edge(UR, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl_mapped)

        self.play(
            FadeOut(lbl_find),
            FadeIn(lbl_mapped),
            FadeOut(cand_lines),
            Create(map_line),
            run_time=1.0,
        )
        # Highlight the grid node
        self.play(
            grid_nodes[bmu0].animate.scale(1.8).set_color(WHITE),
            run_time=0.6,
        )
        self.wait(0.8)

        # Restore, fade map artefacts
        self.play(
            FadeOut(map_line),
            FadeOut(new_dot_lbl),
            w_dots[bmu0].animate.set_color(CLUSTER_COLORS[node_labels[bmu0]]).scale(1 / 1.6),
            grid_nodes[bmu0].animate.scale(1 / 1.8).set_color(CLUSTER_COLORS[node_labels[bmu0]]),
            run_time=0.6,
        )

        # Keep new dot dimmed (marked as mapped)
        self.play(new_dot.animate.set_opacity(0.5), run_time=0.3)
        self.play(FadeOut(lbl_mapped), run_time=0.4)
        self.wait(0.3)

        # ----------------------------------------------------------------
        # PART 5 — REPEAT FAST FOR 2 MORE POINTS
        # ----------------------------------------------------------------
        mapped_dots = VGroup(new_dot)
        map_markers = VGroup()        # small grid-node markers

        for rep_i in range(1, 3):
            pt_i   = new_test_pts[rep_i]
            bmu_i  = new_bmus[rep_i]

            # New dot
            nd = Dot3D(point=pt_i, radius=0.14, color=NEW_PT_COLOR)
            self.play(FadeIn(nd), run_time=0.4)
            self.play(nd.animate.scale(1.4), run_time=0.2)
            self.play(nd.animate.scale(1 / 1.4), run_time=0.2)

            # Closest 4 candidates
            _d_i = np.linalg.norm(weights - pt_i, axis=1)
            _c4  = np.argsort(_d_i)[:4].tolist()
            clines_i = VGroup(*[
                Line(pt_i, w_pos(ni), color=GREY_B, stroke_width=1.2)
                for ni in _c4
            ])
            self.play(LaggedStart(*[Create(ln) for ln in clines_i], lag_ratio=0.1),
                      run_time=0.7)

            # Highlight winner
            wi_idx = _c4.index(bmu_i)
            non_wi = VGroup(*[clines_i[k] for k in range(4) if k != wi_idx])
            self.play(
                non_wi.animate.set_opacity(0.1),
                clines_i[wi_idx].animate.set_stroke(color=GREEN_B, width=2.5),
                w_dots[bmu_i].animate.set_color(WHITE).scale(1.5),
                run_time=0.5,
            )

            # Map to grid
            mline_i = Line(pt_i, GRID_POSITIONS[bmu_i],
                           color=YELLOW, stroke_width=1.8)
            self.play(
                FadeOut(clines_i),
                Create(mline_i),
                grid_nodes[bmu_i].animate.scale(1.6).set_color(WHITE),
                run_time=0.6,
            )
            self.wait(0.3)

            # Restore + dim
            self.play(
                FadeOut(mline_i),
                w_dots[bmu_i].animate.set_color(CLUSTER_COLORS[node_labels[bmu_i]]).scale(1 / 1.5),
                grid_nodes[bmu_i].animate.scale(1 / 1.6).set_color(CLUSTER_COLORS[node_labels[bmu_i]]),
                nd.animate.set_opacity(0.5),
                run_time=0.5,
            )
            mapped_dots.add(nd)

        self.wait(0.5)

        # ----------------------------------------------------------------
        # PART 6 — FINAL VIEW
        # ----------------------------------------------------------------
        lbl_hd = Text("High-dimensional data  →  2D representation",
                      font_size=26, color=WHITE)
        lbl_hd.to_edge(DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(lbl_hd)
        self.play(FadeIn(lbl_hd), run_time=0.8)
        self.wait(2.5)

        # Slow ambient spin to admire the result
        self.begin_ambient_camera_rotation(rate=0.03, about="theta")
        self.wait(5.0)
        self.stop_ambient_camera_rotation()

        # ----------------------------------------------------------------
        # PART 7 — FADE NEW POINTS, KEEP GRID
        # ----------------------------------------------------------------
        self.play(
            FadeOut(mapped_dots),
            FadeOut(lbl_hd),
            run_time=1.0,
        )

        lbl_final = Text("Trained SOM — ready to classify",
                         font_size=30, color=TEAL_B)
        lbl_final.to_edge(DOWN, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl_final)
        self.play(FadeIn(lbl_final), run_time=0.8)
        self.wait(2.0)
        self.play(FadeOut(lbl_final), run_time=0.8)
        self.wait(0.5)
