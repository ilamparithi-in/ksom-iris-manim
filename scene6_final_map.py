"""
Scene 6: Final Map — Data-Driven Embedding

Steps:
  * Generate 300-sample 4-cluster blob dataset (sklearn, 3D, seed=42)
  * Train a 4x4 SOM for 300 iterations (pure NumPy, seed=42)
  * All weight positions come exclusively from trained weights
  * 4x4 grid on LEFT, trained weight vectors in 3D space on RIGHT

Parts:
  1 -- Show data cloud
  2 -- Show 4x4 grid
  3 -- Show trained weight vectors
  4 -- Draw grid->weight mapping lines, then fade
  5 -- "Learned representation"
  6 -- Topology: adjacent close / distant far
  7 -- Final frame: "Structure revealed"

Usage:
    manim -ql --disable_caching scene6_final_map.py Scene6FinalMap
    manim -qh --fps 60 scene6_final_map.py Scene6FinalMap
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from manim import *

DEFAULT_FONT = "CMU Serif"
Text.set_default(font=DEFAULT_FONT)

# ===========================================================================
# STEP 1 -- DATA GENERATION
# ===========================================================================
_pts, _labels = make_blobs(
    n_samples=300,
    centers=4,
    n_features=3,
    cluster_std=0.65,
    random_state=42,
)
_scaler = MinMaxScaler(feature_range=(-2.0, 2.0))
DATA_PTS = _scaler.fit_transform(_pts)   # (300, 3) in [-2, 2]^3

# Shift data to RIGHT side of screen
DATA_PTS[:, 0] += 2.2

# ===========================================================================
# STEP 2 -- SOM TRAINING (pure NumPy, no external SOM library)
# ===========================================================================
GRID_ROWS = 4
GRID_COLS = 4
N_NODES   = GRID_ROWS * GRID_COLS
SOM_ITERS = 300
ALPHA0    = 0.5   # initial learning rate
SIGMA0    = 2.0   # initial neighbourhood radius

_rng_som = np.random.default_rng(42)

# Grid coordinates (row, col) for neighbourhood distance
_grid_coords = np.array(
    [[r, c] for r in range(GRID_ROWS) for c in range(GRID_COLS)],
    dtype=float,
)  # (16, 2)

# Initialise weights randomly inside data range
weights = _rng_som.uniform(-2.0, 2.0, size=(N_NODES, 3))

for _t in range(SOM_ITERS):
    _x = DATA_PTS[_rng_som.integers(0, len(DATA_PTS))]  # random sample

    # BMU
    _diffs = weights - _x
    _bmu   = int(np.argmin(np.linalg.norm(_diffs, axis=1)))

    # Decaying parameters
    _alpha = ALPHA0 * np.exp(-_t / SOM_ITERS)
    _sigma = max(SIGMA0 * np.exp(-_t / (SOM_ITERS / np.log(SIGMA0 + 1e-9))), 0.3)

    # Neighbourhood function h_ci
    _gc_diff  = _grid_coords - _grid_coords[_bmu]        # (16, 2)
    _gc_dist2 = np.sum(_gc_diff ** 2, axis=1)            # (16,)
    _h = np.exp(-_gc_dist2 / (2.0 * _sigma ** 2))       # (16,)

    # Weight update
    weights += _alpha * _h[:, None] * (_x - weights)

# FINAL TRAINED WEIGHTS: shape (16, 3)
# Shift weights to same x-zone as data (right side)
weights[:, 0] += 2.2

# ===========================================================================
# STEP 3 -- LAYOUT CONSTANTS (LEFT side grid)
# ===========================================================================
GRID_SPACING = 0.55
GRID_CX      = -3.6
GRID_CY      =  0.0

GRID_COLOR   = BLUE_E
WEIGHT_COLOR = TEAL_B
DATA_COLOR   = GREY_B


def grid_pos(r: int, c: int) -> np.ndarray:
    x = GRID_CX + (c - (GRID_COLS - 1) / 2.0) * GRID_SPACING
    y = GRID_CY + (r - (GRID_ROWS - 1) / 2.0) * GRID_SPACING
    return np.array([x, y, 0.0])


GRID_POSITIONS = [grid_pos(r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS)]


def grid_idx(r: int, c: int) -> int:
    return r * GRID_COLS + c


def w_pos(node_idx: int) -> np.ndarray:
    return np.array([weights[node_idx, 0],
                     weights[node_idx, 1],
                     weights[node_idx, 2]])


# ===========================================================================
class Scene6FinalMap(ThreeDScene):
    """Final map: 4x4 grid embedded in data space using real trained weights."""

    def construct(self):
        self.set_camera_orientation(
            phi=55 * DEGREES, theta=-60 * DEGREES, zoom=0.9
        )

        # ====================================================================
        # PART 1 -- SHOW DATA CLOUD
        # ====================================================================
        data_dots = VGroup(
            *[
                Dot3D(
                    point=np.array([DATA_PTS[i, 0], DATA_PTS[i, 1], DATA_PTS[i, 2]]),
                    radius=0.035,
                    color=DATA_COLOR,
                )
                for i in range(len(DATA_PTS))
            ]
        )

        # ── Faint reference axes — data space (RIGHT, center x=2.2) ──────────
        data_axes = ThreeDAxes(
            x_range=[-2.2, 2.2, 1], y_range=[-2.2, 2.2, 1], z_range=[-2.2, 2.2, 1],
            x_length=4.4, y_length=4.4, z_length=4.4,
            tips=False,
            axis_config={"stroke_color": GREY_D, "stroke_width": 0.6,
                         "include_ticks": False},
        ).shift(RIGHT * 2.2)

        lbl_data = Text("Training data", font_size=28, color=WHITE)
        lbl_data.to_edge(UR, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl_data)
        self.play(
            LaggedStart(*[FadeIn(d) for d in data_dots], lag_ratio=0.006),
            FadeIn(lbl_data),
            FadeIn(data_axes),
            run_time=2.5,
        )
        self.wait(1.0)

        # ====================================================================
        # PART 2 -- SHOW 4x4 GRID (LEFT)
        # ====================================================================
        h_lines = VGroup(
            *[
                Line(
                    GRID_POSITIONS[r * GRID_COLS + c],
                    GRID_POSITIONS[r * GRID_COLS + c + 1],
                    color=GRID_COLOR, stroke_width=1.2,
                )
                for r in range(GRID_ROWS)
                for c in range(GRID_COLS - 1)
            ]
        )
        v_lines = VGroup(
            *[
                Line(
                    GRID_POSITIONS[r * GRID_COLS + c],
                    GRID_POSITIONS[(r + 1) * GRID_COLS + c],
                    color=GRID_COLOR, stroke_width=1.2,
                )
                for r in range(GRID_ROWS - 1)
                for c in range(GRID_COLS)
            ]
        )
        grid_nodes = VGroup(
            *[
                Dot3D(point=GRID_POSITIONS[i], radius=0.07, color=GRID_COLOR)
                for i in range(N_NODES)
            ]
        )

        lbl_grid = Text("4x4 grid", font_size=28, color=WHITE)
        lbl_grid.to_edge(UL, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl_grid)
        self.play(
            Create(h_lines), Create(v_lines),
            FadeIn(lbl_grid),
            run_time=1.2,
        )
        self.play(
            LaggedStart(*[FadeIn(d) for d in grid_nodes], lag_ratio=0.04),
            run_time=0.8,
        )
        self.wait(1.0)

        # ====================================================================
        # PART 3 -- SHOW TRAINED WEIGHT VECTORS (RIGHT)
        # ====================================================================
        w_dots = VGroup(
            *[
                Dot3D(point=w_pos(i), radius=0.10, color=WEIGHT_COLOR)
                for i in range(N_NODES)
            ]
        )

        lbl_weights = Text("Trained weights", font_size=28, color=TEAL_B)
        lbl_weights.to_edge(DOWN, buff=0.45)
        self.add_fixed_in_frame_mobjects(lbl_weights)
        self.play(FadeOut(lbl_data), run_time=0.5)
        self.play(
            LaggedStart(*[FadeIn(d) for d in w_dots], lag_ratio=0.04),
            FadeIn(lbl_weights),
            run_time=2.0,
        )
        self.wait(1.5)

        # ====================================================================
        # PART 4 -- MAPPING LINES: grid node -> trained weight (then fade)
        # ====================================================================
        conn_lines = VGroup(
            *[
                Line(
                    GRID_POSITIONS[i],
                    w_pos(i),
                    color=GREY_C,
                    stroke_width=0.8,
                )
                for i in range(N_NODES)
            ]
        )

        self.play(FadeOut(lbl_grid), FadeOut(lbl_weights), run_time=0.5)

        lbl_mapping = Text("Each node maps to a weight vector", font_size=26, color=WHITE)
        lbl_mapping.to_edge(UP, buff=0.3)
        self.add_fixed_in_frame_mobjects(lbl_mapping)
        self.play(
            LaggedStart(*[Create(l) for l in conn_lines], lag_ratio=0.05),
            FadeIn(lbl_mapping),
            run_time=2.5,
        )
        self.wait(1.5)
        self.play(FadeOut(conn_lines), FadeOut(lbl_mapping), run_time=1.0)
        self.wait(0.5)

        # ====================================================================
        # PART 5 -- "LEARNED REPRESENTATION"
        # ====================================================================
        lbl_learned = Text("Learned representation", font_size=32, color=WHITE)
        lbl_learned.to_edge(UP, buff=0.3)
        self.add_fixed_in_frame_mobjects(lbl_learned)
        self.play(FadeIn(lbl_learned), run_time=0.8)
        self.wait(2.0)

        # ====================================================================
        # PART 6 -- TOPOLOGY PRESERVATION
        # ====================================================================

        # --- Adjacent pair ---
        adj_a = grid_idx(1, 1)
        adj_b = grid_idx(1, 2)

        self.play(FadeOut(lbl_learned), run_time=0.4)

        lbl_nearby = Text("Nearby nodes -> similar data", font_size=28, color=GREEN_B)
        lbl_nearby.to_edge(UR, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl_nearby)

        self.play(
            grid_nodes[adj_a].animate.set_color(YELLOW).scale(1.6),
            grid_nodes[adj_b].animate.set_color(YELLOW).scale(1.6),
            w_dots[adj_a].animate.set_color(YELLOW).scale(1.5),
            w_dots[adj_b].animate.set_color(YELLOW).scale(1.5),
            FadeIn(lbl_nearby),
            run_time=1.0,
        )
        adj_conn = Line(w_pos(adj_a), w_pos(adj_b), color=YELLOW, stroke_width=2.0)
        self.play(Create(adj_conn), run_time=0.8)
        self.wait(1.8)

        self.play(
            grid_nodes[adj_a].animate.set_color(GRID_COLOR).scale(1 / 1.6),
            grid_nodes[adj_b].animate.set_color(GRID_COLOR).scale(1 / 1.6),
            w_dots[adj_a].animate.set_color(WEIGHT_COLOR).scale(1 / 1.5),
            w_dots[adj_b].animate.set_color(WEIGHT_COLOR).scale(1 / 1.5),
            FadeOut(adj_conn),
            FadeOut(lbl_nearby),
            run_time=0.8,
        )
        self.wait(0.3)

        # --- Distant pair ---
        dist_a = grid_idx(0, 0)
        dist_b = grid_idx(3, 3)

        lbl_distant = Text("Distant nodes -> different data", font_size=28, color=ORANGE)
        lbl_distant.to_edge(UR, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl_distant)

        self.play(
            grid_nodes[dist_a].animate.set_color(ORANGE).scale(1.6),
            grid_nodes[dist_b].animate.set_color(ORANGE).scale(1.6),
            w_dots[dist_a].animate.set_color(ORANGE).scale(1.5),
            w_dots[dist_b].animate.set_color(ORANGE).scale(1.5),
            FadeIn(lbl_distant),
            run_time=1.0,
        )
        dist_conn = Line(w_pos(dist_a), w_pos(dist_b), color=ORANGE, stroke_width=2.0)
        self.play(Create(dist_conn), run_time=0.8)
        self.wait(1.8)

        self.play(
            grid_nodes[dist_a].animate.set_color(GRID_COLOR).scale(1 / 1.6),
            grid_nodes[dist_b].animate.set_color(GRID_COLOR).scale(1 / 1.6),
            w_dots[dist_a].animate.set_color(WEIGHT_COLOR).scale(1 / 1.5),
            w_dots[dist_b].animate.set_color(WEIGHT_COLOR).scale(1 / 1.5),
            FadeOut(dist_conn),
            FadeOut(lbl_distant),
            run_time=0.8,
        )
        self.wait(0.5)

        # ====================================================================
        # PART 7 -- FINAL FRAME: "Structure revealed"
        # ====================================================================
        self.begin_ambient_camera_rotation(rate=0.04, about="theta")

        lbl_dist = Text("The grid reflects the data distribution",
                        font_size=26, color=WHITE)
        lbl_dist.to_edge(DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(lbl_dist)
        self.play(FadeIn(lbl_dist), run_time=0.8)
        self.wait(4.0)

        self.stop_ambient_camera_rotation()

        self.play(FadeOut(lbl_dist), run_time=0.6)

        lbl_final = Text("Structure revealed", font_size=40, color=TEAL_B)
        lbl_final.move_to(ORIGIN)
        self.add_fixed_in_frame_mobjects(lbl_final)
        self.play(FadeIn(lbl_final), run_time=1.0)
        self.wait(3.0)
        self.play(FadeOut(lbl_final), run_time=0.8)
        self.wait(0.5)
