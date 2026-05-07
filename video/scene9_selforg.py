"""
Scene 9: Self-Organization (Final Message)

Calm, minimal, reflective closing scene.
Reuses identical SOM training pipeline (seed 42, 300 iters) so weights
are exactly the same as scenes 6-8.

Parts:
  1 -- Clean frame: trained grid + data cloud (no highlights)
  2 -- Chaos → structure: ghost random initial weights → trained positions
  3 -- Complexity becomes geometry: highlight region ↔ cluster
  4 -- "Self-organization" centre title
  5 -- "Structure emerges from local interactions" (subtle, top)
  6 -- Full grid + data, slow ambient rotation
  7 -- "A way of making the invisible visible" closing line
  8 -- Slow fade to black

Usage:
    manim -ql --disable_caching scene9_selforg.py Scene9SelfOrg
    manim -qh --fps 60 scene9_selforg.py Scene9SelfOrg
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from manim import *

DEFAULT_FONT = "CMU Serif"
Text.set_default(font=DEFAULT_FONT)

# ====================================================================
# DATA  (identical seed / params to scenes 6-8)
# ====================================================================
_pts, _raw_labels = make_blobs(
    n_samples=300,
    centers=4,
    n_features=3,
    cluster_std=0.65,
    random_state=42,
)
_scaler = MinMaxScaler(feature_range=(-2.0, 2.0))
DATA_PTS = _scaler.fit_transform(_pts)
DATA_PTS[:, 0] += 2.2

# ====================================================================
# SOM TRAINING  (identical to scenes 6-8)
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
)

# --- capture initial random weights (for ghost) ---
weights = _rng_som.uniform(-2.0, 2.0, size=(N_NODES, 3))
_initial_weights_raw = weights.copy()   # before training, before x-shift

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

weights[:, 0] += 2.2
_initial_weights_raw[:, 0] += 2.2   # apply same shift to ghost positions

# ====================================================================
# NODE LABELS (majority-vote)
# ====================================================================
def find_bmu(x):
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
# LAYOUT CONSTANTS
# ====================================================================
GRID_SPACING   = 0.55
GRID_CX        = -3.6
GRID_CY        =  0.0
CLUSTER_COLORS = [RED_B, GREEN_B, BLUE_C, YELLOW]


def grid_pos(r, c):
    x = GRID_CX + (c - (GRID_COLS - 1) / 2.0) * GRID_SPACING
    y = GRID_CY + (r - (GRID_ROWS - 1) / 2.0) * GRID_SPACING
    return np.array([x, y, 0.0])


GRID_POSITIONS = [grid_pos(r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS)]


def w_pos(ni):
    return np.array([weights[ni, 0], weights[ni, 1], weights[ni, 2]])


def init_w_pos(ni):
    return np.array([
        _initial_weights_raw[ni, 0],
        _initial_weights_raw[ni, 1],
        _initial_weights_raw[ni, 2],
    ])


# ====================================================================
class Scene9SelfOrg(ThreeDScene):
    """Final scene — calm, reflective, emphasises emergence."""

    def construct(self):
        self.set_camera_orientation(phi=55 * DEGREES, theta=-60 * DEGREES, zoom=0.9)

        # ----------------------------------------------------------------
        # PART 1 — CLEAN FRAME
        # ----------------------------------------------------------------
        # Grid edges
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

        # Weight dots (cluster-coloured)
        w_dots = VGroup(*[
            Dot3D(point=w_pos(i), radius=0.07, color=CLUSTER_COLORS[node_labels[i]])
            for i in range(N_NODES)
        ])

        # Data cloud (cluster colour, small)
        data_dots = VGroup(*[
            Dot3D(
                point=np.array([DATA_PTS[k, 0], DATA_PTS[k, 1], DATA_PTS[k, 2]]),
                radius=0.04,
                color=CLUSTER_COLORS[int(_raw_labels[k])],
            )
            for k in range(len(DATA_PTS))
        ])

        # Faint axes
        axes = ThreeDAxes(
            x_range=[-2.5, 4.8, 1], y_range=[-2.5, 2.5, 1], z_range=[-2.5, 2.5, 1],
            x_length=7, y_length=5, z_length=5,
            tips=False,
            axis_config={"stroke_color": GREY_D, "stroke_width": 0.6,
                         "include_ticks": False},
        ).shift(RIGHT * 2.2)

        # Fade everything in gently together
        self.play(
            LaggedStart(
                FadeIn(h_lines), FadeIn(v_lines),
                FadeIn(grid_nodes), FadeIn(w_dots),
                FadeIn(data_dots), FadeIn(axes),
                lag_ratio=0.15,
            ),
            run_time=2.5,
        )

        lbl_after = Text("After training", font_size=30, color=GREY_A)
        lbl_after.to_edge(UL)
        self.add_fixed_in_frame_mobjects(lbl_after)
        self.play(Write(lbl_after), run_time=1.0)
        self.wait(2.5)   # let it breathe

        # ----------------------------------------------------------------
        # PART 2 — CHAOS → STRUCTURE
        # ----------------------------------------------------------------
        # Hide trained weights — disorder phase starts clean
        self.play(FadeOut(w_dots), run_time=0.8)

        # Build ghost dots at initial (random) weight positions
        ghost_dots = VGroup(*[
            Dot3D(point=init_w_pos(i), radius=0.06, color=GREY_C)
            for i in range(N_NODES)
        ])

        lbl_disorder = Text("From disorder...", font_size=36, color=GREY_B)
        lbl_disorder.move_to(ORIGIN)
        self.add_fixed_in_frame_mobjects(lbl_disorder)

        # Show ghosts + text together (trained weights already hidden)
        self.play(
            FadeIn(ghost_dots, run_time=1.5),
            Write(lbl_disorder, run_time=1.5),
        )
        self.wait(2.0)

        # Morph: ghost dots drift toward trained positions
        self.play(
            AnimationGroup(
                *[ghost_dots[i].animate.move_to(w_pos(i)) for i in range(N_NODES)],
                lag_ratio=0.0,
            ),
            FadeOut(lbl_disorder),
            run_time=2.5,
        )

        lbl_structure = Text("...to structure.", font_size=36, color=WHITE)
        lbl_structure.move_to(ORIGIN)
        self.add_fixed_in_frame_mobjects(lbl_structure)
        # Reveal trained weights as ghosts disappear — the "snap into place" moment
        self.play(
            Write(lbl_structure, run_time=1.0),
            FadeOut(ghost_dots, run_time=1.0),
            FadeIn(w_dots, run_time=1.0),
        )
        self.wait(2.0)
        self.play(FadeOut(lbl_structure), run_time=0.8)
        self.remove(lbl_after, lbl_structure)

        # ----------------------------------------------------------------
        # PART 3 — COMPLEXITY BECOMES GEOMETRY
        # ----------------------------------------------------------------
        # Pick top-left 2×2 patch (rows 2-3, cols 2-3 → indices 10,11,14,15)
        patch_idx = [10, 11, 14, 15]
        patch_color = TEAL_B

        # Pulse patch grid nodes
        self.play(
            AnimationGroup(
                *[grid_nodes[i].animate.set_color(patch_color).scale(1.5)
                  for i in patch_idx],
                lag_ratio=0.05,
            ),
            run_time=1.0,
        )
        # Pulse corresponding weight dots
        self.play(
            AnimationGroup(
                *[w_dots[i].animate.set_color(patch_color).scale(1.5)
                  for i in patch_idx],
                lag_ratio=0.05,
            ),
            run_time=0.8,
        )

        lbl_geom = Text("Complexity becomes geometry", font_size=30, color=GREY_A)
        lbl_geom.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(lbl_geom)
        self.play(Write(lbl_geom), run_time=1.0)
        self.wait(2.5)

        # Restore patch nodes
        self.play(
            AnimationGroup(
                *[grid_nodes[i].animate.set_color(CLUSTER_COLORS[node_labels[i]]).scale(1 / 1.5)
                  for i in patch_idx],
                *[w_dots[i].animate.set_color(CLUSTER_COLORS[node_labels[i]]).scale(1 / 1.5)
                  for i in patch_idx],
                lag_ratio=0.0,
            ),
            run_time=0.8,
        )
        self.play(FadeOut(lbl_geom), run_time=0.6)
        self.remove(lbl_geom)

        # ----------------------------------------------------------------
        # PART 4 — CORE IDEA: "Self-organization"
        # ----------------------------------------------------------------
        lbl_selforg = Text("Self-organization", font_size=54, color=WHITE)
        lbl_selforg.move_to(ORIGIN)
        self.add_fixed_in_frame_mobjects(lbl_selforg)

        # Write animation is iconic enough without the scale trick
        self.play(Write(lbl_selforg), run_time=2.0)
        self.wait(3.5)   # let it sit

        # ----------------------------------------------------------------
        # PART 5 — SUBTLE EXPLANATION
        # ----------------------------------------------------------------
        lbl_local = Text(
            "Structure emerges from local interactions",
            font_size=26,
            color=GREY_B,
        )
        lbl_local.to_edge(UP)
        self.add_fixed_in_frame_mobjects(lbl_local)
        self.play(Write(lbl_local), run_time=1.2)
        self.wait(2.5)

        # Fade out the centre title, keep subtle top label
        self.play(FadeOut(lbl_selforg), run_time=1.0)
        self.remove(lbl_selforg)

        # ----------------------------------------------------------------
        # PART 6 — FULL VIEW + SLOW ROTATION
        # ----------------------------------------------------------------
        lbl_dist = Text(
            "The grid reflects the data distribution",
            font_size=28,
            color=GREY_A,
        )
        lbl_dist.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(lbl_dist)
        self.play(Write(lbl_dist), run_time=1.0)

        # Full 360° rotation: 2π / rate ≈ one complete revolution
        self.begin_ambient_camera_rotation(rate=0.5, about="theta")
        self.wait(2 * np.pi / 0.5)   # ≈ 12.6 s — exactly one full revolution
        self.stop_ambient_camera_rotation()

        self.play(FadeOut(lbl_local), FadeOut(lbl_dist), run_time=0.8)
        self.remove(lbl_local, lbl_dist)

        # ----------------------------------------------------------------
        # PART 7 — CLOSING LINE
        # ----------------------------------------------------------------
        lbl_closing = Text(
            "A way of making the invisible visible",
            font_size=38,
            color=WHITE,
        )
        lbl_closing.move_to(ORIGIN)
        self.add_fixed_in_frame_mobjects(lbl_closing)
        self.play(Write(lbl_closing, run_time=1.8))
        self.wait(4.0)    # longer pause — this is the final thought

        # ----------------------------------------------------------------
        # PART 8 — FADE OUT
        # ----------------------------------------------------------------
        self.play(
            FadeOut(lbl_closing),
            FadeOut(grid_nodes),
            FadeOut(h_lines),
            FadeOut(v_lines),
            FadeOut(w_dots),
            FadeOut(data_dots),
            FadeOut(axes),
            run_time=3.0,
        )
        self.wait(1.0)
