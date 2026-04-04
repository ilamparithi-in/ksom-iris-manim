"""
Scene 7: Clusters and Boundaries (U-Matrix)

Precomputes:
  * dataset  — sklearn make_blobs, 300 pts, 4 clusters, 3D, seed=42
  * SOM      — 4×4 grid, 300 iterations, pure NumPy, seed=42
               (identical hyperparameters to scene6_final_map.py)
  * node_labels — majority-vote cluster label per node
  * U_matrix    — per-node average distance to immediate grid neighbours

Parts:
  1  -- Trained grid + data cloud → "Interpreting the map"
  2  -- Color data by cluster
  3  -- Color grid/weight nodes by node_label → "Clusters appear as regions"
  4  -- Local distance demo: same-cluster pair (short) vs cross-boundary (long)
  5  -- "What if we measure this everywhere?" transition
  6  -- U-matrix buildup: gradually recolor grid nodes TEAL→RED by U value
  7  -- "High values → boundaries"
  8  -- Dim all, spotlight highest-U node + its assigned data points
  9  -- "Boundaries emerge from distances"
  10 -- "Structure becomes visible"

Usage:
    manim -ql --disable_caching scene7_umatrix.py Scene7UMatrix
    manim -qh --fps 60 scene7_umatrix.py Scene7UMatrix
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from manim import *

# ====================================================================
# STEP 1 — DATA  (same seed / params as scene6)
# ====================================================================
_pts, _raw_labels = make_blobs(
    n_samples=300,
    centers=4,
    n_features=3,
    cluster_std=0.65,
    random_state=42,
)
_scaler = MinMaxScaler(feature_range=(-2.0, 2.0))
DATA_PTS = _scaler.fit_transform(_pts)          # (300, 3) in [-2, 2]^3
DATA_PTS[:, 0] += 2.2                           # shift cloud to RIGHT

# ====================================================================
# STEP 2 — SOM TRAINING  (identical to scene6_final_map.py)
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

weights[:, 0] += 2.2   # shift weights to RIGHT (same as data)

# ====================================================================
# STEP 3 — NODE LABELS (majority-vote cluster per grid node)
# ====================================================================
def _bmu_of(x: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(weights - x, axis=1)))

_point_bmu = np.array([_bmu_of(DATA_PTS[i]) for i in range(len(DATA_PTS))])

node_labels = np.zeros(N_NODES, dtype=int)
for _ni in range(N_NODES):
    _assigned = _raw_labels[_point_bmu == _ni]
    if len(_assigned) > 0:
        node_labels[_ni] = int(np.bincount(_assigned).argmax())

# Fill empty nodes (no assigned data points) via nearest-neighbour cascade
for _ in range(4):
    for _ni in range(N_NODES):
        if (_point_bmu == _ni).sum() == 0:
            _d = np.linalg.norm(weights - weights[_ni], axis=1)
            _d[_ni] = 999.0
            node_labels[_ni] = node_labels[int(np.argmin(_d))]

# ====================================================================
# STEP 4 — U-MATRIX  (avg weight-distance to immediate grid neighbours)
# ====================================================================
U_vals = np.zeros(N_NODES)
for _r in range(GRID_ROWS):
    for _c in range(GRID_COLS):
        _ni   = _r * GRID_COLS + _c
        _nbrs = []
        for _dr, _dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            _nr, _nc = _r + _dr, _c + _dc
            if 0 <= _nr < GRID_ROWS and 0 <= _nc < GRID_COLS:
                _nj = _nr * GRID_COLS + _nc
                _nbrs.append(np.linalg.norm(weights[_ni] - weights[_nj]))
        U_vals[_ni] = float(np.mean(_nbrs)) if _nbrs else 0.0

_u_min, _u_max = U_vals.min(), U_vals.max()
U_norm = (U_vals - _u_min) / (_u_max - _u_min + 1e-9)  # [0, 1]

# ====================================================================
# STEP 5 — PICK INTERESTING PAIRS FOR PART 4
# ====================================================================
same_pair: tuple[int, int] | None = None   # adjacent, same cluster
diff_pair: tuple[int, int] | None = None   # adjacent, different cluster

for _r in range(GRID_ROWS):
    for _c in range(GRID_COLS - 1):
        _ni = _r * GRID_COLS + _c
        _nj = _r * GRID_COLS + _c + 1
        if same_pair is None and node_labels[_ni] == node_labels[_nj]:
            same_pair = (_ni, _nj)
        if diff_pair is None and node_labels[_ni] != node_labels[_nj]:
            diff_pair = (_ni, _nj)
    if same_pair and diff_pair:
        break

if same_pair is None:
    same_pair = (5, 6)
if diff_pair is None:
    # fall back to maximum-U adjacent pair
    best_u = -1.0
    for _r in range(GRID_ROWS):
        for _c in range(GRID_COLS - 1):
            _ni = _r * GRID_COLS + _c
            _nj = _r * GRID_COLS + _c + 1
            u_avg = (U_norm[_ni] + U_norm[_nj]) / 2
            if u_avg > best_u:
                best_u = u_avg
                diff_pair = (_ni, _nj)

# Node with highest U value — clearest boundary
boundary_node = int(np.argmax(U_norm))

# ====================================================================
# LAYOUT CONSTANTS
# ====================================================================
GRID_SPACING = 0.55
GRID_CX      = -3.6
GRID_CY      =  0.0

CLUSTER_COLORS = [RED_B, GREEN_B, BLUE_C, YELLOW]  # 4 clusters


def grid_pos(r: int, c: int) -> np.ndarray:
    x = GRID_CX + (c - (GRID_COLS - 1) / 2.0) * GRID_SPACING
    y = GRID_CY + (r - (GRID_ROWS - 1) / 2.0) * GRID_SPACING
    return np.array([x, y, 0.0])


GRID_POSITIONS = [grid_pos(r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS)]


def w_pos(ni: int) -> np.ndarray:
    return np.array([weights[ni, 0], weights[ni, 1], weights[ni, 2]])


def u_color(ni: int):
    """TEAL_B = low distance (within cluster), RED_B = high (boundary)."""
    return interpolate_color(TEAL_B, RED_B, float(U_norm[ni]))


# ====================================================================
class Scene7UMatrix(ThreeDScene):
    """Clusters and Boundaries via the U-Matrix."""

    def construct(self):
        self.set_camera_orientation(phi=55 * DEGREES, theta=-60 * DEGREES, zoom=0.9)

        # ----------------------------------------------------------------
        # PART 1 — SETUP  (scene-6 end state: trained grid + data cloud)
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
            Dot3D(point=GRID_POSITIONS[i], radius=0.08, color=BLUE_E)
            for i in range(N_NODES)
        ])
        w_dots = VGroup(*[
            Dot3D(point=w_pos(i), radius=0.10, color=TEAL_B)
            for i in range(N_NODES)
        ])
        data_dots = VGroup(*[
            Dot3D(
                point=np.array([DATA_PTS[i, 0], DATA_PTS[i, 1], DATA_PTS[i, 2]]),
                radius=0.035, color=GREY_B,
            )
            for i in range(len(DATA_PTS))
        ])

        # ── Faint reference axes — data space (RIGHT, center x=2.2) ──────────
        data_axes = ThreeDAxes(
            x_range=[-2.2, 2.2, 1], y_range=[-2.2, 2.2, 1], z_range=[-2.2, 2.2, 1],
            x_length=4.4, y_length=4.4, z_length=4.4,
            tips=False,
            axis_config={"stroke_color": GREY_D, "stroke_width": 0.6,
                         "include_ticks": False},
        ).shift(RIGHT * 2.2)

        lbl1 = Text("Interpreting the map", font_size=30, color=WHITE)
        lbl1.to_edge(UL, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl1)
        self.play(
            Create(h_lines), Create(v_lines),
            LaggedStart(*[FadeIn(d) for d in grid_nodes], lag_ratio=0.04),
            LaggedStart(*[FadeIn(d) for d in w_dots],    lag_ratio=0.04),
            LaggedStart(*[FadeIn(d) for d in data_dots], lag_ratio=0.006),
            FadeIn(lbl1),
            FadeIn(data_axes),
            run_time=3.0,
        )
        self.wait(1.5)

        # ----------------------------------------------------------------
        # PART 2 — COLOR DATA BY CLUSTER
        # ----------------------------------------------------------------
        self.play(
            AnimationGroup(
                *[data_dots[i].animate.set_color(CLUSTER_COLORS[int(_raw_labels[i])])
                  for i in range(len(DATA_PTS))],
                lag_ratio=0.004,
            ),
            run_time=2.5,
        )
        self.wait(1.0)

        # ----------------------------------------------------------------
        # PART 3 — COLOR GRID NODES + WEIGHT DOTS BY NODE LABEL
        # ----------------------------------------------------------------
        self.play(
            AnimationGroup(
                *[grid_nodes[i].animate.set_color(CLUSTER_COLORS[int(node_labels[i])])
                  for i in range(N_NODES)],
                lag_ratio=0.05,
            ),
            AnimationGroup(
                *[w_dots[i].animate.set_color(CLUSTER_COLORS[int(node_labels[i])])
                  for i in range(N_NODES)],
                lag_ratio=0.05,
            ),
            run_time=1.8,
        )
        self.play(FadeOut(lbl1), run_time=0.4)

        lbl3 = Text("Clusters appear as regions", font_size=28, color=WHITE)
        lbl3.to_edge(UL, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl3)
        self.play(FadeIn(lbl3), run_time=0.7)
        self.wait(2.0)

        # ----------------------------------------------------------------
        # PART 4 — LOCAL DISTANCE DEMO
        # ----------------------------------------------------------------
        self.play(FadeOut(lbl3), run_time=0.4)

        # --- Same-cluster pair → SHORT distance ---
        si, sj = same_pair
        w_line_same = Line(w_pos(si), w_pos(sj), color=GREEN_B, stroke_width=2.5)
        mid_same = (w_pos(si) + w_pos(sj)) / 2.0 + np.array([0.0, 0.28, 0.0])
        d_lbl_same = Text("short\ndistance", font_size=18, color=GREEN_B)
        d_lbl_same.move_to(mid_same)

        lbl4a = Text("Same cluster: weight vectors are close", font_size=26, color=GREEN_B)
        lbl4a.to_edge(UR, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl4a)
        self.play(
            grid_nodes[si].animate.set_color(WHITE).scale(1.5),
            grid_nodes[sj].animate.set_color(WHITE).scale(1.5),
            w_dots[si].animate.set_color(WHITE).scale(1.5),
            w_dots[sj].animate.set_color(WHITE).scale(1.5),
            FadeIn(lbl4a),
            run_time=0.8,
        )
        self.play(Create(w_line_same), run_time=0.7)
        self.play(FadeIn(d_lbl_same), run_time=0.5)
        self.wait(1.8)

        self.play(
            FadeOut(w_line_same), FadeOut(d_lbl_same), FadeOut(lbl4a),
            grid_nodes[si].animate.set_color(CLUSTER_COLORS[int(node_labels[si])]).scale(1 / 1.5),
            grid_nodes[sj].animate.set_color(CLUSTER_COLORS[int(node_labels[sj])]).scale(1 / 1.5),
            w_dots[si].animate.set_color(CLUSTER_COLORS[int(node_labels[si])]).scale(1 / 1.5),
            w_dots[sj].animate.set_color(CLUSTER_COLORS[int(node_labels[sj])]).scale(1 / 1.5),
            run_time=0.7,
        )
        self.wait(0.3)

        # --- Cross-boundary pair → LONG distance ---
        di, dj = diff_pair
        w_line_diff = Line(w_pos(di), w_pos(dj), color=RED_B, stroke_width=2.5)
        mid_diff = (w_pos(di) + w_pos(dj)) / 2.0 + np.array([0.0, 0.28, 0.0])
        d_lbl_diff = Text("long\ndistance", font_size=18, color=RED_B)
        d_lbl_diff.move_to(mid_diff)

        lbl4b = Text("Some neighbors are very different", font_size=26, color=RED_B)
        lbl4b.to_edge(UR, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl4b)
        self.play(
            grid_nodes[di].animate.set_color(WHITE).scale(1.5),
            grid_nodes[dj].animate.set_color(WHITE).scale(1.5),
            w_dots[di].animate.set_color(WHITE).scale(1.5),
            w_dots[dj].animate.set_color(WHITE).scale(1.5),
            FadeIn(lbl4b),
            run_time=0.8,
        )
        self.play(Create(w_line_diff), run_time=0.7)
        self.play(FadeIn(d_lbl_diff), run_time=0.5)
        self.wait(1.8)

        self.play(
            FadeOut(w_line_diff), FadeOut(d_lbl_diff), FadeOut(lbl4b),
            grid_nodes[di].animate.set_color(CLUSTER_COLORS[int(node_labels[di])]).scale(1 / 1.5),
            grid_nodes[dj].animate.set_color(CLUSTER_COLORS[int(node_labels[dj])]).scale(1 / 1.5),
            w_dots[di].animate.set_color(CLUSTER_COLORS[int(node_labels[di])]).scale(1 / 1.5),
            w_dots[dj].animate.set_color(CLUSTER_COLORS[int(node_labels[dj])]).scale(1 / 1.5),
            run_time=0.7,
        )
        self.wait(0.5)

        # ----------------------------------------------------------------
        # PART 5 — TRANSITION TEXT
        # ----------------------------------------------------------------
        lbl5 = Text("What if we measure this everywhere?", font_size=28, color=WHITE)
        lbl5.to_edge(UP, buff=0.3)
        self.add_fixed_in_frame_mobjects(lbl5)
        self.play(FadeIn(lbl5), run_time=0.8)
        self.wait(2.0)
        self.play(FadeOut(lbl5), run_time=0.5)

        # ----------------------------------------------------------------
        # PART 6 — U-MATRIX BUILDUP  (low U → TEAL, high U → RED)
        # ----------------------------------------------------------------
        # Sort nodes low→high so the colour shift sweeps from calm to boundary
        sorted_by_u = sorted(range(N_NODES), key=lambda i: float(U_norm[i]))

        lbl6 = Text("U-matrix: avg. distance to grid neighbours", font_size=26, color=WHITE)
        lbl6.to_edge(UL, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl6)
        self.play(FadeIn(lbl6), run_time=0.6)

        self.play(
            LaggedStart(
                *[grid_nodes[i].animate.set_color(u_color(i)) for i in sorted_by_u],
                lag_ratio=0.12,
            ),
            run_time=3.5,
        )
        self.wait(1.5)

        # ----------------------------------------------------------------
        # PART 7 — INTERPRETATION
        # ----------------------------------------------------------------
        lbl7 = Text("High values → boundaries", font_size=28, color=RED_B)
        lbl7.to_edge(UR, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl7)
        self.play(FadeIn(lbl7), run_time=0.7)
        self.wait(2.0)
        self.play(FadeOut(lbl6), FadeOut(lbl7), run_time=0.5)

        # ----------------------------------------------------------------
        # PART 8 — SPOTLIGHT HIGHEST-U NODE + ITS DATA POINTS
        # ----------------------------------------------------------------
        bn = boundary_node   # index of node with max U value

        # Dim everything
        self.play(
            AnimationGroup(*[
                grid_nodes[i].animate.set_opacity(0.12)
                for i in range(N_NODES) if i != bn
            ], lag_ratio=0.0),
            AnimationGroup(*[
                w_dots[i].animate.set_opacity(0.12)
                for i in range(N_NODES) if i != bn
            ], lag_ratio=0.0),
            data_dots.animate.set_opacity(0.12),
            run_time=0.8,
        )
        # Highlight boundary node
        self.play(
            grid_nodes[bn].animate.set_color(WHITE).scale(1.8),
            w_dots[bn].animate.set_color(WHITE).scale(1.5),
            run_time=0.7,
        )

        # Highlight data points assigned to the boundary node in data space
        bn_pts = [i for i in range(len(DATA_PTS)) if _point_bmu[i] == bn]
        if bn_pts:
            self.play(
                AnimationGroup(
                    *[data_dots[i].animate.set_opacity(1.0).set_color(WHITE)
                      for i in bn_pts],
                    lag_ratio=0.06,
                ),
                run_time=1.0,
            )
        self.wait(2.5)

        # Restore
        self.play(
            AnimationGroup(*[
                (grid_nodes[i].animate.set_opacity(1.0).set_color(u_color(i)).scale(1 / 1.8)
                 if i == bn
                 else grid_nodes[i].animate.set_opacity(1.0).set_color(u_color(i)))
                for i in range(N_NODES)
            ], lag_ratio=0.0),
            AnimationGroup(*[
                (w_dots[i].animate.set_opacity(1.0).set_color(CLUSTER_COLORS[int(node_labels[i])]).scale(1 / 1.5)
                 if i == bn
                 else w_dots[i].animate.set_opacity(1.0).set_color(CLUSTER_COLORS[int(node_labels[i])]))
                for i in range(N_NODES)
            ], lag_ratio=0.0),
            AnimationGroup(*[
                data_dots[i].animate.set_opacity(0.85).set_color(CLUSTER_COLORS[int(_raw_labels[i])])
                for i in range(len(DATA_PTS))
            ], lag_ratio=0.003),
            run_time=1.2,
        )

        # ----------------------------------------------------------------
        # PART 9 — CLEAN VIEW
        # ----------------------------------------------------------------
        lbl9 = Text("Boundaries emerge from distances", font_size=28, color=WHITE)
        lbl9.to_edge(DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(lbl9)
        self.play(FadeIn(lbl9), run_time=0.8)
        self.wait(3.0)
        self.play(FadeOut(lbl9), run_time=0.6)

        # ----------------------------------------------------------------
        # PART 10 — FINAL MESSAGE
        # ----------------------------------------------------------------
        lbl10 = Text("Structure becomes visible", font_size=38, color=TEAL_B)
        lbl10.move_to(ORIGIN)
        self.add_fixed_in_frame_mobjects(lbl10)
        self.play(FadeIn(lbl10), run_time=1.0)
        self.wait(3.5)
        self.play(FadeOut(lbl10), run_time=0.8)
        self.wait(0.5)

        # ================================================================
        # EXTENSION — PARTS 7A–7G
        # State: grid_nodes coloured by U-matrix (TEAL→RED),
        #        w_dots coloured by cluster, data_dots by cluster @0.85,
        #        all screen text cleared.
        # ================================================================

        # Pre-compute ordered boundary / interior node lists once
        high_u_nodes = sorted(range(N_NODES), key=lambda i: -float(U_norm[i]))[:4]

        # ----------------------------------------------------------------
        # 7A — NOTICE THE BOUNDARIES
        # ----------------------------------------------------------------
        lbl7a = Text("But what defines these boundaries?", font_size=28, color=WHITE)
        lbl7a.to_edge(UR, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl7a)
        self.play(FadeIn(lbl7a), run_time=0.7)
        # Pulse RED (high-U) boundary nodes
        self.play(
            LaggedStart(*[grid_nodes[i].animate.scale(1.6) for i in high_u_nodes],
                        lag_ratio=0.15),
            run_time=0.8,
        )
        self.play(
            LaggedStart(*[grid_nodes[i].animate.scale(1 / 1.6) for i in high_u_nodes],
                        lag_ratio=0.15),
            run_time=0.6,
        )
        self.wait(1.5)
        self.play(FadeOut(lbl7a), run_time=0.4)

        # ----------------------------------------------------------------
        # 7B — LOCAL DISTANCE DEMO
        # ----------------------------------------------------------------
        si, sj = same_pair
        di, dj = diff_pair

        # --- Same cluster → small distance ---
        w_line_s = Line(w_pos(si), w_pos(sj), color=GREEN_B, stroke_width=2.5)
        mid_s    = (w_pos(si) + w_pos(sj)) / 2.0 + np.array([0.0, 0.3, 0.0])
        lbl_sd   = Text("small distance", font_size=18, color=GREEN_B)
        lbl_sd.move_to(mid_s)
        lbl7b1 = Text("Adjacent nodes inside same cluster", font_size=26, color=GREEN_B)
        lbl7b1.to_edge(UR, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl7b1)
        self.play(
            grid_nodes[si].animate.set_color(WHITE).scale(1.5),
            grid_nodes[sj].animate.set_color(WHITE).scale(1.5),
            w_dots[si].animate.set_color(WHITE).scale(1.5),
            w_dots[sj].animate.set_color(WHITE).scale(1.5),
            FadeIn(lbl7b1),
            run_time=0.8,
        )
        self.play(Create(w_line_s), run_time=0.6)
        self.play(FadeIn(lbl_sd), run_time=0.5)
        self.wait(1.5)
        self.play(
            FadeOut(w_line_s), FadeOut(lbl_sd), FadeOut(lbl7b1),
            grid_nodes[si].animate.set_color(u_color(si)).scale(1 / 1.5),
            grid_nodes[sj].animate.set_color(u_color(sj)).scale(1 / 1.5),
            w_dots[si].animate.set_color(CLUSTER_COLORS[int(node_labels[si])]).scale(1 / 1.5),
            w_dots[sj].animate.set_color(CLUSTER_COLORS[int(node_labels[sj])]).scale(1 / 1.5),
            run_time=0.6,
        )

        # --- Cross-boundary → large distance ---
        w_line_d = Line(w_pos(di), w_pos(dj), color=RED_B, stroke_width=2.5)
        mid_d    = (w_pos(di) + w_pos(dj)) / 2.0 + np.array([0.0, 0.3, 0.0])
        lbl_ld   = Text("large distance", font_size=18, color=RED_B)
        lbl_ld.move_to(mid_d)
        lbl7b2 = Text("Neighbors very different", font_size=26, color=RED_B)
        lbl7b2.to_edge(UR, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl7b2)
        self.play(
            grid_nodes[di].animate.set_color(WHITE).scale(1.5),
            grid_nodes[dj].animate.set_color(WHITE).scale(1.5),
            w_dots[di].animate.set_color(WHITE).scale(1.5),
            w_dots[dj].animate.set_color(WHITE).scale(1.5),
            FadeIn(lbl7b2),
            run_time=0.8,
        )
        self.play(Create(w_line_d), run_time=0.6)
        self.play(FadeIn(lbl_ld), run_time=0.5)
        self.wait(1.5)
        self.play(
            FadeOut(w_line_d), FadeOut(lbl_ld), FadeOut(lbl7b2),
            grid_nodes[di].animate.set_color(u_color(di)).scale(1 / 1.5),
            grid_nodes[dj].animate.set_color(u_color(dj)).scale(1 / 1.5),
            w_dots[di].animate.set_color(CLUSTER_COLORS[int(node_labels[di])]).scale(1 / 1.5),
            w_dots[dj].animate.set_color(CLUSTER_COLORS[int(node_labels[dj])]).scale(1 / 1.5),
            run_time=0.6,
        )
        self.wait(0.3)

        # ----------------------------------------------------------------
        # 7C — GENERALIZATION
        # ----------------------------------------------------------------
        lbl7c = Text("What if we measure this everywhere?", font_size=28, color=WHITE)
        lbl7c.to_edge(UP, buff=0.3)
        self.add_fixed_in_frame_mobjects(lbl7c)
        self.play(FadeIn(lbl7c), run_time=0.7)
        self.wait(1.8)
        self.play(FadeOut(lbl7c), run_time=0.5)

        # ----------------------------------------------------------------
        # 7D — CONFIRM U-MATRIX IS ALREADY VISIBLE
        # ----------------------------------------------------------------
        lbl7d = Text("U-matrix: colour = avg. neighbour distance", font_size=26, color=TEAL_B)
        lbl7d.to_edge(UL, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl7d)
        self.play(FadeIn(lbl7d), run_time=0.7)
        self.wait(2.0)

        # ----------------------------------------------------------------
        # 7E — INTERPRETATION
        # ----------------------------------------------------------------
        lbl7e = Text("High distance → boundary", font_size=28, color=RED_B)
        lbl7e.to_edge(UR, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl7e)
        self.play(FadeIn(lbl7e), run_time=0.7)
        # Pulse boundary nodes once more
        self.play(
            LaggedStart(*[grid_nodes[i].animate.scale(1.5) for i in high_u_nodes],
                        lag_ratio=0.1),
            run_time=0.7,
        )
        self.play(
            LaggedStart(*[grid_nodes[i].animate.scale(1 / 1.5) for i in high_u_nodes],
                        lag_ratio=0.1),
            run_time=0.5,
        )
        self.wait(1.5)
        self.play(FadeOut(lbl7d), FadeOut(lbl7e), run_time=0.4)

        # ----------------------------------------------------------------
        # 7F — LINK BACK TO DATA SEPARATION
        # ----------------------------------------------------------------
        bn = boundary_node
        # Find up to 2 distinct cluster labels in bn's grid neighbours
        _br, _bc = divmod(bn, GRID_COLS)
        _flanking: set[int] = set()
        for _dr, _dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            _nr2, _nc2 = _br + _dr, _bc + _dc
            if 0 <= _nr2 < GRID_ROWS and 0 <= _nc2 < GRID_COLS:
                _flanking.add(int(node_labels[_nr2 * GRID_COLS + _nc2]))
        flanking_clusters = list(_flanking)[:2]  # highlight ≤2 clusters

        # Dim everything
        self.play(
            data_dots.animate.set_opacity(0.10),
            AnimationGroup(*[
                grid_nodes[i].animate.set_opacity(0.12)
                for i in range(N_NODES) if i != bn
            ], lag_ratio=0.0),
            run_time=0.8,
        )
        # Spotlight: boundary grid node + flanking data clusters
        highlight_pts = [
            i for i in range(len(DATA_PTS))
            if int(_raw_labels[i]) in flanking_clusters
        ]
        self.play(
            grid_nodes[bn].animate.set_color(WHITE).scale(1.6),
            AnimationGroup(
                *[data_dots[i].animate.set_opacity(0.9)
                  for i in highlight_pts],
                lag_ratio=0.003,
            ),
            run_time=1.0,
        )
        lbl7f = Text("Clusters separated in data space too", font_size=26, color=WHITE)
        lbl7f.to_edge(UR, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl7f)
        self.play(FadeIn(lbl7f), run_time=0.7)
        self.wait(2.0)

        # Restore full view
        self.play(
            FadeOut(lbl7f),
            AnimationGroup(*[
                (grid_nodes[i].animate.set_opacity(1.0).set_color(u_color(i)).scale(1 / 1.6)
                 if i == bn
                 else grid_nodes[i].animate.set_opacity(1.0).set_color(u_color(i)))
                for i in range(N_NODES)
            ], lag_ratio=0.0),
            AnimationGroup(*[
                data_dots[i].animate.set_opacity(0.85)
                    .set_color(CLUSTER_COLORS[int(_raw_labels[i])])
                for i in range(len(DATA_PTS))
            ], lag_ratio=0.003),
            run_time=1.2,
        )
        self.wait(0.5)

        # ----------------------------------------------------------------
        # 7G — FINAL CLEAN VIEW
        # ----------------------------------------------------------------
        lbl7g = Text("Regions and boundaries together", font_size=30, color=WHITE)
        lbl7g.to_edge(DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(lbl7g)
        self.play(FadeIn(lbl7g), run_time=0.8)
        self.wait(3.5)
        self.play(FadeOut(lbl7g), run_time=0.8)
        self.wait(0.5)
