"""
Bonus Scene: Complete SOM Training to Convergence

4x4 SOM on a make_blobs 3D dataset (200 pts, 4 clusters).

Parts:
  1  -- Data cloud + 4x4 grid + initial random weights
  2  -- Step 1 in full detail: BMU selection, neighborhood glow, weight update
  3  -- Step 2 in full detail (same treatment)
  4  -- Steps 3-200 sped up (every 10 steps batched, weights fly to place)
         α(t) and σ(t) shown in UL corner throughout and updated every batch
  5  -- Convergence banner
  6  -- Final map: grid-to-weight connection lines + slow spin-out

Camera: phi=45° throughout, continuous ambient theta rotation.

Usage:
    manim -ql --disable_caching scene_bonus_training.py SceneBonusTraining
    manim -qh --fps 60 scene_bonus_training.py SceneBonusTraining
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from manim import *

DEFAULT_FONT = "CMU Serif"
Text.set_default(font=DEFAULT_FONT)

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
INPUT_COLOR  = YELLOW
WEIGHT_COLOR = TEAL_B
GRID_COLOR   = BLUE_E
WINNER_COLOR = GREEN_B
DATA_COLOR   = GREY_B

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
_pts, _labels = make_blobs(
    n_samples=200, centers=4, n_features=3,
    cluster_std=0.65, random_state=42,
)
_scaler = MinMaxScaler(feature_range=(-2.0, 2.0))
DATA_PTS = _scaler.fit_transform(_pts)   # (200, 3)
DATA_PTS[:, 0] += 2.5                    # shift to right side of scene

# ---------------------------------------------------------------------------
# SOM layout
# ---------------------------------------------------------------------------
GRID_ROWS    = 4
GRID_COLS    = 4
N_NODES      = GRID_ROWS * GRID_COLS
SOM_ITERS    = 200
ALPHA0       = 0.5
SIGMA0       = 2.0

GRID_SPACING = 0.55
GRID_CX      = -3.6
GRID_CY      =  0.0

_grid_coords = np.array(
    [[r, c] for r in range(GRID_ROWS) for c in range(GRID_COLS)], dtype=float
)


def grid_pos(r: int, c: int) -> np.ndarray:
    x = GRID_CX + (c - (GRID_COLS - 1) / 2.0) * GRID_SPACING
    y = GRID_CY + (r - (GRID_ROWS - 1) / 2.0) * GRID_SPACING
    return np.array([x, y, 0.0])


GRID_POSITIONS = [grid_pos(r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS)]


def h_val(bmu_idx: int, node_idx: int, sigma: float) -> float:
    d2 = float(np.sum((_grid_coords[bmu_idx] - _grid_coords[node_idx]) ** 2))
    return float(np.exp(-d2 / (2.0 * sigma ** 2)))


def alpha_at(t: int) -> float:
    return ALPHA0 * np.exp(-t / SOM_ITERS)


def sigma_at(t: int) -> float:
    return max(SIGMA0 * np.exp(-t / (SOM_ITERS / np.log(SIGMA0 + 1e-9))), 0.3)


# ---------------------------------------------------------------------------
# Pre-compute all training steps
# _all_weights[t] = weight array BEFORE step t  (index 0 = initial)
# _all_weights[SOM_ITERS] = final trained weights
# ---------------------------------------------------------------------------
_rng = np.random.default_rng(42)
_init_weights = _rng.uniform(-2.0, 2.0, size=(N_NODES, 3))
_init_weights[:, 0] += 2.5              # same x-shift as data

_all_weights = [_init_weights.copy()]
_all_bmu: list[int]       = []
_all_samples: list        = []
_all_alpha: list[float]   = []
_all_sigma: list[float]   = []

_weights = _init_weights.copy()
for _t in range(SOM_ITERS):
    _x     = DATA_PTS[_rng.integers(0, len(DATA_PTS))]
    _a     = alpha_at(_t)
    _s     = sigma_at(_t)
    _diffs = _weights - _x
    _bmu   = int(np.argmin(np.linalg.norm(_diffs, axis=1)))
    _gc    = _grid_coords - _grid_coords[_bmu]
    _h     = np.exp(-np.sum(_gc ** 2, axis=1) / (2.0 * _s ** 2))
    _weights += _a * _h[:, None] * (_x - _weights)

    _all_weights.append(_weights.copy())
    _all_bmu.append(_bmu)
    _all_samples.append(_x.copy())
    _all_alpha.append(_a)
    _all_sigma.append(_s)


# ===========================================================================
class SceneBonusTraining(ThreeDScene):
    """Complete SOM training: steps 1-2 in detail, rest sped up."""

    # ------------------------------------------------------------------
    # Helper: create the three HUD param labels at standard positions
    # ------------------------------------------------------------------
    def _make_param_labels(self, t: int, a: float, s: float):
        """Return (iter_lbl, alpha_lbl, sigma_lbl) pre-positioned, opacity=0."""
        il = Text(f"t = {t}", font_size=26, color=GREY_A)
        il.to_corner(UL, buff=0.35)
        il.set_opacity(0)

        al = MathTex(rf"\alpha = {a:.3f}", font_size=26, color=YELLOW)
        al.to_corner(UL, buff=0.35)
        al.shift(DOWN * 0.55)
        al.set_opacity(0)

        sl = MathTex(rf"\sigma = {s:.3f}", font_size=26, color=TEAL_B)
        sl.to_corner(UL, buff=0.35)
        sl.shift(DOWN * 1.10)
        sl.set_opacity(0)

        return il, al, sl

    def _add_param_labels(self, il, al, sl):
        self.add_fixed_in_frame_mobjects(il, al, sl)

    def _swap_param_labels(self, old_il, old_al, old_sl, new_il, new_al, new_sl,
                           extra_anims=None, run_time=0.45):
        """Fade out old labels, fade in new ones (all fixed-in-frame)."""
        anims = [
            FadeOut(old_il), FadeIn(new_il),
            FadeOut(old_al), FadeIn(new_al),
            FadeOut(old_sl), FadeIn(new_sl),
        ]
        if extra_anims:
            anims += extra_anims
        self.play(*anims, run_time=run_time)
        self.remove(old_il, old_al, old_sl)

    # ------------------------------------------------------------------
    def construct(self):

        # Camera: phi=45°, continuous ambient theta rotation
        self.set_camera_orientation(phi=45 * DEGREES, theta=-60 * DEGREES, zoom=0.85)
        self.begin_ambient_camera_rotation(rate=0.04, about='theta')

        # ==================================================================
        # PART 1 — INITIAL STATE
        # ==================================================================

        # Data cloud (right)
        data_dots = VGroup(*[
            Dot3D(point=DATA_PTS[i], radius=0.030, color=DATA_COLOR)
            for i in range(len(DATA_PTS))
        ])

        data_axes = ThreeDAxes(
            x_range=[-2.5, 2.5, 1], y_range=[-2.5, 2.5, 1], z_range=[-2.5, 2.5, 1],
            x_length=5.0, y_length=5.0, z_length=5.0,
            tips=False,
            axis_config={"stroke_color": GREY_D, "stroke_width": 0.6,
                         "include_ticks": False},
        ).shift(RIGHT * 2.5)

        # 4x4 topology grid (left)
        h_lines = VGroup(*[
            Line(GRID_POSITIONS[r * GRID_COLS + c],
                 GRID_POSITIONS[r * GRID_COLS + c + 1],
                 color=GRID_COLOR, stroke_width=1.0)
            for r in range(GRID_ROWS) for c in range(GRID_COLS - 1)
        ])
        v_lines = VGroup(*[
            Line(GRID_POSITIONS[r * GRID_COLS + c],
                 GRID_POSITIONS[(r + 1) * GRID_COLS + c],
                 color=GRID_COLOR, stroke_width=1.0)
            for r in range(GRID_ROWS - 1) for c in range(GRID_COLS)
        ])
        grid_nodes = VGroup(*[
            Dot3D(point=GRID_POSITIONS[i], radius=0.07, color=GRID_COLOR)
            for i in range(N_NODES)
        ])

        # Weight dots (right, initially random)
        cur_w = _init_weights.copy()
        w_dots = VGroup(*[
            Dot3D(point=cur_w[i], radius=0.10, color=WEIGHT_COLOR)
            for i in range(N_NODES)
        ])

        # Title
        lbl_title = Text("SOM Training", font_size=34, color=WHITE)
        lbl_title.to_edge(UP, buff=0.3)
        lbl_title.set_opacity(0)
        self.add_fixed_in_frame_mobjects(lbl_title)

        self.play(
            LaggedStart(*[FadeIn(d) for d in data_dots], lag_ratio=0.006),
            FadeIn(data_axes), FadeIn(lbl_title),
            run_time=2.5,
        )
        self.wait(0.5)

        self.play(Create(h_lines), Create(v_lines), run_time=1.2)
        self.play(
            LaggedStart(*[FadeIn(d) for d in grid_nodes], lag_ratio=0.04),
            LaggedStart(*[FadeIn(d) for d in w_dots],     lag_ratio=0.04),
            run_time=1.5,
        )
        self.wait(0.8)

        # Initial param HUD (t=0)
        lbl_iter, lbl_alpha_val, lbl_sigma_val = self._make_param_labels(
            0, ALPHA0, SIGMA0
        )
        self._add_param_labels(lbl_iter, lbl_alpha_val, lbl_sigma_val)
        self.play(FadeIn(lbl_iter), FadeIn(lbl_alpha_val), FadeIn(lbl_sigma_val),
                  run_time=0.4)
        self.wait(0.6)

        # ==================================================================
        # PART 2 — STEP 1 IN DETAIL
        # ==================================================================

        step_lbl = Text("Step 1 — Detail", font_size=28, color=WHITE)
        step_lbl.to_edge(UR, buff=0.35)
        step_lbl.set_opacity(0)
        self.add_fixed_in_frame_mobjects(step_lbl)
        self.play(FadeIn(step_lbl), run_time=0.5)

        x0      = _all_samples[0]
        bmu0    = _all_bmu[0]
        a0      = _all_alpha[0]
        s0      = _all_sigma[0]

        # Show input sample
        inp_dot = Dot3D(point=x0, radius=0.14, color=INPUT_COLOR)
        inp_lbl = Text("x", font_size=22, color=INPUT_COLOR)
        inp_lbl.move_to(x0 + np.array([0.0, 0.45, 0.0]))
        self.play(FadeIn(inp_dot), Write(inp_lbl), run_time=0.8)
        self.wait(0.4)

        # Highlight BMU on grid + weight space
        self.play(
            grid_nodes[bmu0].animate.set_color(WINNER_COLOR).scale(1.5),
            w_dots[bmu0].animate.set_color(WINNER_COLOR).scale(1.4),
            run_time=0.8,
        )
        bmu_lbl = Text("BMU", font_size=18, color=WINNER_COLOR)
        bmu_lbl.move_to(cur_w[bmu0] + np.array([0.0, 0.45, 0.0]))
        self.play(Write(bmu_lbl), run_time=0.4)
        self.wait(0.5)

        # Neighbourhood glow on grid
        grid_glow_anims = []
        for j in range(N_NODES):
            h = h_val(bmu0, j, s0)
            col = interpolate_color(GREY_D, WINNER_COLOR, h)
            grid_glow_anims.append(
                grid_nodes[j].animate.set_opacity(max(0.10, h)).set_color(col)
            )
        self.play(*grid_glow_anims, run_time=1.2)
        self.wait(0.5)

        # Weight update
        new_w1 = _all_weights[1]
        move_anims_1 = [
            w_dots[i].animate.move_to(new_w1[i])
            for i in range(N_NODES)
        ]
        self.play(LaggedStart(*move_anims_1, lag_ratio=0.05), run_time=2.5)
        cur_w = new_w1.copy()
        self.wait(0.5)

        # Update HUD to t=1
        ni, na, ns = self._make_param_labels(1, a0, s0)
        self._add_param_labels(ni, na, ns)
        self._swap_param_labels(lbl_iter, lbl_alpha_val, lbl_sigma_val, ni, na, ns,
                                run_time=0.5)
        lbl_iter, lbl_alpha_val, lbl_sigma_val = ni, na, ns

        # Clean up step 1
        self.play(
            FadeOut(inp_dot), FadeOut(inp_lbl), FadeOut(bmu_lbl),
            FadeOut(step_lbl),
            run_time=0.5,
        )
        self.remove(step_lbl)

        # Reset grid colours / scales
        reset_g1 = []
        for j in range(N_NODES):
            if j == bmu0:
                reset_g1.append(
                    grid_nodes[j].animate.set_opacity(1.0)
                                         .set_color(GRID_COLOR)
                                         .scale(1.0 / 1.5)
                )
            else:
                reset_g1.append(
                    grid_nodes[j].animate.set_opacity(1.0).set_color(GRID_COLOR)
                )
        self.play(*reset_g1, run_time=0.5)
        self.play(w_dots[bmu0].animate.set_color(WEIGHT_COLOR).scale(1.0 / 1.4),
                  run_time=0.3)
        self.wait(0.5)

        # ==================================================================
        # PART 3 — STEP 2 IN DETAIL
        # ==================================================================

        step_lbl2 = Text("Step 2 — Detail", font_size=28, color=WHITE)
        step_lbl2.to_edge(UR, buff=0.35)
        step_lbl2.set_opacity(0)
        self.add_fixed_in_frame_mobjects(step_lbl2)
        self.play(FadeIn(step_lbl2), run_time=0.5)

        x1   = _all_samples[1]
        bmu1 = _all_bmu[1]
        a1   = _all_alpha[1]
        s1   = _all_sigma[1]

        inp_dot2 = Dot3D(point=x1, radius=0.14, color=INPUT_COLOR)
        inp_lbl2 = Text("x", font_size=22, color=INPUT_COLOR)
        inp_lbl2.move_to(x1 + np.array([0.0, 0.45, 0.0]))
        self.play(FadeIn(inp_dot2), Write(inp_lbl2), run_time=0.8)
        self.wait(0.4)

        self.play(
            grid_nodes[bmu1].animate.set_color(WINNER_COLOR).scale(1.5),
            w_dots[bmu1].animate.set_color(WINNER_COLOR).scale(1.4),
            run_time=0.8,
        )
        bmu_lbl2 = Text("BMU", font_size=18, color=WINNER_COLOR)
        bmu_lbl2.move_to(cur_w[bmu1] + np.array([0.0, 0.45, 0.0]))
        self.play(Write(bmu_lbl2), run_time=0.4)
        self.wait(0.5)

        grid_glow_anims2 = []
        for j in range(N_NODES):
            h = h_val(bmu1, j, s1)
            col = interpolate_color(GREY_D, WINNER_COLOR, h)
            grid_glow_anims2.append(
                grid_nodes[j].animate.set_opacity(max(0.10, h)).set_color(col)
            )
        self.play(*grid_glow_anims2, run_time=1.2)
        self.wait(0.5)

        new_w2 = _all_weights[2]
        move_anims_2 = [w_dots[i].animate.move_to(new_w2[i]) for i in range(N_NODES)]
        self.play(LaggedStart(*move_anims_2, lag_ratio=0.05), run_time=2.5)
        cur_w = new_w2.copy()
        self.wait(0.5)

        # Update HUD to t=2
        ni2, na2, ns2 = self._make_param_labels(2, a1, s1)
        self._add_param_labels(ni2, na2, ns2)
        self._swap_param_labels(lbl_iter, lbl_alpha_val, lbl_sigma_val, ni2, na2, ns2,
                                run_time=0.5)
        lbl_iter, lbl_alpha_val, lbl_sigma_val = ni2, na2, ns2

        # Clean up step 2
        self.play(
            FadeOut(inp_dot2), FadeOut(inp_lbl2), FadeOut(bmu_lbl2),
            FadeOut(step_lbl2),
            run_time=0.5,
        )
        self.remove(step_lbl2)

        reset_g2 = []
        for j in range(N_NODES):
            if j == bmu1:
                reset_g2.append(
                    grid_nodes[j].animate.set_opacity(1.0)
                                         .set_color(GRID_COLOR)
                                         .scale(1.0 / 1.5)
                )
            else:
                reset_g2.append(
                    grid_nodes[j].animate.set_opacity(1.0).set_color(GRID_COLOR)
                )
        self.play(*reset_g2, run_time=0.5)
        self.play(w_dots[bmu1].animate.set_color(WEIGHT_COLOR).scale(1.0 / 1.4),
                  run_time=0.3)
        self.wait(0.5)

        # ==================================================================
        # PART 4 — SPEED-UP PHASE  (steps 3 … SOM_ITERS-1, batched)
        # ==================================================================

        speed_lbl = Text("Training...", font_size=28, color=WHITE)
        speed_lbl.to_edge(UR, buff=0.35)
        speed_lbl.set_opacity(0)
        self.add_fixed_in_frame_mobjects(speed_lbl)
        self.play(FadeIn(speed_lbl), run_time=0.3)

        BATCH = 10
        for t_start in range(2, SOM_ITERS, BATCH):
            t_end  = min(t_start + BATCH - 1, SOM_ITERS - 1)
            new_w  = _all_weights[t_end + 1]
            a_val  = _all_alpha[t_end]
            s_val  = _all_sigma[t_end]

            move_anims = [w_dots[i].animate.move_to(new_w[i]) for i in range(N_NODES)]

            ni_b, na_b, ns_b = self._make_param_labels(t_end + 1, a_val, s_val)
            self._add_param_labels(ni_b, na_b, ns_b)
            self._swap_param_labels(
                lbl_iter, lbl_alpha_val, lbl_sigma_val,
                ni_b, na_b, ns_b,
                extra_anims=move_anims,
                run_time=0.35,
            )
            lbl_iter, lbl_alpha_val, lbl_sigma_val = ni_b, na_b, ns_b
            cur_w = new_w.copy()

        self.play(FadeOut(speed_lbl), run_time=0.4)
        self.remove(speed_lbl)
        self.wait(1.0)

        # ==================================================================
        # PART 5 — CONVERGENCE BANNER
        # ==================================================================

        conv_lbl = Text("Converged!", font_size=42, color=GREEN_B)
        conv_lbl.move_to(ORIGIN)
        conv_lbl.set_opacity(0)
        self.add_fixed_in_frame_mobjects(conv_lbl)
        self.play(FadeIn(conv_lbl), run_time=1.0)
        self.wait(2.5)
        self.play(FadeOut(conv_lbl), run_time=0.7)
        self.remove(conv_lbl)

        # ==================================================================
        # PART 6 — FINAL MAP: grid-to-weight connection lines
        # ==================================================================

        final_w = _all_weights[SOM_ITERS]
        conn_lines = VGroup(*[
            Line(
                GRID_POSITIONS[i],
                final_w[i],
                color=GREY_C, stroke_width=0.8,
            )
            for i in range(N_NODES)
        ])

        lbl_map = Text("Learned map", font_size=30, color=WHITE)
        lbl_map.to_edge(UR, buff=0.35)
        lbl_map.set_opacity(0)
        self.add_fixed_in_frame_mobjects(lbl_map)

        self.play(
            LaggedStart(*[Create(l) for l in conn_lines], lag_ratio=0.05),
            FadeIn(lbl_map),
            run_time=2.5,
        )
        self.wait(3.5)

        self.play(FadeOut(lbl_map), FadeOut(conn_lines), run_time=0.8)
        self.remove(lbl_map)

        # Final banner
        lbl_final = Text("Structure revealed", font_size=40, color=TEAL_B)
        lbl_final.move_to(ORIGIN)
        lbl_final.set_opacity(0)
        self.add_fixed_in_frame_mobjects(lbl_final)
        self.play(FadeIn(lbl_final), run_time=1.0)
        self.wait(3.0)
        self.play(FadeOut(lbl_final), run_time=0.8)
        self.wait(0.5)
