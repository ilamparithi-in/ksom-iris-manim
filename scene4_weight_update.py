"""
Scene 4: Weight Update + Neighborhood Influence

4x4 SOM grid (LEFT), 4 weight nodes + 1 input point (RIGHT).
Shows: BMU moves toward input, neighbors follow with decreasing magnitude.
No repeat, no multiple iterations, no time-decay.

Parts:
  1  -- Context: recreate Scene 3 end state (BMU already selected)
  2  -- Highlight BMU, screen text "Winner moves toward input"
  3  -- BMU weight moves toward input
  4  -- Introduce neighborhood, h-value color gradient on grid
  5  -- Neighborhood function equation + radial glow on grid
  6  -- Neighbor updates (staggered LaggedStart, magnitude encodes distance)
  7  -- Distant node label "far -> small change"
  8  -- Full update equation at bottom (fixed)
  9  -- Single step summary: ghost dots + "One training step"
  10 -- Prep for loop: fade labels, keep updated weights + grid

Usage:
    manim -ql --disable_caching scene4_weight_update.py Scene4WeightUpdate
    manim -qh --fps 60 scene4_weight_update.py Scene4WeightUpdate
"""

import numpy as np
from manim import *

DEFAULT_FONT = "CMU Serif"
Text.set_default(font=DEFAULT_FONT)

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
INPUT_COLOR    = YELLOW
WEIGHT_COLOR   = BLUE_D
GRID_COLOR     = BLUE_E
WINNER_COLOR   = GREEN_B
NEIGHBOR_COLOR = TEAL_B
DIM_OP         = 0.12

# ---------------------------------------------------------------------------
# 4x4 Grid layout  (LEFT side, x ~ -3.6)
# ---------------------------------------------------------------------------
GRID_ROWS    = 4
GRID_COLS    = 4
GRID_SPACING = 0.55
GRID_CX      = -3.6
GRID_CY      = 0.0


def grid_pos(r: int, c: int) -> np.ndarray:
    x = GRID_CX + (c - (GRID_COLS - 1) / 2.0) * GRID_SPACING
    y = GRID_CY + (r - (GRID_ROWS - 1) / 2.0) * GRID_SPACING
    return np.array([x, y, 0.0])


GRID_POSITIONS = [grid_pos(r, c)
                  for r in range(GRID_ROWS) for c in range(GRID_COLS)]


def grid_row_col(idx: int):
    return divmod(idx, GRID_COLS)


def h_func(bmu_g: int, node_g: int, sigma: float = 1.5) -> float:
    """Gaussian neighborhood function h(c,i)."""
    rc, cc = grid_row_col(bmu_g)
    ri, ci = grid_row_col(node_g)
    d2 = (rc - ri) ** 2 + (cc - ci) ** 2
    return float(np.exp(-d2 / (2.0 * sigma ** 2)))


# ---------------------------------------------------------------------------
# 4 candidate weight vectors  (RIGHT side)
# Index 0 = BMU, 1 & 2 = close neighbours, 3 = far node
# ---------------------------------------------------------------------------
CAND_GRID_IDX = [5, 0, 10, 15]   # grid index for each candidate

_INIT_POS = np.array([
    [ 2.2,  0.4,  0.0],   # w0  BMU  -> grid 5  (row 1, col 1)
    [ 1.5,  1.5,  0.0],   # w1  near -> grid 0  (row 0, col 0)
    [ 1.5, -1.2,  0.0],   # w2  near -> grid 10 (row 2, col 2)
    [ 3.0, -0.6,  0.0],   # w3  far  -> grid 15 (row 3, col 3)
], dtype=float)

INPUT_POS = np.array([2.9, 0.8, 0.0])

BMU_IDX = 0     # w_dots[0] is closest to INPUT_POS
ALPHA   = 0.5
SIGMA   = 1.5


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

class Scene4WeightUpdate(ThreeDScene):
    """Weight update + neighbourhood influence -- one slow, clear training step."""

    def construct(self):

        # Working copy of positions (updated in-place as nodes animate)
        cur_pos = _INIT_POS.copy()

        # ====================================================================
        # PART 1 -- CONTEXT  (Scene 3 end state: BMU already selected)
        # ====================================================================

        self.set_camera_orientation(phi=15 * DEGREES, theta=-75 * DEGREES)

        # -- 4x4 grid (LEFT) -------------------------------------------------
        grid_nodes = VGroup(*[
            Dot3D(point=p, radius=0.07, color=GRID_COLOR)
            for p in GRID_POSITIONS
        ])
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

        # -- Weight nodes (RIGHT) + input ------------------------------------
        w_dots = VGroup(*[
            Dot3D(point=cur_pos[i], radius=0.12, color=WEIGHT_COLOR)
            for i in range(len(cur_pos))
        ])

        inp_dot = Dot3D(point=INPUT_POS, radius=0.14, color=INPUT_COLOR)
        inp_lbl = Text("x", font_size=24, color=INPUT_COLOR)
        inp_lbl.move_to(INPUT_POS + np.array([0.0, 0.44, 0.0]))

        # Bring in scene quickly (context recap)
        # ── Faint reference axes — data space (RIGHT) ────────────────────────
        data_axes = ThreeDAxes(
            x_range=[-1.5, 1.5, 1], y_range=[-1.5, 1.5, 1], z_range=[-1.2, 1.2, 1],
            x_length=3.0, y_length=3.0, z_length=2.4,
            tips=False,
            axis_config={"stroke_color": GREY_D, "stroke_width": 0.6,
                         "include_ticks": False},
        ).shift(RIGHT * 2.2)

        self.play(Create(h_lines), Create(v_lines), FadeIn(data_axes), run_time=0.8)
        self.play(
            LaggedStart(*[FadeIn(d) for d in grid_nodes], lag_ratio=0.03),
            LaggedStart(*[FadeIn(d) for d in w_dots],    lag_ratio=0.12),
            run_time=1.0,
        )
        self.play(FadeIn(inp_dot), FadeIn(inp_lbl), run_time=0.6)

        bmu_g_idx = CAND_GRID_IDX[BMU_IDX]   # = 5

        # Dim non-BMU weight nodes + most grid nodes (Scene 3 end state)
        dim_anims = (
            [w_dots[i].animate.set_opacity(DIM_OP)
             for i in range(len(w_dots)) if i != BMU_IDX]
            + [grid_nodes[j].animate.set_opacity(DIM_OP)
               for j in range(len(GRID_POSITIONS)) if j != bmu_g_idx]
        )
        self.play(*dim_anims, run_time=0.8)

        # Mark BMU as already-highlighted winner from Scene 3
        self.play(
            w_dots[BMU_IDX].animate.set_color(WINNER_COLOR).scale(1.4),
            grid_nodes[bmu_g_idx].animate.set_opacity(1.0)
                                         .set_color(WINNER_COLOR)
                                         .scale(1.5),
            run_time=0.7,
        )
        self.wait(0.8)

        # ====================================================================
        # PART 2 -- HIGHLIGHT BMU + screen text
        # ====================================================================

        # Pulse BMU weight node
        self.play(w_dots[BMU_IDX].animate.scale(1.3),        run_time=0.35)
        self.play(w_dots[BMU_IDX].animate.scale(1.0 / 1.3),  run_time=0.35)

        # Pulse grid node
        self.play(grid_nodes[bmu_g_idx].animate.scale(1.25), run_time=0.3)
        self.play(grid_nodes[bmu_g_idx].animate.scale(1.0 / 1.25), run_time=0.3)

        # STRICT TEXT PATTERN
        lbl_winner_moves = Text("Winner moves toward input",
                                font_size=28, color=WINNER_COLOR)
        lbl_winner_moves.to_edge(UL, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl_winner_moves)
        self.play(FadeIn(lbl_winner_moves), run_time=0.6)
        self.wait(1.2)

        # ====================================================================
        # PART 3 -- BMU WEIGHT MOVES TOWARD INPUT
        # ====================================================================

        # Save positions before any movement (used for ghost in Part 9)
        pre_step_pos = cur_pos.copy()

        self.stop_ambient_camera_rotation()
        bmu_new_pos = cur_pos[BMU_IDX] + ALPHA * (INPUT_POS - cur_pos[BMU_IDX])

        # Dashed pull-line shows direction
        pull_line = DashedLine(cur_pos[BMU_IDX], INPUT_POS,
                               color=WINNER_COLOR, stroke_width=1.8,
                               dash_length=0.10)
        self.play(Create(pull_line), run_time=0.8)
        self.wait(0.4)

        self.play(
            w_dots[BMU_IDX].animate.move_to(bmu_new_pos),
            FadeOut(pull_line),
            run_time=2.2,
        )
        cur_pos[BMU_IDX] = bmu_new_pos
        self.wait(1.0)
        self.begin_ambient_camera_rotation(rate=0.03, about='theta')

        # ====================================================================
        # PART 4 -- INTRODUCE NEIGHBOURHOOD (h-value color gradient on grid)
        # ====================================================================

        lbl_neighbors = Text("Neighbors move too", font_size=28, color=TEAL_B)
        lbl_neighbors.to_edge(UR, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl_neighbors)
        self.play(FadeIn(lbl_neighbors), run_time=0.6)

        # Colorize every grid node by its h value from the BMU grid node
        bmu_rc = grid_row_col(bmu_g_idx)
        grid_color_anims = []
        for j in range(len(GRID_POSITIONS)):
            rj, cj = grid_row_col(j)
            d2     = (bmu_rc[0] - rj) ** 2 + (bmu_rc[1] - cj) ** 2
            h_val  = float(np.exp(-d2 / (2.0 * SIGMA ** 2)))
            col    = interpolate_color(GREY_D, WINNER_COLOR, h_val)
            grid_color_anims.append(
                grid_nodes[j].animate.set_opacity(max(0.10, h_val))
                                     .set_color(col)
            )

        # Restore candidate weight nodes as neighbours
        restore_w = [
            w_dots[i].animate.set_opacity(0.85).set_color(NEIGHBOR_COLOR)
            for i in range(len(w_dots)) if i != BMU_IDX
        ]
        self.play(*grid_color_anims, *restore_w, run_time=1.8)
        self.wait(1.5)

        # ====================================================================
        # PART 5 -- NEIGHBOURHOOD FUNCTION EQUATION + radial glow
        # ====================================================================

        # Short form first
        lbl_h_short = MathTex(r"h_{ci}(t)", font_size=42, color=WHITE)
        lbl_h_short.to_edge(UP, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl_h_short)
        self.play(FadeIn(lbl_h_short), run_time=0.6)
        self.wait(1.0)

        # Expand to full Gaussian form
        lbl_h_full = MathTex(
            r"h_{ci}(t) = \exp\!\left("
            r"-\frac{\|r_c - r_i\|^2}{2\sigma^2}\right)",
            font_size=32, color=WHITE,
        )
        lbl_h_full.to_edge(UP, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl_h_full)
        self.play(FadeOut(lbl_h_short), FadeIn(lbl_h_full), run_time=1.0)
        self.remove(lbl_h_short)
        self.wait(1.5)

        self.stop_ambient_camera_rotation()
        # Radial glow rings centred on BMU grid node (world space)
        bmu_world = np.array(GRID_POSITIONS[bmu_g_idx])
        ring_radii = [0.28, 0.56, 0.88, 1.22]
        glow_rings = VGroup()
        for r_val in ring_radii:
            ring = Circle(radius=r_val, color=WINNER_COLOR, stroke_width=2.0)
            ring.set_stroke(opacity=max(0.06, 0.72 - r_val * 0.46))
            ring.move_to(bmu_world)
            glow_rings.add(ring)

        self.play(
            LaggedStart(*[Create(r) for r in glow_rings], lag_ratio=0.28),
            run_time=1.4,
        )
        self.wait(0.8)
        self.play(FadeOut(glow_rings), run_time=0.7)

        # ====================================================================
        # PART 6 -- NEIGHBOUR UPDATES (staggered; closer nodes move more)
        # ====================================================================

        self.stop_ambient_camera_rotation()
        move_anims = []
        for i in range(len(cur_pos)):
            if i == BMU_IDX:
                continue
            h_val   = h_func(bmu_g_idx, CAND_GRID_IDX[i], sigma=SIGMA)
            delta   = ALPHA * h_val * (INPUT_POS - cur_pos[i])
            new_pos = cur_pos[i] + delta
            cur_pos[i] = new_pos
            move_anims.append(w_dots[i].animate.move_to(new_pos))

        self.play(
            LaggedStart(*move_anims, lag_ratio=0.30),
            run_time=2.5,
        )
        self.wait(1.0)
        self.begin_ambient_camera_rotation(rate=0.03, about='theta')

        # ====================================================================
        # PART 7 -- DISTANT NODE: barely moved
        # ====================================================================

        far_idx = 3   # w3 -> grid 15 (furthest from BMU grid node 5)
        far_lbl = Text("far  ->  small change", font_size=18, color=GREY_A)
        far_lbl.move_to(cur_pos[far_idx] + np.array([0.0, 0.50, 0.0]))
        self.play(FadeIn(far_lbl), run_time=0.5)
        self.wait(2.0)
        self.play(FadeOut(far_lbl), run_time=0.5)

        # ====================================================================
        # PART 8 -- FULL UPDATE EQUATION (BOTTOM, FIXED)
        # ====================================================================

        lbl_update_eq = MathTex(
            r"w_i(t{+}1) = w_i(t) "
            r"+ \alpha(t)\,h_{ci}(t)\bigl(x - w_i(t)\bigr)",
            font_size=30, color=WHITE,
        )
        lbl_update_eq.to_edge(DOWN, buff=0.40)
        self.add_fixed_in_frame_mobjects(lbl_update_eq)
        self.play(FadeIn(lbl_update_eq), run_time=1.0)
        self.wait(3.0)

        # ====================================================================
        # PART 9 -- SINGLE STEP SUMMARY (ghost dots + "One training step")
        # ====================================================================

        self.stop_ambient_camera_rotation()
        ghost_dots = VGroup(*[
            Dot3D(
                point=pre_step_pos[i], radius=0.09,
                color=WINNER_COLOR if i == BMU_IDX else NEIGHBOR_COLOR,
            )
            for i in range(len(pre_step_pos))
        ])
        for gd in ghost_dots:
            gd.set_opacity(0.30)

        self.play(
            LaggedStart(*[FadeIn(gd) for gd in ghost_dots], lag_ratio=0.10),
            run_time=0.8,
        )

        lbl_step = Text("One training step", font_size=34, color=WHITE)
        lbl_step.move_to(ORIGIN)
        self.add_fixed_in_frame_mobjects(lbl_step)
        self.play(FadeIn(lbl_step), run_time=0.7)
        self.wait(2.2)

        self.play(FadeOut(lbl_step), FadeOut(ghost_dots), run_time=0.8)
        self.remove(lbl_step)
        self.begin_ambient_camera_rotation(rate=0.03, about='theta')

        # ====================================================================
        # PART 10 -- PREP FOR LOOP
        # Fade explanation labels; keep updated weights + grid visible.
        # ====================================================================

        self.play(
            FadeOut(lbl_winner_moves),
            FadeOut(lbl_neighbors),
            FadeOut(lbl_h_full),
            run_time=0.8,
        )
        self.remove(lbl_winner_moves, lbl_neighbors, lbl_h_full)
        self.stop_ambient_camera_rotation()
        self.wait(1.5)
