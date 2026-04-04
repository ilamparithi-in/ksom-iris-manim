"""
Scene 5: Global → Local Organization

4x4 SOM grid (LEFT), 6 weight vectors in data space (RIGHT).
Three simulated training phases show decreasing neighbourhood size and
movement magnitude: global (large σ, α) → local (small σ, α) → convergence.

Parts:
  1  -- Initial state: grid + weights + "Training over time"
  2  -- Early phase: large σ/α, most grid glows, all nodes move far
  3  -- α(t) and σ(t) decay equations (fixed, top-centre)
  4  -- Shrinking neighbourhood glow rings on grid
  5  -- Mid phase: σ medium, 4-node glow, medium moves + "Refining structure"
  6  -- Late phase: σ small, only BMU + adj, tiny moves + "Fine tuning"
  7  -- Stabilisation: grid resets colour + "Convergence" (center)
  8  -- Before vs after: ghost initial positions vs final positions
  9  -- Summary: "Global → Local organization"

Usage:
    manim -ql --disable_caching scene5_org.py Scene5Organization
    manim -qh --fps 60 scene5_org.py Scene5Organization
"""

import numpy as np
from manim import *

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
INPUT_COLOR  = YELLOW
WEIGHT_COLOR = BLUE_D
GRID_COLOR   = BLUE_E
WINNER_COLOR = GREEN_B
DIM_OP       = 0.12

# ---------------------------------------------------------------------------
# 4×4 grid layout  (LEFT side)
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


GRID_POSITIONS = [
    grid_pos(r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS)
]


def grid_row_col(idx: int):
    return divmod(idx, GRID_COLS)


def h_func(bmu_g: int, node_g: int, sigma: float) -> float:
    """Gaussian neighbourhood function."""
    rc, cc = grid_row_col(bmu_g)
    ri, ci = grid_row_col(node_g)
    d2 = (rc - ri) ** 2 + (cc - ci) ** 2
    return float(np.exp(-d2 / (2.0 * sigma ** 2)))


# ---------------------------------------------------------------------------
# 6 weight vectors in data space (RIGHT side)
# Index 0 = BMU (fixed BMU for all phases for visual clarity)
# Indices 1-3 = nearby, 4-5 = far
# ---------------------------------------------------------------------------
WEIGHT_GRID_IDX = [5, 6, 9, 1, 0, 15]
#  5=(1,1) BMU, 6=(1,2) adj, 9=(2,1) adj, 1=(0,1) adj, 0=(0,0) far, 15=(3,3) far

INIT_WEIGHTS = np.array([
    [ 2.2,  0.3,  0.0],   # w0  BMU   -> grid 5
    [ 2.7,  1.0,  0.0],   # w1  adj   -> grid 6
    [ 1.8, -0.5,  0.0],   # w2  adj   -> grid 9
    [ 2.6, -0.8,  0.0],   # w3  adj   -> grid 1
    [ 1.3,  1.6,  0.0],   # w4  far   -> grid 0
    [ 3.2, -0.2,  0.0],   # w5  far   -> grid 15
], dtype=float)

BMU_G = WEIGHT_GRID_IDX[0]   # = 5  (grid index of BMU, fixed across phases)

# Phase parameters: (alpha, sigma, input_point)
PHASES = [
    (0.65, 2.0, np.array([2.8,  0.6,  0.0])),   # early  – large
    (0.35, 1.0, np.array([2.4, -0.1,  0.0])),   # mid    – medium
    (0.10, 0.5, np.array([2.6,  0.2,  0.0])),   # late   – fine
]


# ===========================================================================
class Scene5Organization(ThreeDScene):
    """Global → Local organisation over 3 simulated training phases."""

    def construct(self):
        self.set_camera_orientation(phi=15 * DEGREES, theta=-75 * DEGREES,
                                    zoom=1.0)

        # Working copy of weight positions (modified in-place each phase)
        cur_pos = INIT_WEIGHTS.copy()
        init_pos = INIT_WEIGHTS.copy()   # preserved for Part 8 before/after

        # ====================================================================
        # PART 1 — INITIAL STATE
        # ====================================================================

        # 4×4 topology grid (LEFT)
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

        # Weight dots (RIGHT)
        w_dots = VGroup(*[
            Dot3D(point=cur_pos[i], radius=0.12, color=WEIGHT_COLOR)
            for i in range(len(cur_pos))
        ])
        # Mark BMU as already-selected winner (from Scene 4)
        w_dots[0].set_color(WINNER_COLOR)
        grid_nodes[BMU_G].set_color(WINNER_COLOR)

        self.play(Create(h_lines), Create(v_lines), run_time=1.0)
        self.play(
            LaggedStart(*[FadeIn(d) for d in grid_nodes], lag_ratio=0.04),
            LaggedStart(*[FadeIn(d) for d in w_dots],    lag_ratio=0.10),
            run_time=1.2,
        )
        self.wait(0.5)

        # STRICT TEXT PATTERN ▼
        lbl_training = Text("Training over time", font_size=30, color=WHITE)
        lbl_training.to_edge(UL, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl_training)
        self.play(FadeIn(lbl_training), run_time=0.8)
        self.wait(1.0)

        # ====================================================================
        # PART 2 — EARLY PHASE: LARGE NEIGHBOURHOOD, LARGE MOVES
        # ====================================================================

        alpha0, sigma0, inp0_pt = PHASES[0]
        inp0 = Dot3D(point=inp0_pt, radius=0.14, color=INPUT_COLOR)
        inp0_lbl = Text("x", font_size=24, color=INPUT_COLOR)
        inp0_lbl.move_to(inp0_pt + np.array([0.0, 0.44, 0.0]))
        self.play(FadeIn(inp0), FadeIn(inp0_lbl), run_time=0.7)

        # Colorise ALL grid nodes by h value (sigma=2.0 → almost whole grid)
        grid_anims_early = []
        for j in range(len(GRID_POSITIONS)):
            h = h_func(BMU_G, j, sigma0)
            col = interpolate_color(GREY_D, GREEN_B, h)
            grid_anims_early.append(
                grid_nodes[j].animate.set_opacity(max(0.15, h)).set_color(col)
            )
        self.play(*grid_anims_early, run_time=1.2)
        self.wait(0.4)

        # STRICT TEXT PATTERN ▼
        lbl_large_nbr = Text("Large neighborhood", font_size=28, color=GREEN_B)
        lbl_large_nbr.to_edge(UR, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl_large_nbr)
        self.play(FadeIn(lbl_large_nbr), run_time=0.6)
        self.wait(0.4)

        # Move ALL 6 weight nodes — large displacements
        move_early = []
        for i in range(len(cur_pos)):
            h = h_func(BMU_G, WEIGHT_GRID_IDX[i], sigma0)
            new_p = cur_pos[i] + alpha0 * h * (inp0_pt - cur_pos[i])
            cur_pos[i] = new_p
            move_early.append(w_dots[i].animate.move_to(new_p))
        self.play(LaggedStart(*move_early, lag_ratio=0.10), run_time=2.5)
        self.wait(1.0)

        self.play(
            FadeOut(inp0), FadeOut(inp0_lbl),
            FadeOut(lbl_large_nbr),
            run_time=0.6,
        )
        self.remove(lbl_large_nbr)

        # ====================================================================
        # PART 3 — α(t) AND σ(t) DECAY EQUATIONS (fixed, top-centre)
        # ====================================================================

        # STRICT TEXT PATTERN ▼  — α(t)
        lbl_alpha = MathTex(r"\alpha(t)", font_size=46, color=WHITE)
        lbl_alpha.to_edge(UP, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl_alpha)
        self.play(FadeIn(lbl_alpha), run_time=0.6)
        self.wait(0.8)

        # STRICT TEXT PATTERN ▼  — α(t) ↓
        lbl_alpha_down = MathTex(r"\alpha(t) \downarrow", font_size=46, color=YELLOW)
        lbl_alpha_down.to_edge(UP, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl_alpha_down)
        self.play(FadeOut(lbl_alpha), FadeIn(lbl_alpha_down), run_time=0.8)
        self.remove(lbl_alpha)
        self.wait(0.8)

        # STRICT TEXT PATTERN ▼  — σ(t)
        lbl_sigma = MathTex(r"\sigma(t)", font_size=46, color=WHITE)
        lbl_sigma.to_edge(UP, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl_sigma)
        self.play(FadeOut(lbl_alpha_down), FadeIn(lbl_sigma), run_time=0.8)
        self.remove(lbl_alpha_down)
        self.wait(0.8)

        # STRICT TEXT PATTERN ▼  — σ(t) ↓
        lbl_sigma_down = MathTex(r"\sigma(t) \downarrow", font_size=46, color=YELLOW)
        lbl_sigma_down.to_edge(UP, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl_sigma_down)
        self.play(FadeOut(lbl_sigma), FadeIn(lbl_sigma_down), run_time=0.8)
        self.remove(lbl_sigma)
        self.wait(1.0)

        self.play(FadeOut(lbl_sigma_down), run_time=0.5)
        self.remove(lbl_sigma_down)

        # ====================================================================
        # PART 4 — SHRINKING NEIGHBOURHOOD VISUAL (glow rings on grid)
        # ====================================================================

        bmu_world = np.array(GRID_POSITIONS[BMU_G])

        # Phase 1 rings — large sigma (covers most of grid)
        rings_large = VGroup(*[
            Circle(radius=rv, color=GREEN_B, stroke_width=2.0)
            .set_stroke(opacity=max(0.07, 0.72 - rv * 0.22))
            .move_to(bmu_world)
            for rv in [0.28, 0.58, 0.90, 1.22, 1.55]
        ])
        self.play(
            LaggedStart(*[Create(r) for r in rings_large], lag_ratio=0.15),
            run_time=1.1,
        )
        self.wait(0.5)

        # Transition to medium sigma rings
        rings_med = VGroup(*[
            Circle(radius=rv, color=TEAL_B, stroke_width=2.0)
            .set_stroke(opacity=max(0.08, 0.75 - rv * 0.30))
            .move_to(bmu_world)
            for rv in [0.28, 0.56, 0.85]
        ])
        self.play(FadeOut(rings_large), run_time=0.4)
        self.play(
            LaggedStart(*[Create(r) for r in rings_med], lag_ratio=0.18),
            run_time=0.8,
        )
        self.wait(0.4)

        # Transition to small sigma rings
        rings_small = VGroup(*[
            Circle(radius=rv, color=BLUE_B, stroke_width=2.0)
            .set_stroke(opacity=max(0.12, 0.80 - rv * 0.45))
            .move_to(bmu_world)
            for rv in [0.26, 0.48]
        ])
        self.play(FadeOut(rings_med), run_time=0.4)
        self.play(
            LaggedStart(*[Create(r) for r in rings_small], lag_ratio=0.22),
            run_time=0.6,
        )
        self.wait(0.4)
        self.play(FadeOut(rings_small), run_time=0.4)

        # ====================================================================
        # PART 5 — MID PHASE: REFINING STRUCTURE
        # ====================================================================

        alpha1, sigma1, inp1_pt = PHASES[1]
        inp1 = Dot3D(point=inp1_pt, radius=0.14, color=INPUT_COLOR)
        inp1_lbl = Text("x", font_size=24, color=INPUT_COLOR)
        inp1_lbl.move_to(inp1_pt + np.array([0.0, 0.44, 0.0]))
        self.play(FadeIn(inp1), FadeIn(inp1_lbl), run_time=0.6)

        # Recolour grid: sigma=1.0 → only 3×3 centre glows meaningfully
        grid_anims_mid = []
        for j in range(len(GRID_POSITIONS)):
            h = h_func(BMU_G, j, sigma1)
            col = interpolate_color(GREY_D, TEAL_B, h)
            grid_anims_mid.append(
                grid_nodes[j].animate.set_opacity(max(0.12, h)).set_color(col)
            )
        self.play(*grid_anims_mid, run_time=1.0)
        self.wait(0.3)

        # STRICT TEXT PATTERN ▼
        lbl_refine = Text("Refining structure", font_size=28, color=TEAL_B)
        lbl_refine.to_edge(UR, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl_refine)
        self.play(FadeIn(lbl_refine), run_time=0.6)
        self.wait(0.3)

        # Move nodes with h > 0.10 (drops far nodes)
        move_mid = []
        for i in range(len(cur_pos)):
            h = h_func(BMU_G, WEIGHT_GRID_IDX[i], sigma1)
            if h < 0.10:
                continue
            new_p = cur_pos[i] + alpha1 * h * (inp1_pt - cur_pos[i])
            cur_pos[i] = new_p
            move_mid.append(w_dots[i].animate.move_to(new_p))
        self.play(LaggedStart(*move_mid, lag_ratio=0.15), run_time=2.0)
        self.wait(1.0)

        self.play(
            FadeOut(inp1), FadeOut(inp1_lbl),
            FadeOut(lbl_refine),
            run_time=0.6,
        )
        self.remove(lbl_refine)

        # ====================================================================
        # PART 6 — LATE PHASE: FINE TUNING
        # ====================================================================

        alpha2, sigma2, inp2_pt = PHASES[2]
        inp2 = Dot3D(point=inp2_pt, radius=0.14, color=INPUT_COLOR)
        inp2_lbl = Text("x", font_size=24, color=INPUT_COLOR)
        inp2_lbl.move_to(inp2_pt + np.array([0.0, 0.44, 0.0]))
        self.play(FadeIn(inp2), FadeIn(inp2_lbl), run_time=0.6)

        # Recolour: sigma=0.5 → barely 2×2 centre survives threshold
        grid_anims_late = []
        for j in range(len(GRID_POSITIONS)):
            h = h_func(BMU_G, j, sigma2)
            col = interpolate_color(GREY_D, BLUE_B, h)
            grid_anims_late.append(
                grid_nodes[j].animate.set_opacity(max(0.12, h)).set_color(col)
            )
        self.play(*grid_anims_late, run_time=0.8)
        self.wait(0.3)

        # STRICT TEXT PATTERN ▼
        lbl_fine = Text("Fine tuning", font_size=28, color=BLUE_B)
        lbl_fine.to_edge(UR, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl_fine)
        self.play(FadeIn(lbl_fine), run_time=0.6)
        self.wait(0.3)

        # Move only BMU + nearest (h > 0.10); tiny displacements
        move_late = []
        for i in range(len(cur_pos)):
            h = h_func(BMU_G, WEIGHT_GRID_IDX[i], sigma2)
            if h < 0.10:
                continue
            new_p = cur_pos[i] + alpha2 * h * (inp2_pt - cur_pos[i])
            cur_pos[i] = new_p
            move_late.append(w_dots[i].animate.move_to(new_p))
        self.play(LaggedStart(*move_late, lag_ratio=0.20), run_time=1.5)
        self.wait(1.0)

        self.play(
            FadeOut(inp2), FadeOut(inp2_lbl),
            FadeOut(lbl_fine),
            run_time=0.6,
        )
        self.remove(lbl_fine)

        # ====================================================================
        # PART 7 — STABILISATION / CONVERGENCE
        # ====================================================================

        # Reset grid to uniform neutral colour
        grid_reset = [
            grid_nodes[j].animate.set_opacity(0.85).set_color(GRID_COLOR)
            for j in range(len(GRID_POSITIONS))
        ]
        self.play(*grid_reset, run_time=1.2)
        self.wait(0.5)

        # STRICT TEXT PATTERN ▼  (center)
        lbl_converge = Text("Convergence", font_size=42, color=GREEN_B)
        lbl_converge.move_to(ORIGIN)
        self.add_fixed_in_frame_mobjects(lbl_converge)
        self.play(FadeIn(lbl_converge), run_time=0.8)
        self.wait(2.5)
        self.play(FadeOut(lbl_converge), run_time=0.7)
        self.remove(lbl_converge)

        # ====================================================================
        # PART 8 — BEFORE vs AFTER (ghost initial positions)
        # ====================================================================

        # Ghost dots at original weight positions (world space)
        ghost_dots = VGroup(*[
            Dot3D(point=init_pos[i], radius=0.10, color=GREY_B)
            for i in range(len(init_pos))
        ])
        for gd in ghost_dots:
            gd.set_opacity(0.30)

        self.play(
            LaggedStart(*[FadeIn(gd) for gd in ghost_dots], lag_ratio=0.08),
            run_time=0.8,
        )

        # World-space labels (not fixed to screen)
        ghost_anno = Text("before", font_size=18, color=GREY_B)
        ghost_anno.move_to(init_pos[4] + np.array([-0.10, 0.52, 0.0]))
        after_anno = Text("after", font_size=18, color=WINNER_COLOR)
        after_anno.move_to(cur_pos[0] + np.array([0.30, 0.52, 0.0]))
        self.play(FadeIn(ghost_anno), FadeIn(after_anno), run_time=0.7)
        self.wait(3.0)

        self.play(
            FadeOut(ghost_dots), FadeOut(ghost_anno), FadeOut(after_anno),
            run_time=0.8,
        )

        # ====================================================================
        # PART 9 — SUMMARY
        # ====================================================================

        self.play(FadeOut(lbl_training), run_time=0.5)
        self.remove(lbl_training)

        # STRICT TEXT PATTERN ▼  (center)
        lbl_summary = Text("Global  →  Local organization",
                           font_size=36, color=WHITE)
        lbl_summary.move_to(ORIGIN)
        self.add_fixed_in_frame_mobjects(lbl_summary)
        self.play(FadeIn(lbl_summary), run_time=0.9)
        self.wait(3.5)
        self.play(FadeOut(lbl_summary), run_time=0.8)
        self.remove(lbl_summary)
        self.wait(0.8)
