"""
Scene 3: Best Matching Unit (BMU) — Focused View  [REWRITE]

4×4 SOM grid (LEFT), 4 candidate weight nodes (RIGHT).
One input point competes against those nodes once — slowly and clearly.
No repeat, no weight update.

Parts:
  1  — Zoomed setup: 4×4 grid LEFT, 4 candidate nodes RIGHT
  2  — Input point + world-space "x" label
  3  — Distance lines + world-space "d" labels
  4  — Fixed screen text: "Compare distances" / "Closest wins" / formula
  5  — Winner highlight + world-space "winner" label
  6  — Grid link: highlight matching grid node + "same node" label
  7  — Equation fixed top-centre  c = argmin ||x - w_i||
  8  — End state: winner + equation + "Best Matching Unit" centred

Usage (low-quality preview):
    manim -ql --disable_caching scene3_bmu.py Scene3BMU

Usage (full quality):
    manim -qh --fps 60 scene3_bmu.py Scene3BMU
"""

import numpy as np
from manim import *

# ── Colours ───────────────────────────────────────────────────────────────────
INPUT_COLOR  = YELLOW
WEIGHT_COLOR = BLUE_D
GRID_COLOR   = BLUE_E
WINNER_COLOR = GREEN_B
CONN_COLOR   = TEAL_B
LINE_COLOR   = GREY_B
WIN_LINE     = GREEN_B
DIM_OP       = 0.12

# ── 4×4 Grid layout (LEFT side of scene) ─────────────────────────────────────
GRID_ROWS    = 4
GRID_COLS    = 4
GRID_SPACING = 0.55
GRID_CX      = -3.8
GRID_CY      = 0.0

def grid_pos(r: int, c: int) -> np.ndarray:
    x = GRID_CX + (c - (GRID_COLS - 1) / 2) * GRID_SPACING
    y = GRID_CY + (r - (GRID_ROWS - 1) / 2) * GRID_SPACING
    return np.array([x, y, 0.0])

GRID_POSITIONS = [grid_pos(r, c)
                  for r in range(GRID_ROWS) for c in range(GRID_COLS)]

# ── 4 candidate weight-vector positions (RIGHT side) ─────────────────────────
# Each also has a corresponding grid node index (diagonal of the 4×4 grid).
CAND_GRID_IDX = [0, 5, 10, 15]   # one per corner-ish diagonal

CAND_POSITIONS = [
    np.array([ 1.8,  1.1,  0.4]),   # → grid[0]
    np.array([ 2.6,  0.3, -0.3]),   # → grid[5]
    np.array([ 1.4, -0.7,  0.6]),   # → grid[10]
    np.array([ 2.2, -1.3,  0.1]),   # → grid[15]
]

# Input point — closest to CAND_POSITIONS[1]
INPUT_POS = np.array([2.5, 0.5, -0.2])


def _winner_idx() -> int:
    return int(np.argmin([np.linalg.norm(INPUT_POS - w)
                          for w in CAND_POSITIONS]))


# ── Scene ─────────────────────────────────────────────────────────────────────

class Scene3BMU(ThreeDScene):
    """Best Matching Unit — single, slow, clear demonstration."""

    def construct(self):

        # ══════════════════════════════════════════════════════════════════════
        # PART 1 — ZOOMED SETUP
        # ══════════════════════════════════════════════════════════════════════

        self.set_camera_orientation(phi=0 * DEGREES, theta=-90 * DEGREES,
                                    zoom=0.85)

        # ── 4×4 Grid (LEFT) ───────────────────────────────────────────────────
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

        self.play(Create(h_lines), Create(v_lines), run_time=1.0)
        self.play(
            LaggedStart(*[FadeIn(d) for d in grid_nodes], lag_ratio=0.035),
            run_time=1.2,
        )

        # ── 4 candidate weight nodes (RIGHT) ──────────────────────────────────
        cand_dots = VGroup(*[
            Dot3D(point=p, radius=0.12, color=WEIGHT_COLOR)
            for p in CAND_POSITIONS
        ])
        self.play(
            LaggedStart(*[FadeIn(d) for d in cand_dots], lag_ratio=0.18),
            run_time=1.0,
        )

        # Dim grid nodes that are NOT one of the 4 candidate representatives
        non_cand_anims = [
            grid_nodes[j].animate.set_opacity(DIM_OP)
            for j in range(len(GRID_POSITIONS))
            if j not in CAND_GRID_IDX
        ]
        self.play(*non_cand_anims, run_time=0.8)

        # Light camera tilt for slight 3-D feel
        self.move_camera(phi=15 * DEGREES, theta=-75 * DEGREES, zoom=1.1,
                         run_time=1.5)
        self.wait(0.4)

        # ══════════════════════════════════════════════════════════════════════
        # PART 2 — INPUT POINT + WORLD-SPACE "x" LABEL
        # ══════════════════════════════════════════════════════════════════════

        inp_dot = Dot3D(point=INPUT_POS, radius=0.14, color=INPUT_COLOR)
        self.play(FadeIn(inp_dot), run_time=0.8)

        inp_lbl = Text("x", font_size=22, color=INPUT_COLOR)
        inp_lbl.move_to(INPUT_POS + np.array([0.0, 0.40, 0.0]))
        self.play(FadeIn(inp_lbl), run_time=0.5)
        self.wait(0.8)

        # ══════════════════════════════════════════════════════════════════════
        # PART 3 — DISTANCE LINES + SMALL WORLD-SPACE "d" LABELS
        # ══════════════════════════════════════════════════════════════════════

        dists = [np.linalg.norm(INPUT_POS - CAND_POSITIONS[i])
                 for i in range(len(CAND_POSITIONS))]
        d_min, d_max = min(dists), max(dists)

        dist_lines = []
        for i in range(len(CAND_POSITIONS)):
            # Brighter colour = shorter distance
            col = interpolate_color(WHITE, GREY_C,
                                    (dists[i] - d_min) / (d_max - d_min + 1e-9))
            line = Line(INPUT_POS, CAND_POSITIONS[i],
                        color=col, stroke_width=2.0)
            dist_lines.append(line)

        self.play(
            LaggedStart(*[Create(l) for l in dist_lines], lag_ratio=0.22),
            run_time=1.6,
        )

        d_labels = []
        for i in range(len(CAND_POSITIONS)):
            mid = (INPUT_POS + CAND_POSITIONS[i]) / 2
            lbl = Text("d", font_size=18, color=GREY_A)
            lbl.move_to(mid + np.array([0.22, 0.22, 0.0]))
            d_labels.append(lbl)

        self.play(
            LaggedStart(*[FadeIn(l) for l in d_labels], lag_ratio=0.22),
            run_time=0.8,
        )
        self.wait(0.6)

        # ══════════════════════════════════════════════════════════════════════
        # PART 4 — FIXED SCREEN TEXT
        # Strict pattern: create → to_edge → add_fixed_in_frame → FadeIn
        # ══════════════════════════════════════════════════════════════════════

        # TOP LEFT
        lbl_compare = Text("Compare distances", font_size=28, color=GREY_A)
        lbl_compare.to_edge(UL, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl_compare)
        self.play(FadeIn(lbl_compare), run_time=0.6)

        # TOP RIGHT
        lbl_wins = Text("Closest wins", font_size=28, color=GREY_A)
        lbl_wins.to_edge(UR, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl_wins)
        self.play(FadeIn(lbl_wins), run_time=0.6)

        # BOTTOM
        lbl_formula = MathTex(r"\| x - w_i \|", font_size=34, color=GREY_A)
        lbl_formula.to_edge(DOWN, buff=0.40)
        self.add_fixed_in_frame_mobjects(lbl_formula)
        self.play(FadeIn(lbl_formula), run_time=0.6)

        self.wait(2.0)

        # ══════════════════════════════════════════════════════════════════════
        # PART 5 — FIND WINNER
        # ══════════════════════════════════════════════════════════════════════

        win_idx = _winner_idx()

        # Dim non-winner lines, nodes, and "d" labels
        fade_anims = []
        for i in range(len(CAND_POSITIONS)):
            if i != win_idx:
                fade_anims.append(
                    dist_lines[i].animate.set_stroke(opacity=DIM_OP))
                fade_anims.append(
                    cand_dots[i].animate.set_opacity(DIM_OP))
                fade_anims.append(
                    d_labels[i].animate.set_opacity(DIM_OP))
        self.play(*fade_anims, run_time=1.0)

        # Brighten winner line + node
        self.play(
            dist_lines[win_idx].animate.set_stroke(
                color=WIN_LINE, width=4.0, opacity=1.0),
            cand_dots[win_idx].animate.set_color(WINNER_COLOR).scale(1.5),
            run_time=1.0,
        )

        winner_lbl = Text("winner", font_size=20, color=WINNER_COLOR)
        winner_lbl.move_to(CAND_POSITIONS[win_idx] + np.array([0.0, 0.44, 0.0]))
        self.play(FadeIn(winner_lbl), run_time=0.5)
        self.wait(1.2)

        # ══════════════════════════════════════════════════════════════════════
        # PART 6 — GRID LINK
        # Highlight the corresponding grid node (LEFT) and draw a brief
        # connection line to the winning weight node (RIGHT).
        # ══════════════════════════════════════════════════════════════════════

        g_idx = CAND_GRID_IDX[win_idx]
        g_dot = grid_nodes[g_idx]

        self.play(
            g_dot.animate.set_opacity(1.0)
                         .set_color(WINNER_COLOR)
                         .scale(1.6),
            run_time=0.8,
        )

        conn_line = Line(GRID_POSITIONS[g_idx], CAND_POSITIONS[win_idx],
                         color=CONN_COLOR, stroke_width=2.0)
        self.play(Create(conn_line), run_time=0.9)

        same_node_lbl = Text("same node", font_size=18, color=CONN_COLOR)
        same_node_lbl.move_to(
            (GRID_POSITIONS[g_idx] + CAND_POSITIONS[win_idx]) / 2
            + np.array([0.0, 0.40, 0.0])
        )
        self.play(FadeIn(same_node_lbl), run_time=0.5)
        self.wait(1.5)

        # ══════════════════════════════════════════════════════════════════════
        # PART 7 — EQUATION (FIXED, TOP CENTRE)
        # Remove corner labels first so equation has clear space at the top.
        # Strict pattern: create → to_edge → add_fixed_in_frame → FadeIn
        # ══════════════════════════════════════════════════════════════════════

        self.play(FadeOut(lbl_compare), FadeOut(lbl_wins), run_time=0.5)
        self.remove(lbl_compare, lbl_wins)

        lbl_eq = MathTex(r"c = \arg\min_i \| x - w_i \|",
                         font_size=36, color=WHITE)
        lbl_eq.to_edge(UP, buff=0.35)
        self.add_fixed_in_frame_mobjects(lbl_eq)
        self.play(FadeIn(lbl_eq), run_time=0.8)
        self.wait(2.5)

        # ══════════════════════════════════════════════════════════════════════
        # PART 8 — END STATE
        # Keep: winning weight node, winning grid node, equation.
        # Fade all else. Show centred "Best Matching Unit".
        # ══════════════════════════════════════════════════════════════════════

        fade_out_mob = VGroup(
            *[dist_lines[i] for i in range(len(dist_lines))],
            *[cand_dots[i]  for i in range(len(cand_dots)) if i != win_idx],
            *[d_labels[i]   for i in range(len(d_labels))],
            inp_dot, inp_lbl, winner_lbl,
            conn_line, same_node_lbl,
            h_lines, v_lines,
            *[grid_nodes[j] for j in range(len(GRID_POSITIONS))
              if j != g_idx],
        )

        self.play(
            FadeOut(fade_out_mob),
            FadeOut(lbl_formula),
            run_time=1.2,
        )
        self.remove(lbl_formula)

        # Level camera back to flat
        self.move_camera(phi=0 * DEGREES, theta=-90 * DEGREES, zoom=1.0,
                         run_time=1.0)

        # CENTRE TEXT — strict pattern
        lbl_bmu = Text("Best Matching Unit", font_size=40, color=WHITE)
        lbl_bmu.move_to(ORIGIN)
        self.add_fixed_in_frame_mobjects(lbl_bmu)
        self.play(FadeIn(lbl_bmu), run_time=1.0)

        self.wait(2.5)
