"""
Scene 3 (Rewritten): Best Matching Unit — Close-Up Demonstration

Structure:
  Part 1  — brief Scene-2-style overview → zoom into data space
  Part 2  — input point appears with local label
  Part 3  — 4 distance lines + local "distance" / "smallest" labels
  Part 4  — sequential line glow to guide attention → BMU line stays lit
  Part 5  — winner pulse, "winner" → "Best Matching Unit" label
  Part 6  — zoom out, dim data space, reveal corresponding grid node
  Part 7  — equation in 3 steps (text → norm → argmin), top-right fixed
  Part 8  — quick repeat with new input point
  Final   — dim all except winner, "Closest node wins"

Usage:
    manim -ql --disable_caching scene3_bmu.py Scene3BMU
    manim -qh --fps 60 scene3_bmu.py Scene3BMU
"""
import numpy as np
from manim import *

# ── Colour palette ─────────────────────────────────────────────────────────────
WEIGHT_COLOR = YELLOW
WINNER_COLOR = GREEN_B
INPUT_COLOR  = "#00FFFF"

# ── Geometry ───────────────────────────────────────────────────────────────────
GRID_ROWS    = 6
GRID_COLS    = 6
GRID_SPACING = 0.55
GRID_CX      = -3.8
DATA_CX      = 2.8        # x-centre of 3D data space


def grid_pos(r: int, c: int) -> np.ndarray:
    x = GRID_CX + (c - (GRID_COLS - 1) / 2) * GRID_SPACING
    y = (r - (GRID_ROWS - 1) / 2) * GRID_SPACING
    return np.array([x, y, 0.0])


def _line_color(d: float, d_min: float, d_max: float):
    """White = close, grey = far."""
    t = (d - d_min) / (d_max - d_min + 1e-9)
    return interpolate_color(WHITE, GREY_D, t)


def _line_width(d: float, d_min: float, d_max: float) -> float:
    t = (d - d_min) / (d_max - d_min + 1e-9)
    return max(0.8, 3.0 * (1.0 - t))


class Scene3BMU(ThreeDScene):
    """Close-up BMU demonstration."""

    def construct(self):
        np.random.seed(42)

        # ══════════════════════════════════════════════════════════════════════
        # PRE-COMPUTE GEOMETRY
        # ══════════════════════════════════════════════════════════════════════

        all_grid_pos = [
            grid_pos(r, c)
            for r in range(GRID_ROWS)
            for c in range(GRID_COLS)
        ]

        # 4 weight vectors — index 0 is clearly the closest to input
        inp_pos = np.array([DATA_CX + 0.25,  0.25,  0.15])
        w_pos = [
            np.array([DATA_CX + 0.10,  0.05,  0.10]),  # 0 — closest
            np.array([DATA_CX - 0.55,  0.45,  0.25]),  # 1 — medium
            np.array([DATA_CX + 0.65, -0.35,  0.30]),  # 2 — medium-far
            np.array([DATA_CX - 0.20, -0.75, -0.25]),  # 3 — farthest
        ]
        N = len(w_pos)

        dists        = [np.linalg.norm(inp_pos - w_pos[k]) for k in range(N)]
        bmu_idx      = int(np.argmin(dists))
        d_min, d_max = min(dists), max(dists)

        # Grid cells corresponding to each weight (used in Part 6)
        grid_sel = [(2, 2), (1, 1), (3, 4), (4, 2)]

        # Round-2: different input, different winner (index 3)
        inp2_pos       = np.array([DATA_CX - 0.45, -0.40, -0.20])
        dists2         = [np.linalg.norm(inp2_pos - w_pos[k]) for k in range(N)]
        bmu2_idx       = int(np.argmin(dists2))
        d2_min, d2_max = min(dists2), max(dists2)

        # ══════════════════════════════════════════════════════════════════════
        # PART 1 — BRIEF OVERVIEW → ZOOM IN
        # ══════════════════════════════════════════════════════════════════════

        self.set_camera_orientation(phi=70 * DEGREES, theta=-50 * DEGREES, zoom=0.72)

        # Grid (objects kept in memory for Part 6 copy)
        h_lines = VGroup(*[
            Line(
                all_grid_pos[r * GRID_COLS + c],
                all_grid_pos[r * GRID_COLS + c + 1],
                color=BLUE_E, stroke_width=0.9,
            )
            for r in range(GRID_ROWS) for c in range(GRID_COLS - 1)
        ])
        v_lines = VGroup(*[
            Line(
                all_grid_pos[r * GRID_COLS + c],
                all_grid_pos[(r + 1) * GRID_COLS + c],
                color=BLUE_E, stroke_width=0.9,
            )
            for r in range(GRID_ROWS - 1) for c in range(GRID_COLS)
        ])
        grid_nodes = VGroup(*[
            Dot3D(point=p, radius=0.055, color=BLUE_D)
            for p in all_grid_pos
        ])

        axes_3d = ThreeDAxes(
            x_range=[-1.5, 1.5, 1], y_range=[-1.5, 1.5, 1], z_range=[-1.5, 1.5, 1],
            x_length=2.8, y_length=2.8, z_length=2.8,
        ).shift(RIGHT * DATA_CX)

        w_overview = VGroup(*[
            Dot3D(point=p, radius=0.09, color=YELLOW_D)
            for p in w_pos
        ])

        self.play(Create(h_lines), Create(v_lines), run_time=0.8)
        self.play(
            LaggedStart(*[FadeIn(d) for d in grid_nodes], lag_ratio=0.015),
            run_time=0.9,
        )
        self.play(
            FadeIn(axes_3d),
            LaggedStart(*[FadeIn(d) for d in w_overview], lag_ratio=0.18),
            run_time=0.9,
        )
        self.wait(0.8)

        # Fade out overview, zoom into data space
        self.play(
            FadeOut(h_lines), FadeOut(v_lines),
            FadeOut(grid_nodes), FadeOut(axes_3d), FadeOut(w_overview),
            run_time=0.4,
        )
        self.move_camera(phi=62 * DEGREES, theta=-38 * DEGREES, zoom=1.45,
                         run_time=1.0)

        # ══════════════════════════════════════════════════════════════════════
        # PART 2 — INPUT POINT APPEARS
        # ══════════════════════════════════════════════════════════════════════

        w_dots = VGroup(*[
            Dot3D(point=p, radius=0.11, color=WEIGHT_COLOR)
            for p in w_pos
        ])
        self.play(
            LaggedStart(*[FadeIn(d) for d in w_dots], lag_ratio=0.18),
            run_time=0.8,
        )

        inp_dot = Dot3D(point=inp_pos, radius=0.16, color=INPUT_COLOR)
        self.play(FadeIn(inp_dot),                    run_time=0.45)
        self.play(inp_dot.animate.scale(1.5),          run_time=0.20)
        self.play(inp_dot.animate.scale(1 / 1.5),      run_time=0.20)

        inp_lbl = Text("input x", font_size=22, color=INPUT_COLOR)
        inp_lbl.move_to(inp_pos + np.array([0.0, 0.32, 0.0]))
        self.play(FadeIn(inp_lbl), run_time=0.4)
        self.wait(0.8)

        # ══════════════════════════════════════════════════════════════════════
        # PART 3 — DISTANCE LINES WITH LOCAL LABELS
        # ══════════════════════════════════════════════════════════════════════

        dist_lines = []
        dist_lbls  = []
        for k in range(N):
            col  = _line_color(dists[k], d_min, d_max)
            w    = _line_width(dists[k], d_min, d_max)
            line = Line(inp_pos, w_pos[k], color=col, stroke_width=w)
            dist_lines.append(line)

            mid  = (inp_pos + w_pos[k]) / 2 + np.array([0.0, 0.22, 0.0])
            text = "smallest" if k == bmu_idx else "distance"
            tcol = YELLOW if k == bmu_idx else GREY_B
            lbl  = Text(text, font_size=16, color=tcol)
            lbl.move_to(mid)
            dist_lbls.append(lbl)

        self.play(
            LaggedStart(*[Create(l) for l in dist_lines], lag_ratio=0.20),
            run_time=1.4,
        )
        self.play(
            LaggedStart(*[FadeIn(lbl) for lbl in dist_lbls], lag_ratio=0.20),
            run_time=1.1,
        )
        self.wait(0.8)

        # ══════════════════════════════════════════════════════════════════════
        # PART 4 — GUIDE ATTENTION: sequential glow, then leave BMU lit
        # ══════════════════════════════════════════════════════════════════════

        for k in range(N):
            self.play(
                dist_lines[k].animate.set_stroke(color=WHITE, width=3.5),
                run_time=0.28,
            )
            self.play(
                dist_lines[k].animate.set_stroke(
                    color=_line_color(dists[k], d_min, d_max),
                    width=_line_width(dists[k], d_min, d_max),
                ),
                run_time=0.22,
            )

        non_bmu = [k for k in range(N) if k != bmu_idx]
        self.play(
            *[dist_lines[k].animate.set_stroke(opacity=0.25, width=0.5)
              for k in non_bmu],
            dist_lines[bmu_idx].animate.set_stroke(color=WHITE, width=3.5),
            run_time=0.65,
        )
        self.wait(0.4)

        # ══════════════════════════════════════════════════════════════════════
        # PART 5 — BMU SELECTION
        # ══════════════════════════════════════════════════════════════════════

        non_bmu_lines = [dist_lines[k] for k in non_bmu]
        non_bmu_lbls  = [dist_lbls[k]  for k in non_bmu]
        self.play(
            *[FadeOut(l)  for l  in non_bmu_lines],
            *[FadeOut(lb) for lb in non_bmu_lbls],
            run_time=0.45,
        )

        self.play(
            w_dots[bmu_idx].animate.scale(1.7).set_color(WINNER_COLOR),
            run_time=0.40,
        )
        self.play(w_dots[bmu_idx].animate.scale(1 / 1.7), run_time=0.28)
        self.play(w_dots[bmu_idx].animate.scale(1.3),      run_time=0.25)

        lbl_pos    = w_pos[bmu_idx] + np.array([0.0, -0.32, 0.0])
        winner_lbl = Text("winner", font_size=20, color=WINNER_COLOR)
        winner_lbl.move_to(lbl_pos)
        self.play(FadeIn(winner_lbl), run_time=0.4)
        self.wait(0.5)

        bmu_lbl = Text("Best Matching Unit", font_size=20, color=WINNER_COLOR)
        bmu_lbl.move_to(lbl_pos)
        self.play(ReplacementTransform(winner_lbl, bmu_lbl), run_time=0.5)
        self.wait(0.8)

        # ══════════════════════════════════════════════════════════════════════
        # PART 6 — GRID CONNECTION (zoom out, highlight grid node)
        # ══════════════════════════════════════════════════════════════════════

        self.move_camera(phi=70 * DEGREES, theta=-50 * DEGREES, zoom=0.88,
                         run_time=0.9)

        self.play(
            w_dots.animate.set_opacity(0.3),
            inp_dot.animate.set_opacity(0.3),
            inp_lbl.animate.set_opacity(0.3),
            bmu_lbl.animate.set_opacity(0.3),
            dist_lines[bmu_idx].animate.set_stroke(opacity=0.2),
            dist_lbls[bmu_idx].animate.set_opacity(0.2),
            run_time=0.5,
        )

        # Dim grid copy (fresh objects since originals were faded out)
        grid_copy = VGroup(h_lines.copy(), v_lines.copy())
        grid_copy.set_opacity(0.3)
        self.play(FadeIn(grid_copy), run_time=0.5)

        bmu_r, bmu_c = grid_sel[bmu_idx]
        bmu_gnode = Dot3D(
            point=grid_pos(bmu_r, bmu_c), radius=0.14, color=WINNER_COLOR,
        )
        self.play(FadeIn(bmu_gnode),           run_time=0.35)
        self.play(bmu_gnode.animate.scale(2.0), run_time=0.25)
        self.play(bmu_gnode.animate.scale(0.5), run_time=0.22)

        conn = Line(
            grid_pos(bmu_r, bmu_c), w_pos[bmu_idx],
            color=WINNER_COLOR, stroke_width=2.5,
        )
        self.play(Create(conn), run_time=0.55)

        same_lbl = Text("same node", font_size=18, color=WINNER_COLOR)
        same_lbl.move_to(grid_pos(bmu_r, bmu_c) + np.array([-0.7, 0.28, 0.0]))
        self.play(FadeIn(same_lbl), run_time=0.4)
        self.wait(1.0)

        self.play(
            FadeOut(conn), FadeOut(bmu_gnode),
            FadeOut(grid_copy), FadeOut(same_lbl),
            run_time=0.45,
        )

        # ══════════════════════════════════════════════════════════════════════
        # PART 7 — EQUATION IN 3 STEPS (top-right, fixed to frame)
        # ══════════════════════════════════════════════════════════════════════

        self.play(
            w_dots.animate.set_opacity(1.0),
            inp_dot.animate.set_opacity(1.0),
            inp_lbl.animate.set_opacity(1.0),
            bmu_lbl.animate.set_opacity(1.0),
            dist_lines[bmu_idx].animate.set_stroke(opacity=1.0),
            dist_lbls[bmu_idx].animate.set_opacity(1.0),
            run_time=0.4,
        )

        # Step 1: plain text description
        eq1 = Text("distance between x and w\u1d62", font_size=26, color=GREY_A)
        eq1.to_corner(UR, buff=0.45)
        eq1.set_opacity(0)
        self.add_fixed_in_frame_mobjects(eq1)
        self.play(FadeIn(eq1), run_time=0.6)
        self.wait(1.0)

        # Step 2: norm notation
        eq2 = MathTex(r"\|x - w_i\|", font_size=42, color=GREY_A)
        eq2.to_corner(UR, buff=0.45)
        eq2.set_opacity(0)
        self.add_fixed_in_frame_mobjects(eq2)
        self.play(FadeOut(eq1), FadeIn(eq2), run_time=0.7)
        self.wait(1.0)

        # Step 3: argmin
        eq3 = MathTex(r"c = \arg\min_i \|x - w_i\|", font_size=36, color=GREY_A)
        eq3.to_corner(UR, buff=0.45)
        eq3.set_opacity(0)
        self.add_fixed_in_frame_mobjects(eq3)
        self.play(FadeOut(eq2), FadeIn(eq3), run_time=0.7)
        self.wait(1.2)

        # ══════════════════════════════════════════════════════════════════════
        # PART 8 — QUICK REPEAT WITH NEW INPUT
        # ══════════════════════════════════════════════════════════════════════

        self.play(
            FadeOut(inp_dot), FadeOut(inp_lbl),
            FadeOut(dist_lines[bmu_idx]), FadeOut(dist_lbls[bmu_idx]),
            FadeOut(bmu_lbl),
            w_dots[bmu_idx].animate.set_color(WEIGHT_COLOR).scale(1 / 1.3),
            run_time=0.55,
        )

        inp2_dot = Dot3D(point=inp2_pos, radius=0.16, color=INPUT_COLOR)
        self.play(FadeIn(inp2_dot),                      run_time=0.30)
        self.play(inp2_dot.animate.scale(1.35),           run_time=0.16)
        self.play(inp2_dot.animate.scale(1 / 1.35),       run_time=0.16)

        dist2_lines = [
            Line(
                inp2_pos, w_pos[k],
                color=_line_color(dists2[k], d2_min, d2_max),
                stroke_width=_line_width(dists2[k], d2_min, d2_max),
            )
            for k in range(N)
        ]
        self.play(
            LaggedStart(*[Create(l) for l in dist2_lines], lag_ratio=0.12),
            run_time=0.7,
        )
        self.wait(0.28)

        non_bmu2 = [dist2_lines[k] for k in range(N) if k != bmu2_idx]
        self.play(*[FadeOut(l) for l in non_bmu2], run_time=0.35)
        self.play(
            w_dots[bmu2_idx].animate.scale(1.5).set_color(WINNER_COLOR),
            dist2_lines[bmu2_idx].animate.set_stroke(color=WINNER_COLOR, width=3.0),
            run_time=0.40,
        )
        self.wait(0.7)

        # ══════════════════════════════════════════════════════════════════════
        # FINAL FRAME
        # ══════════════════════════════════════════════════════════════════════

        others = VGroup(*[w_dots[k] for k in range(N) if k != bmu2_idx])
        self.play(
            others.animate.set_opacity(0.2),
            inp2_dot.animate.set_opacity(0.2),
            dist2_lines[bmu2_idx].animate.set_stroke(opacity=0.35),
            run_time=0.45,
        )

        lbl_end = Text("Closest node wins", font_size=36, color=WHITE)
        lbl_end.to_edge(UP, buff=0.35)
        lbl_end.set_opacity(0)
        self.add_fixed_in_frame_mobjects(lbl_end)
        self.play(FadeIn(lbl_end), run_time=0.8)

        self.begin_ambient_camera_rotation(rate=0.04, about="theta")
        self.wait(4.5)
        self.stop_ambient_camera_rotation()

        self.play(
            FadeOut(w_dots), FadeOut(inp2_dot),
            FadeOut(dist2_lines[bmu2_idx]),
            FadeOut(lbl_end), FadeOut(eq3),
            run_time=1.2,
        )
        self.wait(0.3)
