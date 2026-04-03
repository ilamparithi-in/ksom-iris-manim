"""
Scene 2: The Grid and Dual Identity

A ~40-second ThreeDScene showing a 2D SOM grid on the left linked to 3D
weight vectors on the right, illustrating the dual identity of every node:
  LEFT  — fixed position in the 2D grid topology
  RIGHT — movable weight vector in 3D data space

Usage (low-quality preview):
    manim -ql --disable_caching scene2_grid_dual_identity.py Scene2GridDualIdentity

Usage (full quality):
    manim -qh --fps 60 scene2_grid_dual_identity.py Scene2GridDualIdentity
"""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from manim import *

# ── Colour palette ──────────────────────────────────────────────────────────────
GRID_NODE_COLOR = BLUE_D
GRID_EDGE_COLOR = BLUE_E
DATA_PALETTE    = [TEAL_B, PURPLE_B, ORANGE, PINK]
WEIGHT_COLOR    = YELLOW
CONN_COLOR      = GREY_B
HIGHLIGHT_COLOR = RED_B

# ── Grid geometry ───────────────────────────────────────────────────────────────
GRID_ROWS    = 8
GRID_COLS    = 8
GRID_SPACING = 0.60
GRID_CX      = -4.2       # world-space x-centre of the grid

# ── Data space geometry ─────────────────────────────────────────────────────────
DATA_CX   = 3.2           # world-space x-centre of axes / point cloud
DATA_HALF = 1.3           # half-range for axis labels and blob scaling


def grid_pos(r: int, c: int) -> np.ndarray:
    """World position of grid node at row r, column c (z = 0)."""
    x = GRID_CX + (c - (GRID_COLS - 1) / 2) * GRID_SPACING
    y = (r - (GRID_ROWS - 1) / 2) * GRID_SPACING
    return np.array([x, y, 0.0])


class Scene2GridDualIdentity(ThreeDScene):
    """LEFT = 2D grid (structure). RIGHT = 3D data space (weights)."""

    # ─── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _conn_line(gd: Dot3D, wd: Dot3D) -> VMobject:
        """always_redraw connection from grid node to weight vector."""
        return always_redraw(
            lambda g=gd, w=wd: Line(
                g.get_center(), w.get_center(),
                color=CONN_COLOR, stroke_width=1.2,
            )
        )

    # ─── construct ──────────────────────────────────────────────────────────────

    def construct(self):
        np.random.seed(42)

        # ══════════════════════════════════════════════════════════════════════
        # SETUP — camera, pre-compute geometry, generate blobs
        # ══════════════════════════════════════════════════════════════════════

        self.set_camera_orientation(phi=70 * DEGREES, theta=-50 * DEGREES, zoom=0.75)

        all_grid_pos = [
            grid_pos(r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS)
        ]

        # Blob point cloud, shifted to the right side of the scene
        pts, labels = make_blobs(
            n_samples=300, n_features=3, centers=4,
            cluster_std=0.6, random_state=42,
        )
        scaler = MinMaxScaler(feature_range=(-DATA_HALF, DATA_HALF))
        pts    = scaler.fit_transform(pts)
        pts[:, 0] += DATA_CX    # shift entirely to right half of screen

        # ══════════════════════════════════════════════════════════════════════
        # PART 1 — INTRODUCE GRID
        # Clean 8×8 grid: edges first → nodes on top.
        # ══════════════════════════════════════════════════════════════════════

        grid_nodes = VGroup(*[
            Dot3D(point=p, radius=0.06, color=GRID_NODE_COLOR)
            for p in all_grid_pos
        ])

        h_lines = VGroup(*[
            Line(
                all_grid_pos[r * GRID_COLS + c],
                all_grid_pos[r * GRID_COLS + c + 1],
                color=GRID_EDGE_COLOR, stroke_width=1.0,
            )
            for r in range(GRID_ROWS)
            for c in range(GRID_COLS - 1)
        ])
        v_lines = VGroup(*[
            Line(
                all_grid_pos[r * GRID_COLS + c],
                all_grid_pos[(r + 1) * GRID_COLS + c],
                color=GRID_EDGE_COLOR, stroke_width=1.0,
            )
            for r in range(GRID_ROWS - 1)
            for c in range(GRID_COLS)
        ])

        self.play(Create(h_lines), Create(v_lines), run_time=1.8)
        self.play(
            LaggedStart(*[FadeIn(d) for d in grid_nodes], lag_ratio=0.012),
            run_time=1.5,
        )
        self.wait(1.5)      # PAUSE — appreciate the clean grid

        # ══════════════════════════════════════════════════════════════════════
        # PART 2 — GRID AS STRUCTURE
        # Colour wave emphasises uniform spacing and regularity.
        # ══════════════════════════════════════════════════════════════════════

        lbl_grid = Text("A regular 2D grid", font_size=28, color=BLUE_B)
        lbl_grid.to_corner(UL)
        self.add_fixed_in_frame_mobjects(lbl_grid)
        self.play(FadeIn(lbl_grid), run_time=0.7)

        # Row-by-row colour ripple: brighten → restore
        self.play(
            LaggedStart(*[
                grid_nodes[r * GRID_COLS + c].animate.set_color(BLUE_A)
                for r in range(GRID_ROWS) for c in range(GRID_COLS)
            ], lag_ratio=0.025),
            run_time=2.0,
        )
        self.play(
            LaggedStart(*[
                grid_nodes[r * GRID_COLS + c].animate.set_color(GRID_NODE_COLOR)
                for r in range(GRID_ROWS) for c in range(GRID_COLS)
            ], lag_ratio=0.025),
            run_time=2.0,
        )
        self.wait(1.0)      # PAUSE — structure clarity

        # ══════════════════════════════════════════════════════════════════════
        # PART 3 — INTRODUCE DATA SPACE
        # 3D axes appear on the right; blob point cloud materialises.
        # ══════════════════════════════════════════════════════════════════════

        axes_3d = ThreeDAxes(
            x_range=[-DATA_HALF, DATA_HALF, 1],
            y_range=[-DATA_HALF, DATA_HALF, 1],
            z_range=[-DATA_HALF, DATA_HALF, 1],
            x_length=3.0,
            y_length=3.0,
            z_length=3.0,
        ).shift(RIGHT * DATA_CX)

        data_dots = VGroup(*[
            Dot3D(
                point=pts[i],
                radius=0.04,
                color=DATA_PALETTE[int(labels[i])],
            )
            for i in range(len(pts))
        ])

        lbl_data = Text("Data Space (3D)", font_size=28, color=TEAL_B)
        lbl_data.to_corner(UR)
        self.add_fixed_in_frame_mobjects(lbl_data)

        self.play(FadeIn(lbl_data), FadeIn(axes_3d), run_time=1.2)
        self.play(
            LaggedStart(*[FadeIn(d) for d in data_dots], lag_ratio=0.012),
            run_time=2.5,
        )
        self.wait(1.5)      # PAUSE — both spaces visible

        # ══════════════════════════════════════════════════════════════════════
        # PART 4 — INTRODUCE WEIGHT VECTORS
        # 5 nodes (4 corners + centre) gain yellow weight-vector spheres.
        # Connection lines link each grid node to its weight vector.
        # ══════════════════════════════════════════════════════════════════════

        sel_idx = [
            0 * GRID_COLS + 0,                              # top-left
            0 * GRID_COLS + (GRID_COLS - 1),                # top-right
            (GRID_ROWS - 1) * GRID_COLS + 0,                # bottom-left
            (GRID_ROWS - 1) * GRID_COLS + (GRID_COLS - 1),  # bottom-right
            (GRID_ROWS // 2) * GRID_COLS + (GRID_COLS // 2),# centre
        ]
        sel_grid = [grid_nodes[i] for i in sel_idx]

        rng = np.random.default_rng(seed=7)
        w_init = [
            np.array([
                DATA_CX + rng.uniform(-DATA_HALF * 0.7, DATA_HALF * 0.7),
                rng.uniform(-DATA_HALF * 0.7, DATA_HALF * 0.7),
                rng.uniform(-DATA_HALF * 0.5, DATA_HALF * 0.5),
            ])
            for _ in sel_idx
        ]

        w_dots = VGroup(*[
            Dot3D(point=p, radius=0.12, color=WEIGHT_COLOR)
            for p in w_init
        ])

        # always_redraw connection lines (for dynamic updates in parts 7 & 8)
        conn_list = [self._conn_line(sel_grid[k], w_dots[k]) for k in range(len(sel_idx))]

        # Static lines used only for the Create animation
        static_conns = [
            Line(
                sel_grid[k].get_center(), w_dots[k].get_center(),
                color=CONN_COLOR, stroke_width=1.2,
            )
            for k in range(len(sel_idx))
        ]

        # Highlight selected grid nodes to distinguish them from the rest
        self.play(
            AnimationGroup(*[g.animate.set_color(YELLOW_B) for g in sel_grid], lag_ratio=0.15),
            run_time=1.0,
        )
        # Weight vectors appear first, then connection lines are drawn toward them
        self.play(
            LaggedStart(*[FadeIn(wd) for wd in w_dots], lag_ratio=0.18),
            run_time=1.5,
        )
        self.play(
            LaggedStart(*[Create(c) for c in static_conns], lag_ratio=0.18),
            run_time=1.2,
        )
        # Swap static lines for always_redraw versions (needed for parts 7 & 8)
        for c in static_conns:
            self.remove(c)
        for conn in conn_list:
            self.add(conn)
        self.wait(1.0)

        # ══════════════════════════════════════════════════════════════════════
        # PART 5 — DUAL IDENTITY EXPLANATION
        # ══════════════════════════════════════════════════════════════════════

        lbl_dual  = Text("Each node has two identities",    font_size=30, color=WHITE)
        lbl_left  = Text("Position on grid",                font_size=24, color=BLUE_B)
        lbl_right = Text("Weight vector in data space",     font_size=24, color=YELLOW)

        lbl_dual.to_edge(UP, buff=0.35)
        lbl_left.to_corner(DL)
        lbl_right.to_corner(DR)

        # Set opacity=0 before adding so they are invisible until FadeIn plays
        lbl_dual.set_opacity(0)
        lbl_left.set_opacity(0)
        lbl_right.set_opacity(0)
        self.add_fixed_in_frame_mobjects(lbl_dual, lbl_left, lbl_right)
        self.play(FadeIn(lbl_dual), run_time=0.8)
        self.play(
            LaggedStart(FadeIn(lbl_left), FadeIn(lbl_right), lag_ratio=0.5),
            run_time=1.2,
        )
        self.wait(1.5)      # PAUSE — dual identity absorbed

        # ══════════════════════════════════════════════════════════════════════
        # PART 6 — HIGHLIGHT ONE NODE (centre)
        # Pulse grid node; highlight matching weight vector.
        # ══════════════════════════════════════════════════════════════════════

        # Clear explanation labels before the focus animation
        self.play(
            FadeOut(lbl_dual), FadeOut(lbl_left), FadeOut(lbl_right),
            run_time=0.7,
        )

        focus_k    = 4          # centre node (index into sel_idx)
        focus_grid = sel_grid[focus_k]
        focus_w    = w_dots[focus_k]

        # Pulse: scale up → back
        self.play(focus_grid.animate.scale(2.4).set_color(HIGHLIGHT_COLOR), run_time=0.5)
        self.play(focus_grid.animate.scale(1 / 2.4),                        run_time=0.4)

        # Highlight weight vector
        self.play(focus_w.animate.scale(1.6).set_color(HIGHLIGHT_COLOR),    run_time=0.6)
        self.wait(0.8)

        # ══════════════════════════════════════════════════════════════════════
        # PART 7 — MOVE WEIGHT VECTOR (grid stays fixed)
        # The always_redraw connection updates automatically.
        # ══════════════════════════════════════════════════════════════════════

        w_target = w_init[focus_k] + np.array([-0.45, 0.40, -0.28])
        self.play(focus_w.animate.move_to(w_target), run_time=2.0)
        self.wait(1.0)      # PAUSE — grid node fixed, vector moves

        # ══════════════════════════════════════════════════════════════════════
        # PART 8 — MULTIPLE NODES UPDATING TOGETHER
        # All 5 weight vectors shift while grid topology remains static.
        # ══════════════════════════════════════════════════════════════════════

        upd_targets = [
            w_init[k] + np.array([
                rng.uniform(-0.50, 0.50),
                rng.uniform(-0.50, 0.50),
                rng.uniform(-0.30, 0.30),
            ])
            for k in range(len(sel_idx))
        ]

        self.play(
            AnimationGroup(*[
                w_dots[k].animate.move_to(upd_targets[k])
                for k in range(len(sel_idx))
            ], lag_ratio=0.12),
            run_time=2.5,
        )
        self.wait(0.5)

        # Restore focus-node colours
        self.play(
            focus_grid.animate.set_color(YELLOW_B),
            focus_w.animate.scale(1 / 1.6).set_color(WEIGHT_COLOR),
            run_time=0.6,
        )

        # ══════════════════════════════════════════════════════════════════════
        # MATHEMATICAL OVERLAY
        # ══════════════════════════════════════════════════════════════════════

        math_lbl = MathTex(r"w_i \in \mathbb{R}^n",  font_size=36, color=GREY_A)
        math_lbl.to_corner(DR)
        idx_lbl  = MathTex(r"i = \text{grid index}", font_size=30, color=GREY_A)
        idx_lbl.next_to(math_lbl, UP, buff=0.20)

        self.add_fixed_in_frame_mobjects(math_lbl, idx_lbl)
        self.play(FadeIn(math_lbl), run_time=0.9)
        self.wait(0.4)
        self.play(FadeIn(idx_lbl),  run_time=0.7)

        # ── Final slow ambient rotation ───────────────────────────────────────
        self.begin_ambient_camera_rotation(rate=0.08, about="theta")
        self.wait(5.0)
        self.stop_ambient_camera_rotation()

        # ── Final fade ────────────────────────────────────────────────────────
        for conn in conn_list:
            self.remove(conn)
        self.play(
            FadeOut(grid_nodes),
            FadeOut(h_lines),
            FadeOut(v_lines),
            FadeOut(data_dots),
            FadeOut(axes_3d),
            FadeOut(w_dots),
            FadeOut(lbl_grid),
            FadeOut(lbl_data),
            FadeOut(math_lbl),
            FadeOut(idx_lbl),
            run_time=1.5,
        )
        self.wait(0.3)
