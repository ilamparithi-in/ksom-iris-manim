"""Test 7: Dual-space visualization — 2D SOM grid (left) mapped to 3D point cloud (right)
via always_redraw connection lines that update during animation."""
import numpy as np
from manim import *


GRID_ROWS = 4
GRID_COLS = 4
N_NODES = GRID_ROWS * GRID_COLS  # 16

GRID_ORIGIN_X = -3.5
GRID_SPACING = 0.8

CLOUD_X_BASE = 2.5


def grid_position(row: int, col: int) -> np.ndarray:
    x = GRID_ORIGIN_X + col * GRID_SPACING
    y = (row - (GRID_ROWS - 1) / 2.0) * GRID_SPACING
    return np.array([x, y, 0.0])


class Test7DualSpace(ThreeDScene):
    def construct(self):
        np.random.seed(77)

        # ── 2D SOM grid nodes ──────────────────────────────────────────────
        grid_positions = [
            grid_position(r, c)
            for r in range(GRID_ROWS)
            for c in range(GRID_COLS)
        ]
        grid_dots = VGroup(*[
            Dot3D(point=p, radius=0.07, color=GREEN)
            for p in grid_positions
        ])

        # Grid edges (static lines — grid topology never changes)
        grid_h_lines = VGroup()
        grid_v_lines = VGroup()
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS - 1):
                i, j = r * GRID_COLS + c, r * GRID_COLS + c + 1
                grid_h_lines.add(
                    Line(grid_positions[i], grid_positions[j],
                         color=GREEN_D, stroke_width=1.5)
                )
        for r in range(GRID_ROWS - 1):
            for c in range(GRID_COLS):
                i, j = r * GRID_COLS + c, (r + 1) * GRID_COLS + c
                grid_v_lines.add(
                    Line(grid_positions[i], grid_positions[j],
                         color=GREEN_D, stroke_width=1.5)
                )

        # ── 3D point cloud (input space) ───────────────────────────────────
        cloud_positions_init = [
            np.array([
                CLOUD_X_BASE + np.random.uniform(-0.9, 0.9),
                np.random.uniform(-2.0, 2.0),
                np.random.uniform(-1.2, 1.2),
            ])
            for _ in range(N_NODES)
        ]
        cloud_dots = VGroup(*[
            Dot3D(point=p, radius=0.08, color=RED)
            for p in cloud_positions_init
        ])

        # ── Connection lines that redraw every frame ────────────────────────
        def make_conn(i: int):
            return always_redraw(
                lambda i=i: Line(
                    grid_dots[i].get_center(),
                    cloud_dots[i].get_center(),
                    color=GREY_B,
                    stroke_width=1.0,
                )
            )

        connections = VGroup(*[make_conn(i) for i in range(N_NODES)])

        # ── Camera & labels ─────────────────────────────────────────────────
        self.set_camera_orientation(phi=65 * DEGREES, theta=-30 * DEGREES)

        lbl_grid = Text("SOM Map (2D)", font_size=30, color=GREEN)
        lbl_grid.to_corner(UL)
        lbl_cloud = Text("Input Space (3D)", font_size=30, color=RED)
        lbl_cloud.to_corner(UR)
        self.add_fixed_in_frame_mobjects(lbl_grid, lbl_cloud)

        # ── Scene build-up ──────────────────────────────────────────────────
        self.play(FadeIn(lbl_grid), FadeIn(lbl_cloud))

        # Show grid
        self.play(
            LaggedStart(*[FadeIn(d) for d in grid_dots], lag_ratio=0.05),
            Create(grid_h_lines),
            Create(grid_v_lines),
            run_time=2,
        )
        self.wait(0.3)

        # Show cloud
        self.play(
            LaggedStart(*[FadeIn(d) for d in cloud_dots], lag_ratio=0.05),
            run_time=2,
        )
        self.wait(0.3)

        # Draw connections
        self.add(connections)
        self.play(
            LaggedStart(*[Create(connections[i]) for i in range(N_NODES)], lag_ratio=0.04),
            run_time=1.5,
        )
        self.wait(0.5)

        # ── Animate "SOM weight update": cloud dots converge toward grid ────
        # Round 1: cloud dots move toward their grid node counterparts
        cloud_positions_upd1 = [
            grid_positions[i] + np.array([2.8, 0.0, np.random.uniform(-0.3, 0.3)])
            for i in range(N_NODES)
        ]
        self.play(
            AnimationGroup(
                *[cloud_dots[i].animate.move_to(cloud_positions_upd1[i])
                  for i in range(N_NODES)],
                lag_ratio=0.03,
            ),
            run_time=2.5,
        )
        self.wait(0.5)

        # Round 2: tighter convergence
        cloud_positions_upd2 = [
            grid_positions[i] + np.array([2.5, 0.0, np.random.uniform(-0.1, 0.1)])
            for i in range(N_NODES)
        ]
        self.play(
            AnimationGroup(
                *[cloud_dots[i].animate.move_to(cloud_positions_upd2[i])
                  for i in range(N_NODES)],
                lag_ratio=0.02,
            ),
            run_time=2.0,
        )
        self.wait(0.3)

        # ── Camera sweep while connections are live ─────────────────────────
        self.begin_ambient_camera_rotation(rate=0.25, about="theta")
        self.wait(4)
        self.stop_ambient_camera_rotation()
        self.wait(0.5)
