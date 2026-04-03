"""Test 8: Smooth transitions — random scatter → clustered blobs
Tests: FadeTransform, ReplacementTransform, AnimationGroup at 1080p/60fps."""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from manim import *

N = 150
CLUSTER_COLORS = [BLUE, RED, GREEN, YELLOW]


class Test8SmoothTransition(ThreeDScene):
    def construct(self):
        np.random.seed(42)

        # ── Data ────────────────────────────────────────────────────────────
        scatter_pts = np.random.uniform(-2.8, 2.8, size=(N, 3))

        cluster_pts, labels = make_blobs(
            n_samples=N, n_features=3, centers=4,
            cluster_std=0.45, random_state=42,
        )
        scaler = MinMaxScaler(feature_range=(-2.5, 2.5))
        cluster_pts = scaler.fit_transform(cluster_pts)

        # ── Initial scatter dots ────────────────────────────────────────────
        dots = VGroup(*[
            Dot3D(point=scatter_pts[i], radius=0.05, color=WHITE)
            for i in range(N)
        ])

        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)

        # Label: "Scattered"
        lbl_scatter = Text("Scattered", font_size=40, color=GREY_A)
        lbl_scatter.to_corner(UL)
        self.add_fixed_in_frame_mobjects(lbl_scatter)

        # ── Phase 1: FadeIn scatter with LaggedStart ─────────────────────────
        self.play(
            LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.015),
            FadeIn(lbl_scatter),
            run_time=2,
        )
        self.wait(0.5)

        # ── Phase 2: ReplacementTransform on first 50 dots ──────────────────
        # Build target dots at cluster positions (same count, different position/color)
        group_a = VGroup(*[dots[i] for i in range(50)])
        group_b = VGroup(*[
            Dot3D(point=cluster_pts[i], radius=0.05, color=CLUSTER_COLORS[labels[i]])
            for i in range(50)
        ])

        lbl_cluster = Text("Clustered", font_size=40, color=YELLOW)
        lbl_cluster.to_corner(UL)
        self.add_fixed_in_frame_mobjects(lbl_cluster)

        self.play(
            ReplacementTransform(group_a, group_b),
            FadeTransform(lbl_scatter, lbl_cluster),
            run_time=2,
        )
        self.wait(0.3)

        # ── Phase 3: AnimationGroup — move remaining 100 dots to cluster positions
        self.play(
            AnimationGroup(
                *[
                    dots[i].animate
                    .move_to(cluster_pts[i])
                    .set_color(CLUSTER_COLORS[labels[i]])
                    for i in range(50, N)
                ],
                lag_ratio=0.02,
            ),
            run_time=3,
        )
        self.wait(0.5)

        # ── Phase 4: FadeTransform on a second label pair ────────────────────
        lbl_done = Text("SOM Ready", font_size=40, color=GREEN)
        lbl_done.to_corner(UL)
        self.add_fixed_in_frame_mobjects(lbl_done)
        self.play(FadeTransform(lbl_cluster, lbl_done))
        self.wait(0.5)

        # ── Phase 5: Camera rotation — verify no dropped frames post-transition
        self.begin_ambient_camera_rotation(rate=0.25, about="theta")
        self.wait(4)
        self.stop_ambient_camera_rotation()
        self.wait(0.5)
