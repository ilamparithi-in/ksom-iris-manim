"""Test 5: Large dataset — 300 Dot3D from make_blobs, LaggedStart FadeIn at 1080p/60fps"""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from manim import *


class Test5LargeDataset(ThreeDScene):
    def construct(self):
        # Generate 300 3D points in 4 clusters
        pts, labels = make_blobs(
            n_samples=300, n_features=3, centers=4,
            cluster_std=0.7, random_state=42,
        )
        scaler = MinMaxScaler(feature_range=(-2.5, 2.5))
        pts = scaler.fit_transform(pts)

        cluster_colors = [BLUE, RED, GREEN, YELLOW]

        dots = VGroup(*[
            Dot3D(point=pts[i], radius=0.05, color=cluster_colors[labels[i]])
            for i in range(len(pts))
        ])

        axes = ThreeDAxes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1], z_range=[-3, 3, 1],
            x_length=6, y_length=6, z_length=6,
        )

        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.add(axes)

        # LaggedStart FadeIn all 300 dots — stress test at 60fps
        self.play(
            LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.025),
            run_time=4,
        )
        self.wait(0.5)

        # Ambient rotation — verify smooth camera at 60fps
        self.begin_ambient_camera_rotation(rate=0.3, about="theta")
        self.wait(4)
        self.stop_ambient_camera_rotation()

        # Zoom + phi sweep
        self.move_camera(phi=50 * DEGREES, theta=-90 * DEGREES, zoom=1.2, run_time=2)
        self.wait(0.5)
        self.move_camera(phi=75 * DEGREES, theta=-45 * DEGREES, zoom=1.0, run_time=2)
        self.wait(0.5)
