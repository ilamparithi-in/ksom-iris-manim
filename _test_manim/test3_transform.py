"""Test 3: Transform scattered points → clustered points using animate.move_to"""
import numpy as np
from manim import *

class Test3Transformation(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        np.random.seed(7)

        # Scattered line of points along X axis
        n = 20
        scattered = [np.array([x, 0.0, 0.0]) for x in np.linspace(-3, 3, n)]

        # Target: all points cluster toward origin
        cluster_center = np.array([0.0, 0.0, 0.0])

        dots = VGroup(*[
            Dot3D(point=p, radius=0.08, color=ORANGE)
            for p in scattered
        ])

        self.add(dots)
        self.wait(0.5)

        # Animate each dot moving to cluster center with slight jitter
        anims = []
        for i, dot in enumerate(dots):
            jitter = np.random.uniform(-0.15, 0.15, size=3)
            target = cluster_center + jitter
            anims.append(dot.animate.move_to(target))

        self.play(AnimationGroup(*anims, lag_ratio=0.0), run_time=2)
        self.wait(1)
