"""Test 2: 50 random Dot3D points in VGroup, animated with LaggedStart FadeIn"""
import numpy as np
from manim import *

class Test2GroupedDots(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        np.random.seed(42)
        points = np.random.uniform(-2, 2, size=(50, 3))

        dots = VGroup(*[
            Dot3D(point=p, radius=0.06, color=BLUE)
            for p in points
        ])

        self.play(LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.05))
        self.wait(1)
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(2)
        self.stop_ambient_camera_rotation()
