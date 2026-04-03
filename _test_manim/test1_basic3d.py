"""Test 1: Basic 3D scene — ThreeDAxes + Dot3D + camera rotation"""
from manim import *

class Test1Basic3D(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-3, 3, 1],
            x_length=6,
            y_length=6,
            z_length=6,
        )
        dot = Dot3D(point=axes.c2p(1, 1, 1), radius=0.12, color=YELLOW)

        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.add(axes, dot)
        self.begin_ambient_camera_rotation(rate=0.3)
        self.wait(3)
        self.stop_ambient_camera_rotation()
