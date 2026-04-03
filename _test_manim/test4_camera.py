"""Test 4: Camera control — set_camera_orientation and begin_ambient_camera_rotation"""
from manim import *

class Test4CameraControl(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-3, 3, 1],
            x_length=6, y_length=6, z_length=6,
        )
        label = Text("Camera Test", font_size=36)
        self.add_fixed_in_frame_mobjects(label)
        label.to_corner(UL)

        # Start overhead
        self.set_camera_orientation(phi=0 * DEGREES, theta=-90 * DEGREES, zoom=0.8)
        self.add(axes)
        self.wait(0.5)

        # Move to angled view
        self.move_camera(phi=70 * DEGREES, theta=-45 * DEGREES, zoom=1.0, run_time=2)
        self.wait(0.5)

        # Ambient rotation
        self.begin_ambient_camera_rotation(rate=0.25, about='theta')
        self.wait(3)
        self.stop_ambient_camera_rotation()

        # Tilt on phi
        self.begin_ambient_camera_rotation(rate=0.1, about='phi')
        self.wait(2)
        self.stop_ambient_camera_rotation(about='phi')
        self.wait(0.5)
