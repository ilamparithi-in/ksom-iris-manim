"""Test 6: MathTex transitions in a ThreeDScene — crisp text, no flicker, readable at 1080p"""
import numpy as np
from manim import *


class Test6EquationRendering(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1], z_range=[-3, 3, 1],
            x_length=5, y_length=5, z_length=5,
        )
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        self.add(axes)

        # Backdrop: 30 random dots to confirm text readability over 3D content
        np.random.seed(0)
        pts = np.random.uniform(-2, 2, size=(30, 3))
        dots = VGroup(*[Dot3D(point=p, radius=0.05, color=BLUE_D) for p in pts])
        self.add(dots)

        # --- Section label ---
        title = Text("SOM Update Rule", font_size=36, color=GREY_A)
        title.to_corner(UL)
        self.add_fixed_in_frame_mobjects(title)
        self.play(FadeIn(title))
        self.wait(0.5)

        # --- Equation 1: input space ---
        eq1 = MathTex(r"x \in \mathbb{R}^n", font_size=56, color=WHITE)
        eq1.to_corner(DR)
        self.add_fixed_in_frame_mobjects(eq1)
        self.play(FadeIn(eq1))
        self.wait(1.5)

        # --- Equation 2: distance measure ---
        eq2 = MathTex(r"\| x - w_i \|", font_size=56, color=YELLOW)
        eq2.to_corner(DR)
        self.add_fixed_in_frame_mobjects(eq2)
        self.play(FadeOut(eq1), FadeIn(eq2))
        self.wait(1.5)

        # --- Equation 3: weight update rule ---
        eq3 = MathTex(
            r"w_i(t{+}1) = w_i(t) + \alpha(t)\bigl(x - w_i(t)\bigr)",
            font_size=44,
            color=GREEN_B,
        )
        eq3.to_corner(DR)
        self.add_fixed_in_frame_mobjects(eq3)
        self.play(FadeOut(eq2), FadeIn(eq3))
        self.wait(0.5)

        # Camera rotation while full equation is on screen — verify no flicker
        self.begin_ambient_camera_rotation(rate=0.2, about="theta")
        self.wait(4)
        self.stop_ambient_camera_rotation()

        self.play(FadeOut(eq3), FadeOut(title))
        self.wait(0.5)
