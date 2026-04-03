"""
LatexTest — full LaTeX rendering validation for Manim CE.

Covers:
  Test 1: Basic MathTex rendering (centered, FadeIn)
  Test 2: Step-by-step Transform between 3 equations
  Test 3: Equation placed in 3D world-space with camera motion
  Test 4: Fixed overlay text that ignores camera rotation
  Test 5: Multiple equations arranged with VGroup + LaggedStart(FadeIn)

Run:
  manim -pql --renderer=opengl _test_manim/test_latex.py LatexTest
"""
import numpy as np
from manim import *


class LatexTest(ThreeDScene):
    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _section_banner(self, label: str, color=GREY_A) -> None:
        """Flash a brief section label (fixed to frame) then remove it."""
        banner = Text(label, font_size=28, color=color)
        banner.to_corner(UL)
        self.add_fixed_in_frame_mobjects(banner)
        self.play(FadeIn(banner, run_time=0.4))
        self.wait(0.3)
        self.play(FadeOut(banner, run_time=0.3))
        self.remove(banner)

    # ------------------------------------------------------------------ #
    #  Main                                                                #
    # ------------------------------------------------------------------ #
    def construct(self):
        # ── TEST 1: Basic rendering ─────────────────────────────────────
        self._section_banner("Test 1 · Basic Rendering")

        # Start with a flat (2D) default view so text is comfortably centred
        eq1 = MathTex(r"x \in \mathbb{R}^n", font_size=72, color=WHITE)
        eq1.move_to(ORIGIN)
        self.play(FadeIn(eq1, run_time=1.0))
        self.wait(1.5)

        # ── TEST 2: Step-by-step Transform ─────────────────────────────
        self._section_banner("Test 2 · Transforms")

        eq2 = MathTex(r"\| x - w_i \|", font_size=72, color=YELLOW)
        eq2.move_to(ORIGIN)

        eq3 = MathTex(
            r"w_i(t{+}1) = w_i(t) + \alpha(t)\bigl(x - w_i(t)\bigr)",
            font_size=48,
            color=GREEN_B,
        )
        eq3.move_to(ORIGIN)

        # eq1 → eq2
        self.play(Transform(eq1, eq2), run_time=1.2)
        self.wait(1.0)

        # eq1 (now visually eq2) → eq3
        self.play(Transform(eq1, eq3), run_time=1.4)
        self.wait(1.5)

        self.play(FadeOut(eq1))
        self.wait(0.3)

        # ── TEST 3: Equation in 3D world-space ─────────────────────────
        self._section_banner("Test 3 · 3D Camera Interaction")

        axes = ThreeDAxes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1], z_range=[-2, 2, 1],
            x_length=5, y_length=5, z_length=3,
        )
        axes_labels = axes.get_axis_labels(
            Text("X", font_size=24),
            Text("Y", font_size=24),
            Text("Z", font_size=24),
        )

        # World-space equation (NOT fixed to frame)
        eq_world = MathTex(
            r"\| x - w_i \|", font_size=48, color=YELLOW
        )
        eq_world.move_to(axes.c2p(0, 0, 1.5))

        self.play(FadeIn(axes), FadeIn(axes_labels))
        self.play(FadeIn(eq_world))

        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.15, about="theta")
        self.wait(4)
        self.stop_ambient_camera_rotation()

        self.play(FadeOut(eq_world), FadeOut(axes), FadeOut(axes_labels))
        self.wait(0.3)

        # ── TEST 4: Fixed overlay text (ignores camera) ─────────────────
        self._section_banner("Test 4 · Fixed Frame Overlay")

        # Reset camera to a tilted view so we can confirm the fixed text
        # does NOT rotate with the scene
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)

        axes2 = ThreeDAxes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1], z_range=[-2, 2, 1],
            x_length=5, y_length=5, z_length=3,
        )
        np.random.seed(42)
        bg_dots = VGroup(*[
            Dot3D(
                point=np.random.uniform(-2, 2, size=3),
                radius=0.06,
                color=BLUE_D,
            )
            for _ in range(40)
        ])
        self.play(FadeIn(axes2), FadeIn(bg_dots))

        # Fixed overlay — top-right corner
        eq_fixed = MathTex(
            r"\alpha(t) \rightarrow 0",
            font_size=48,
            color=ORANGE,
        )
        eq_fixed.to_corner(UR).shift(DOWN * 0.2 + LEFT * 0.2)
        self.add_fixed_in_frame_mobjects(eq_fixed)
        self.play(FadeIn(eq_fixed, run_time=0.6))

        # Camera rotates — eq_fixed must stay put
        self.begin_ambient_camera_rotation(rate=0.2, about="theta")
        self.wait(4)
        self.stop_ambient_camera_rotation()

        self.play(FadeOut(eq_fixed), FadeOut(axes2), FadeOut(bg_dots))
        self.wait(0.3)

        # ── TEST 5: Multiple equations with LaggedStart ─────────────────
        self._section_banner("Test 5 · Multiple Equations + LaggedStart")

        # Return to flat view for readability
        self.set_camera_orientation(phi=0 * DEGREES, theta=-90 * DEGREES)

        eqs = VGroup(
            MathTex(r"x", font_size=60, color=WHITE),
            MathTex(r"w_i", font_size=60, color=GREEN_B),
            MathTex(r"\| x - w_i \|", font_size=60, color=YELLOW),
        )
        eqs.arrange(RIGHT, buff=0.9)
        eqs.move_to(ORIGIN)

        # Fix all three to frame so camera tilt won't affect them
        self.add_fixed_in_frame_mobjects(eqs)
        self.play(LaggedStart(*[FadeIn(e) for e in eqs], lag_ratio=0.35))
        self.wait(2.0)

        # Bring them together (demonstrate world-space movement even when fixed)
        self.play(LaggedStart(*[FadeOut(e) for e in eqs], lag_ratio=0.25))
        self.wait(0.5)

        # ── END ──────────────────────────────────────────────────────────
        end_text = Text(
            "LaTeX validation complete",
            font_size=36,
            color=GREY_A,
        )
        self.add_fixed_in_frame_mobjects(end_text)
        self.play(FadeIn(end_text))
        self.wait(1.5)
        self.play(FadeOut(end_text))
