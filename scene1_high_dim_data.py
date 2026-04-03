"""
Scene 1: The Problem — High-Dimensional Data

A ~30-second ThreeDScene progressing from 1D clarity to an overwhelming
3D point cloud, visually embodying the curse of dimensionality.

Usage (low-quality preview):
    manim -ql --disable_caching scene1_high_dim_data.py Scene1HighDimData

Usage (full quality):
    manim -qh --fps 60 scene1_high_dim_data.py Scene1HighDimData
"""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from manim import *

CLUSTER_PALETTE = [BLUE_B, RED_B, GREEN_B, YELLOW_B]


class Scene1HighDimData(ThreeDScene):
    """Visual journey: 1D clarity → overwhelming high-dimensional data."""

    # ─── helpers ────────────────────────────────────────────────────────

    def _dim_label(self, n_str: str) -> MathTex:
        """Build  x ∈ ℝ^n  label anchored to the bottom-right corner."""
        lbl = MathTex(
            rf"x \in \mathbb{{R}}^{{{n_str}}}",
            font_size=36,
            color=GREY_A,
        )
        lbl.to_corner(DR)
        return lbl

    def _swap_dim(self, old_lbl: MathTex, new_n: str, rt: float = 0.55) -> MathTex:
        """Cross-fade the dimension overlay from old_lbl to a new value of n."""
        new_lbl = self._dim_label(new_n)
        self.add_fixed_in_frame_mobjects(new_lbl)
        self.play(FadeOut(old_lbl), FadeIn(new_lbl), run_time=rt)
        self.remove(old_lbl)
        return new_lbl

    # ─── construct ──────────────────────────────────────────────────────

    def construct(self):
        np.random.seed(42)

        # ═══════════════════════════════════════════════════════════════
        # PART 1 — SIMPLE 1D
        # Flat overhead camera — clean, calm, symmetric.
        # ═══════════════════════════════════════════════════════════════
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES, zoom=1.2)

        nline = NumberLine(
            x_range=[-3, 3, 1],
            length=7.0,
            color=GREY_B,
            tick_size=0.08,
            include_numbers=False,
        )

        dot_xs  = np.linspace(-2.5, 2.5, 6)
        dots_1d = VGroup(*[
            Dot3D(point=np.array([x, 0.0, 0.0]), radius=0.09, color=BLUE_B)
            for x in dot_xs
        ])

        dim_lbl = self._dim_label("1")
        self.add_fixed_in_frame_mobjects(dim_lbl)

        self.play(FadeIn(nline), FadeIn(dim_lbl), run_time=0.9)
        self.play(
            LaggedStart(*[FadeIn(d) for d in dots_1d], lag_ratio=0.18),
            run_time=1.5,
        )
        self.wait(1.5)                                    # PAUSE — clarity

        # ═══════════════════════════════════════════════════════════════
        # PART 2 — 2D EXPANSION
        # NumberLine morphs into 2D Axes; dots spread into loose clusters.
        # ═══════════════════════════════════════════════════════════════
        axes_2d = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=6.0,
            y_length=6.0,
            tips=False,
            axis_config={"color": GREY_C, "stroke_width": 1.5},
        )

        targets_2d = [
            np.array([-1.9,  1.7, 0.0]),
            np.array([ 1.8,  1.6, 0.0]),
            np.array([-1.7, -1.8, 0.0]),
            np.array([ 1.6, -1.7, 0.0]),
            np.array([-0.2,  0.3, 0.0]),
            np.array([ 0.9,  0.8, 0.0]),
        ]

        dim_lbl = self._swap_dim(dim_lbl, "2")
        self.play(Transform(nline, axes_2d), run_time=1.4)
        self.play(
            AnimationGroup(
                *[dots_1d[i].animate.move_to(targets_2d[i]) for i in range(6)],
                lag_ratio=0.10,
            ),
            run_time=1.8,
        )
        self.wait(1.0)                                    # PAUSE — 2D structure

        # ═══════════════════════════════════════════════════════════════
        # PART 3 — 3D TRANSITION
        # 2D plane fades out; camera tilts; ThreeDAxes and depth appear.
        # Camera tilt + axis swap + dot z-gain are all simultaneous.
        # ═══════════════════════════════════════════════════════════════
        axes_3d = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-3, 3, 1],
            x_length=6.0,
            y_length=6.0,
            z_length=6.0,
        )

        z_vals     = [1.4, -1.3, 1.2, -1.5, 0.6, -0.8]
        targets_3d = [
            targets_2d[i] + np.array([0.0, 0.0, z_vals[i]])
            for i in range(6)
        ]

        dim_lbl = self._swap_dim(dim_lbl, "3")

        # Dissolve 2D axes first, then lift into 3D in a single camera move.
        self.play(FadeOut(nline), run_time=0.7)
        self.remove(nline)

        self.move_camera(
            phi=75 * DEGREES,
            theta=-45 * DEGREES,
            zoom=0.85,
            run_time=2.5,
            added_anims=[
                FadeIn(axes_3d),
                *[dots_1d[i].animate.move_to(targets_3d[i]) for i in range(6)],
            ],
        )
        self.begin_ambient_camera_rotation(rate=0.10, about="theta")
        self.wait(1.8)                                    # PAUSE — depth revealed

        # ═══════════════════════════════════════════════════════════════
        # PART 4 — REAL DATA (BLOBS)
        # Replace sparse placeholder dots with 300-point clustered cloud.
        # ═══════════════════════════════════════════════════════════════
        pts, labels = make_blobs(
            n_samples=300, n_features=3, centers=4,
            cluster_std=0.65, random_state=42,
        )
        scaler = MinMaxScaler(feature_range=(-2.2, 2.2))
        pts    = scaler.fit_transform(pts)

        blob_dots = VGroup(*[
            Dot3D(
                point=pts[i],
                radius=0.05,
                color=CLUSTER_PALETTE[int(labels[i])],
            )
            for i in range(len(pts))
        ])

        # Quick fade-out of sparse dots; dense cloud materialises point by point.
        self.play(FadeOut(dots_1d), run_time=0.7)
        self.remove(dots_1d)
        self.play(
            LaggedStart(*[FadeIn(d) for d in blob_dots], lag_ratio=0.012),
            run_time=3.0,
        )
        self.wait(1.0)

        # ═══════════════════════════════════════════════════════════════
        # PART 5 — HIGH DIMENSION HINT
        # n counter climbs in overlay; floating banner; subtle jitter.
        # ═══════════════════════════════════════════════════════════════
        dim_lbl = self._swap_dim(dim_lbl, "10",  rt=0.5)
        self.wait(0.35)
        dim_lbl = self._swap_dim(dim_lbl, "100", rt=0.5)

        banner = Text("4D  →  10D  →  100D", font_size=40, color=YELLOW_B)
        banner.to_edge(UP, buff=0.35)
        self.add_fixed_in_frame_mobjects(banner)
        self.play(FadeIn(banner), run_time=0.8)
        self.wait(0.7)

        # Jitter 80 random points — "this is just a projection" effect.
        rng        = np.random.default_rng(seed=77)
        jitter_idx = rng.choice(len(pts), size=80, replace=False)
        self.play(
            AnimationGroup(
                *[
                    blob_dots[int(k)].animate.move_to(
                        pts[int(k)] + rng.uniform(-0.12, 0.12, 3)
                    )
                    for k in jitter_idx
                ],
                lag_ratio=0.0,
            ),
            run_time=0.55,
        )
        self.wait(0.6)

        # ═══════════════════════════════════════════════════════════════
        # PART 6 — LOSS OF INTUITION
        # Camera spins faster; cloud dims; closing text fades in.
        # ═══════════════════════════════════════════════════════════════
        self.stop_ambient_camera_rotation()
        self.begin_ambient_camera_rotation(rate=0.30, about="theta")

        self.play(
            blob_dots.animate.set_opacity(0.50),
            run_time=1.8,
        )

        conclusion = Text(
            "Structure exists…  but is hidden.",
            font_size=36,
            color=GREY_A,
        )
        conclusion.to_edge(DOWN, buff=0.6)
        self.add_fixed_in_frame_mobjects(conclusion)
        self.play(FadeIn(conclusion), run_time=1.5)
        self.wait(3.5)                                    # PAUSE — contemplation

        # ═══════════════════════════════════════════════════════════════
        # FINAL FADE
        # ═══════════════════════════════════════════════════════════════
        self.stop_ambient_camera_rotation()
        self.play(
            FadeOut(axes_3d),
            FadeOut(blob_dots),
            FadeOut(dim_lbl),
            FadeOut(banner),
            FadeOut(conclusion),
            run_time=1.5,
        )
        self.wait(0.5)
