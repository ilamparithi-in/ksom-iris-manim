# Manim CE — Everything Learned (v0.20.1)

## Environment

```
Manim:  0.20.1
Python: 3.13.12  (pyenv env named "manim")
Exec:   /home/ilam/.pyenv/versions/manim/bin/python
```

Activate with `pyenv shell manim` in the terminal **before** running any
`manim` command.  The env must be active; calling the bare `manim` binary
without activating the env fails.

### Available libraries (in this env)
- `sklearn` — `make_blobs`, `MinMaxScaler`
- `numpy`
- All standard Manim CE classes

---

## Run Commands

```bash
# Fast preview (480p, 15 fps) — use during iteration
manim -ql --disable_caching scene.py ClassName

# Full quality (1080p, 60 fps) — use for final render
manim -qh --fps 60 scene.py ClassName

# Open on completion (-p) + low quality
manim -pql scene.py ClassName
```

> **Never pipe manim output through `tail` or any filter.**
> It hides the live frame-progress output the user needs to see.

---

## Verified Class Signatures

### `ThreeDScene`

```python
class MyScene(ThreeDScene):
    def construct(self): ...
```

Key methods:

| Method | Notes |
|---|---|
| `self.set_camera_orientation(phi, theta, gamma, zoom, focal_distance, frame_center)` | Instant, no animation |
| `self.move_camera(phi, theta, zoom, run_time, added_anims=[])` | Animated; pass extra anims via `added_anims` |
| `self.begin_ambient_camera_rotation(rate=0.02, about='theta')` | `about='theta'` = horizontal spin; `'phi'` = vertical tilt |
| `self.stop_ambient_camera_rotation(about='theta')` | Must match the `about` used to start |
| `self.add_fixed_in_frame_mobjects(*mobs)` | Pins 2D overlays (text/equations) to HUD — they don't move with camera |

### `Dot3D`

```python
Dot3D(point=ORIGIN, radius=0.08, color=WHITE, resolution=(8,8))
```

- `dot.animate.move_to(target_np_array)` — smooth translation
- `dot.animate.set_color(COLOR)` — smooth color change
- `dot.animate.scale(factor)` — pulse / highlight
- `dot.animate.set_opacity(value)` — fade in/out via animate

### `ThreeDAxes`

```python
ThreeDAxes(
    x_range=[-3, 3, 1], y_range=[-3, 3, 1], z_range=[-3, 3, 1],
    x_length=6, y_length=6, z_length=6,
)
```

- `.c2p(x, y, z)` → converts data coordinates to world-space `np.ndarray`
- Shift entire axes with `.shift(RIGHT * offset)` to move the origin

### `VGroup`

```python
group = VGroup(*list_of_vmobjects)
```

- Iterate: `for dot in group`
- Index: `group[i]`
- Modify: `group.add(mob)`, `group.remove(mob)`
- Arrange: `group.arrange(RIGHT, buff=0.9)` (works in 2D context)

### `MathTex`

```python
MathTex(r"x \in \mathbb{R}^n", font_size=56, color=WHITE)
```

- Position: `.to_corner(DR)`, `.move_to(point)`, `.to_edge(UP)`
- Works in both 2D (default camera) and 3D world-space
- For HUD placement, use `add_fixed_in_frame_mobjects` (see below)

### `Text`

```python
Text("Some string", font_size=36, color=GREY_A)
```

Same positioning API as `MathTex`.

### `NumberLine`

```python
NumberLine(x_range=[-3, 3, 1], length=7.0, color=GREY_B,
           tick_size=0.08, include_numbers=False)
```

- Can be `Transform`-ed into `Axes` for the 1D → 2D expansion effect

### `Axes` (2D)

```python
Axes(x_range=[-3,3,1], y_range=[-3,3,1], x_length=6, y_length=6,
     tips=False, axis_config={"color": GREY_C, "stroke_width": 1.5})
```

### `Line` / `Line3D`

```python
Line(start_np, end_np, color=GREY_B, stroke_width=1.5)
```

- For connection lines that must update every frame as dots move, use
  `always_redraw(lambda: Line(a.get_center(), b.get_center(), ...))`
- For static lines (dots don't move after creation), use plain `Line`
  and animate with `.animate.set_stroke(color=..., width=...)`

---

## Animation Classes

### `FadeIn` / `FadeOut`

```python
self.play(FadeIn(mob, run_time=1.0))
self.play(FadeOut(mob))
```

### `LaggedStart`

```python
self.play(
    LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.012),
    run_time=3.0,
)
```

- `lag_ratio=0.012` for 300 dots → smooth wave effect
- `lag_ratio=0.05`  for ~50 dots
- `lag_ratio=0.18`  for 5–7 dots (clear stagger)
- `lag_ratio=0.35`  for 3 equations (very visible delay)

### `AnimationGroup`

```python
self.play(
    AnimationGroup(*[dot.animate.move_to(p) for ...], lag_ratio=0.0),
    run_time=2,
)
```

`lag_ratio=0.0` → all start simultaneously.

### `Transform`

```python
self.play(Transform(source_mob, target_mob), run_time=1.2)
```

- After `Transform(A, B)`, the object in the scene is **still called `A`**
  (it just looks like B). Do NOT reference `B` afterwards.
- To hand off to a third: `self.play(Transform(A, C))`

### `ReplacementTransform`

```python
self.play(ReplacementTransform(group_a, group_b))
```

- Unlike `Transform`, `group_a` is removed and `group_b` takes its place.

### `Create`

```python
self.play(Create(line))
self.play(LaggedStart(*[Create(connections[i]) for i in range(N)], lag_ratio=0.04))
```

- Use for lines, grids, shapes — draws them progressively.

### `always_redraw`

```python
conn = always_redraw(
    lambda a=dot_a, b=dot_b: Line(a.get_center(), b.get_center(),
                                   color=GREY_B, stroke_width=1.0)
)
self.add(conn)   # add to scene — redraws every frame automatically
```

- Correct for connection lines that must track moving dots.
- Do NOT use `self.play(Create(conn))` — add with `self.add(conn)`.
- When using `LaggedStart(Create(...))` on connections, build static
  copies first, then swap to `always_redraw` versions afterwards
  (mixing `Create` and `always_redraw` on the same object causes issues).

---

## Critical Bug: Fixed-in-Frame Labels Flashing Early

### Problem
`add_fixed_in_frame_mobjects(mob)` calls `self.add(mob)` **immediately**,
making the object visible before your `FadeIn` animation runs.

### Fix — Always do this:

```python
lbl = Text("Some label", font_size=32)
lbl.set_opacity(0)                          # invisible before FadeIn
lbl.to_corner(UL)
self.add_fixed_in_frame_mobjects(lbl)       # pins to HUD (adds to scene)
self.play(FadeIn(lbl))                      # now animates from 0 → 1
```

This applies to **every** `Text`, `MathTex`, or `VGroup` passed to
`add_fixed_in_frame_mobjects`.

---

## Fixed-in-Frame Pattern (complete)

```python
# Create the label
overlay = MathTex(r"x \in \mathbb{R}^n", font_size=36, color=GREY_A)
overlay.set_opacity(0)          # REQUIRED — prevent early flash
overlay.to_corner(DR)
self.add_fixed_in_frame_mobjects(overlay)
self.play(FadeIn(overlay))

# To remove cleanly:
self.play(FadeOut(overlay))
self.remove(overlay)            # also remove from scene graph
```

---

## Data Generation

```python
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

pts, labels = make_blobs(
    n_samples=300, n_features=3, centers=4,
    cluster_std=0.65, random_state=42,
)
scaler = MinMaxScaler(feature_range=(-2.2, 2.2))
pts = scaler.fit_transform(pts)

# Shift cloud to right side of screen
pts[:, 0] += DATA_CX
```

---

## Colour Palette (verified Manim names)

```python
# Blues
BLUE, BLUE_A, BLUE_B, BLUE_C, BLUE_D, BLUE_E

# Greens
GREEN, GREEN_A, GREEN_B, GREEN_C, GREEN_D, GREEN_E

# Reds
RED, RED_A, RED_B

# Yellows
YELLOW, YELLOW_A, YELLOW_B

# Teals / Purples / Oranges
TEAL, TEAL_B, PURPLE_B, ORANGE, PINK

# Greys
WHITE, GREY_A, GREY_B, GREY_C, GREY_D, BLACK

# Named special
ORIGIN  # np.array([0,0,0])
```

### Colors that do NOT exist in Manim 0.20.1
- `CYAN` — use `"#00FFFF"` instead
- Any CSS color names not listed above — use hex strings

---

## Camera Cheat Sheet

```python
# Instant camera placement (no animation)
self.set_camera_orientation(phi=75*DEGREES, theta=-45*DEGREES, zoom=1.0)

# Animated camera move (+ optional co-animations)
self.move_camera(
    phi=50*DEGREES, theta=-90*DEGREES, zoom=1.2, run_time=2,
    added_anims=[FadeIn(axes), dot.animate.move_to(p)],
)

# Slow horizontal spin (use rate=0.05 for subtle, 0.3 for obvious)
self.begin_ambient_camera_rotation(rate=0.05, about='theta')
self.wait(6)
self.stop_ambient_camera_rotation()   # default about='theta'

# Zoom in to specific region
self.move_camera(zoom=1.45, run_time=1.5)

# Return to overview
self.move_camera(zoom=0.88, run_time=2.0)
```

phi=0, theta=-90*DEGREES → flat top-down (good for 1D / 2D intros)
phi=75*DEGREES, theta=-45*DEGREES → standard 3D perspective

---

## Grid Construction Pattern

```python
GRID_ROWS, GRID_COLS = 8, 8
GRID_SPACING = 0.60
GRID_CX = -4.2  # center-left of scene

def grid_pos(r, c):
    x = GRID_CX + (c - (GRID_COLS-1)/2) * GRID_SPACING
    y = (r - (GRID_ROWS-1)/2) * GRID_SPACING
    return np.array([x, y, 0.0])

all_pos = [grid_pos(r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS)]

grid_nodes = VGroup(*[Dot3D(point=p, radius=0.06, color=BLUE_D) for p in all_pos])

h_lines = VGroup(*[
    Line(all_pos[r*GRID_COLS+c], all_pos[r*GRID_COLS+c+1],
         color=BLUE_E, stroke_width=1.0)
    for r in range(GRID_ROWS) for c in range(GRID_COLS-1)
])
v_lines = VGroup(*[
    Line(all_pos[r*GRID_COLS+c], all_pos[(r+1)*GRID_COLS+c],
         color=BLUE_E, stroke_width=1.0)
    for r in range(GRID_ROWS-1) for c in range(GRID_COLS)
])

self.play(Create(h_lines), Create(v_lines), run_time=1.8)
self.play(LaggedStart(*[FadeIn(d) for d in grid_nodes], lag_ratio=0.012), run_time=1.5)
```

---

## always_redraw Connection Lines (correct pattern)

```python
def _conn_line(gd: Dot3D, wd: Dot3D) -> VMobject:
    return always_redraw(
        lambda g=gd, w=wd: Line(
            g.get_center(), w.get_center(),
            color=GREY_B, stroke_width=1.2,
        )
    )

connections = VGroup(*[_conn_line(grid_nodes[i], w_dots[i]) for i in range(N)])

# Add statically (not via Create — always_redraw redraws every frame itself)
self.add(connections)

# Then animate the dots — connections follow automatically
self.play(AnimationGroup(*[w_dots[i].animate.move_to(new_pos[i]) for i in range(N)]))
```

---

## Smooth Color Interpolation

```python
def _line_color(d, d_min, d_max):
    """White = close, grey = far."""
    t = (d - d_min) / (d_max - d_min + 1e-9)
    return interpolate_color(WHITE, GREY_D, t)

def _line_width(d, d_min, d_max):
    t = (d - d_min) / (d_max - d_min + 1e-9)
    return max(0.8, 3.0 * (1.0 - t))
```

---

## Local Labels Near 3D Objects (world-space, not HUD)

```python
label = Text("winner", font_size=22, color=GREEN_B)
label.move_to(dot.get_center() + np.array([0.35, 0.35, 0.0]))
self.play(FadeIn(label))
```

Use small font sizes (18–26) for labels near 3D objects.
Use `add_fixed_in_frame_mobjects` only when the label must stay on-screen
regardless of camera rotation (e.g. corner equations, section titles).

---

## MathTex Proven Symbols

All of these compile without error:

```python
MathTex(r"x \in \mathbb{R}^n")
MathTex(r"\| x - w_i \|")
MathTex(r"w_i(t{+}1) = w_i(t) + \alpha(t)\bigl(x - w_i(t)\bigr)")
MathTex(r"\alpha(t) \rightarrow 0")
MathTex(r"c = \arg\min_i \| x - w_i \|")
MathTex(r"c(x) = \arg\min_i \| x - w_i \|")
MathTex(r"w_i \in \mathbb{R}^n")
```

Avoid `+` bare inside subscripts — wrap in `{+}` to prevent parse errors:
`t{+}1` not `t+1` inside MathTex subscript context.

---

## MathTex Transition Patterns

### FadeOut / FadeIn (cleanest, no residue)

```python
eq1 = MathTex(r"x \in \mathbb{R}^n", font_size=48)
eq1.set_opacity(0); eq1.to_corner(UR)
self.add_fixed_in_frame_mobjects(eq1)
self.play(FadeIn(eq1))

eq2 = MathTex(r"\| x - w_i \|", font_size=48)
eq2.set_opacity(0); eq2.to_corner(UR)
self.add_fixed_in_frame_mobjects(eq2)
self.play(FadeOut(eq1), FadeIn(eq2))
self.remove(eq1)
```

### Transform (morphing letters — same position required)

```python
self.play(Transform(eq1, eq2), run_time=1.2)
# eq1 is now visually eq2 — do NOT reference eq2
self.play(Transform(eq1, eq3), run_time=1.4)
```

---

## Dim / Highlight Pattern

```python
# Dim everything to focus on one object
self.play(
    AnimationGroup(*[d.animate.set_opacity(0.12) for d in all_dots
                     if d is not highlight_dot]),
    run_time=1.0,
)

# Restore
self.play(
    AnimationGroup(*[d.animate.set_opacity(1.0) for d in all_dots]),
    run_time=0.8,
)
```

---

## Performance Notes (300 Dot3D at 1080p/60fps)

- 300 × `Dot3D(radius=0.05)` renders without dropped frames at 1080p/60.
- `LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.012, run_time=3.0)` — smooth.
- Ambient camera rotation on 300-dot clouds is smooth.
- `always_redraw` connections for 5–16 nodes have zero performance impact.
- VGroup opacity (`blob_dots.animate.set_opacity(0.4)`) works correctly on 300 dots.

---

## File Structure

```
scene1_high_dim_data.py     # Scene 1: 1D→2D→3D, curse of dimensionality
scene2_grid_dual_identity.py # Scene 2: SOM grid ↔ data space dual identity
scene3_bmu.py               # Scene 3: Best Matching Unit close-up demo
_test_manim/
  test1_basic3d.py          # ThreeDAxes + Dot3D + ambient rotation
  test2_grouped_dots.py     # 50 Dot3D in VGroup + LaggedStart FadeIn
  test3_transform.py        # scatter → cluster transform
  test4_camera.py           # camera controls
  test5_large_dataset.py    # 300 Dot3D at 1080p/60fps stress test
  test6_equations.py        # MathTex in ThreeDScene, fixed overlays
  test7_dual_space.py       # 2D grid + 3D cloud + always_redraw connections
  test8_transitions.py      # ReplacementTransform, FadeTransform, AnimationGroup
  test_latex.py             # Full LaTeX validation (5 sub-tests)
```
