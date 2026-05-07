#!/usr/bin/env python3
import argparse
import multiprocessing
import re
import subprocess
import sys
import time
from pathlib import Path

QUALITY_DIR = {
    "l": "480p15",
    "m": "720p30",
    "h": "1080p60",
    "k": "2160p60",
}

P_CORES   = list(range(8))
MEDIA_DIR = Path("media2/videos")
LOGS_DIR  = Path("logs")


def find_py_files(root: Path) -> list[Path]:
    self_path = Path(__file__).resolve()
    return [
        p for p in sorted(root.glob("*.py"))
        if p.resolve() != self_path
    ]


def detect_scenes(py_file: Path) -> list[str]:
    try:
        text = py_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    if not re.search(r"^(?:from manim|import manim)", text, re.MULTILINE):
        return []
    return re.findall(r"^class\s+(\w+)\s*\([^)]*Scene[^)]*\)", text, re.MULTILINE)


def output_path(py_file: Path, scene: str, quality_dir: str) -> Path:
    return MEDIA_DIR / py_file.stem / quality_dir / f"{scene}.mp4"


def render_job(args: tuple) -> tuple:
    py_file_str, scene, quality_flag, quality_dir, core, dry_run = args
    py_file = Path(py_file_str)
    out     = output_path(py_file, scene, quality_dir)
    label   = f"{py_file_str}::{scene}"

    if out.exists():
        print(f"[SKIP]  {label} (already exists)", flush=True)
        return ("skip", py_file_str, scene, 0.0)

    if dry_run:
        print(f"[DRY]   {label} → core {core}", flush=True)
        return ("dry", py_file_str, scene, 0.0)

    print(f"[START] {label} (core {core})", flush=True)

    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / f"{py_file.stem}__{scene}.log"
    cmd      = ["taskset", "-c", str(core), "manim", quality_flag,
                "--disable_caching", "--media_dir", str(MEDIA_DIR.parent), py_file_str, scene]
    t0       = time.monotonic()

    try:
        with open(log_path, "w") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
        elapsed = time.monotonic() - t0

        if proc.returncode == 0 and out.exists():
            print(f"[DONE]  {label} (time: {elapsed:.1f}s)", flush=True)
            return ("ok", py_file_str, scene, elapsed)

        print(f"[FAIL]  {label}", flush=True)
        return ("fail", py_file_str, scene, elapsed)

    except Exception:
        elapsed = time.monotonic() - t0
        print(f"[FAIL]  {label}", flush=True)
        return ("fail", py_file_str, scene, elapsed)


def write_list_txt(scenes: list[tuple[str, str]], quality_dir: str) -> Path:
    entries = sorted(scenes, key=lambda x: (x[0], x[1]))
    lines   = [
        f"file '{output_path(Path(f), s, quality_dir).resolve()}'"
        for f, s in entries
    ]
    list_txt = Path("list.txt")
    list_txt.write_text("\n".join(lines) + "\n")
    return list_txt


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel Manim renderer. Activate the manim pyenv shell before running.\n"
            "Scans all .py files for Scene subclasses and renders them in parallel,\n"
            "pinned to P-cores (0-7) via taskset."
        ),
        epilog=(
            "Examples:\n"
            "  python render_all.py h\n"
            "  python render_all.py h --workers 4\n"
            "  python render_all.py h --dry-run"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "quality",
        choices=list(QUALITY_DIR),
        metavar="quality",
        help="Manim quality flag value: l | m | h | k  (passed as -ql / -qm / -qh / -qk)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=len(P_CORES),
        metavar="N",
        help=f"Parallel worker count (default: {len(P_CORES)})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be rendered without executing",
    )
    args = parser.parse_args()

    quality_flag = f"-q{args.quality}"
    quality_dir  = QUALITY_DIR[args.quality]
    root = Path(".")

    all_scenes: list[tuple[str, str]] = []
    for py_file in find_py_files(root):
        for scene in detect_scenes(py_file):
            all_scenes.append((str(py_file), scene))

    if not all_scenes:
        print("No scenes found.", file=sys.stderr)
        sys.exit(1)

    jobs = [
        (f, s, quality_flag, quality_dir, P_CORES[i % len(P_CORES)], args.dry_run)
        for i, (f, s) in enumerate(all_scenes)
    ]

    t_wall = time.monotonic()
    with multiprocessing.Pool(processes=args.workers) as pool:
        results = pool.map(render_job, jobs)
    t_wall = time.monotonic() - t_wall

    counts: dict[str, int] = {}
    for status, *_ in results:
        counts[status] = counts.get(status, 0) + 1

    print()
    print(f"Total:     {len(results)}")
    print(f"Rendered:  {counts.get('ok',   0)}")
    print(f"Skipped:   {counts.get('skip', 0)}")
    print(f"Failed:    {counts.get('fail', 0)}")
    print(f"Wall time: {t_wall:.1f}s")

    if args.dry_run:
        list_txt = write_list_txt([(f, s) for f, s, *_ in jobs], quality_dir)
        print(f"\n{list_txt}  → {len(jobs)} entries (dry run)")
    else:
        available = [
            (f, s)
            for status, f, s, _ in results
            if status in ("ok", "skip")
        ]
        if available:
            list_txt = write_list_txt(available, quality_dir)
            print(f"\n{list_txt}  → {len(available)} entries")

    print("\nffmpeg -f concat -safe 0 -i list.txt -c copy output.mp4")


if __name__ == "__main__":
    main()
