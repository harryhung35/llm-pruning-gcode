#!/usr/bin/env python3
"""
Overlay DXF geometry and G-code toolpaths into PNG images.

Usage:
    python overlay_dxf_gcode.py

This script assumes:
- DXF files and G-code files are in different folders.
- Files are matched by filename stem, e.g. test102.dxf <-> test102.gcode.
- DXF is drawn in red.
- G-code is drawn in blue.
- Output PNG adds a centered filename title and a legend in the upper-right corner.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

try:
    import ezdxf
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: ezdxf\n"
        "Please install required packages first:\n"
        "  pip install ezdxf matplotlib numpy"
    ) from exc

# =========================
# Change folders here when running this file directly
# =========================
DXF_DIR = Path("./inference_input")
GCODE_DIR = Path("./gcode_out_test")
OUT_DIR = Path("./overlay_png")

# =========================
# Drawing / export settings
# =========================
ARC_STEP_DEG = 2.0
CURVE_FLATTEN_DIST = 0.05
DPI = 300
FIGSIZE = (8.0, 8.0)
PAD_RATIO = 0.03
SHOW_G0 = False
SHOW_AXES = False
TITLE_FONTSIZE = 14
LEGEND_FONTSIZE = 11
LEGEND_LOCATION = "upper left"
LEGEND_BBOX_TO_ANCHOR = (1.01, 1.0)

Point = Tuple[float, float]
Polyline = np.ndarray

GCODE_EXTS = {".gcode", ".nc", ".ngc", ".tap", ".txt"}
DXF_EXTS = {".dxf"}
TOKEN_RE = re.compile(r"([A-Za-z])\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
PAREN_COMMENT_RE = re.compile(r"\([^)]*\)")


def list_files(folder: Path, exts: Sequence[str]) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    extset = {e.lower() for e in exts}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in extset:
            result[path.stem.lower()] = path
    return result



def strip_gcode_comments(line: str) -> str:
    line = PAREN_COMMENT_RE.sub("", line)
    if ";" in line:
        line = line.split(";", 1)[0]
    return line.strip()



def tokenize_gcode(line: str) -> List[Tuple[str, float]]:
    return [(m.group(1).upper(), float(m.group(2))) for m in TOKEN_RE.finditer(line)]



def compute_arc_delta(a0: float, a1: float, cw: bool) -> float:
    delta = a1 - a0
    if cw:
        if delta >= 0:
            delta -= 2 * math.pi
    else:
        if delta <= 0:
            delta += 2 * math.pi
    return delta



def build_arc_points(
    center: Point,
    radius: float,
    start_angle: float,
    end_angle: float,
    cw: bool,
    arc_step_deg: float,
) -> Polyline:
    delta = compute_arc_delta(start_angle, end_angle, cw)
    step = math.radians(max(0.1, arc_step_deg))
    n = max(2, int(math.ceil(abs(delta) / step)) + 1)
    theta = np.linspace(start_angle, start_angle + delta, n)
    cx, cy = center
    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)
    return np.column_stack([x, y])



def arc_center_from_r(
    start: Point,
    end: Point,
    radius_value: float,
    cw: bool,
) -> Tuple[Point, float, float, float]:
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    chord = math.hypot(dx, dy)
    radius = abs(radius_value)

    if chord == 0:
        raise ValueError("Arc start and end are identical.")
    if chord > 2 * radius:
        raise ValueError("R arc is geometrically impossible.")

    mx = (sx + ex) / 2.0
    my = (sy + ey) / 2.0
    h = math.sqrt(max(radius * radius - (chord / 2.0) ** 2, 0.0))

    ux = -dy / chord
    uy = dx / chord
    candidates = [
        (mx + h * ux, my + h * uy),
        (mx - h * ux, my - h * uy),
    ]

    scored: List[Tuple[bool, float, Point, float, float]] = []
    want_major = radius_value < 0
    for center in candidates:
        cx, cy = center
        a0 = math.atan2(sy - cy, sx - cx)
        a1 = math.atan2(ey - cy, ex - cx)
        delta = compute_arc_delta(a0, a1, cw)
        is_major = abs(delta) > math.pi + 1e-9
        score = abs(abs(delta) - math.pi)
        scored.append((is_major == want_major, score, center, a0, a1))

    scored.sort(key=lambda item: (not item[0], item[1]))
    _, _, center, a0, a1 = scored[0]
    return center, radius, a0, a1



def parse_gcode_file(path: Path, arc_step_deg: float, show_g0: bool) -> Tuple[List[Polyline], List[Polyline]]:
    state = {
        "x": 0.0,
        "y": 0.0,
        "motion": 1,
        "absolute": True,
        "unit_scale": 1.0,
    }

    feed_paths: List[Polyline] = []
    rapid_paths: List[Polyline] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = strip_gcode_comments(raw_line)
            if not line:
                continue

            tokens = tokenize_gcode(line)
            if not tokens:
                continue

            motion_in_line: Optional[int] = None
            vals: Dict[str, float] = {}

            for letter, value in tokens:
                if letter == "G":
                    g_rounded = round(value, 3)
                    if g_rounded in (0, 1, 2, 3):
                        motion_in_line = int(g_rounded)
                    elif g_rounded == 20:
                        state["unit_scale"] = 25.4
                    elif g_rounded == 21:
                        state["unit_scale"] = 1.0
                    elif g_rounded == 90:
                        state["absolute"] = True
                    elif g_rounded == 91:
                        state["absolute"] = False
                elif letter in {"X", "Y", "I", "J", "R"}:
                    vals[letter] = value * state["unit_scale"]

            if motion_in_line is not None:
                state["motion"] = motion_in_line

            motion = state["motion"]
            if "X" not in vals and "Y" not in vals:
                continue

            x0 = float(state["x"])
            y0 = float(state["y"])

            if state["absolute"]:
                x1 = vals.get("X", x0)
                y1 = vals.get("Y", y0)
            else:
                x1 = x0 + vals.get("X", 0.0)
                y1 = y0 + vals.get("Y", 0.0)

            if motion in (0, 1):
                if x0 != x1 or y0 != y1:
                    segment = np.array([[x0, y0], [x1, y1]], dtype=float)
                    if motion == 0:
                        rapid_paths.append(segment)
                    else:
                        feed_paths.append(segment)
            elif motion in (2, 3):
                if x0 != x1 or y0 != y1:
                    cw = motion == 2
                    try:
                        if "I" in vals or "J" in vals:
                            cx = x0 + vals.get("I", 0.0)
                            cy = y0 + vals.get("J", 0.0)
                            radius = math.hypot(x0 - cx, y0 - cy)
                            a0 = math.atan2(y0 - cy, x0 - cx)
                            a1 = math.atan2(y1 - cy, x1 - cx)
                            arc = build_arc_points((cx, cy), radius, a0, a1, cw, arc_step_deg)
                        elif "R" in vals:
                            center, radius, a0, a1 = arc_center_from_r((x0, y0), (x1, y1), vals["R"], cw)
                            arc = build_arc_points(center, radius, a0, a1, cw, arc_step_deg)
                        else:
                            arc = np.array([[x0, y0], [x1, y1]], dtype=float)
                    except Exception:
                        arc = np.array([[x0, y0], [x1, y1]], dtype=float)
                    feed_paths.append(arc)

            state["x"] = x1
            state["y"] = y1

    if not show_g0:
        rapid_paths = []
    return feed_paths, rapid_paths



def approx_arc_dxf(entity, arc_step_deg: float) -> Polyline:
    center = entity.dxf.center
    radius = float(entity.dxf.radius)
    start_deg = float(entity.dxf.start_angle)
    end_deg = float(entity.dxf.end_angle)
    start = math.radians(start_deg)
    end = math.radians(end_deg)
    if end <= start:
        end += 2 * math.pi
    delta = end - start
    step = math.radians(max(0.1, arc_step_deg))
    n = max(2, int(math.ceil(delta / step)) + 1)
    theta = np.linspace(start, end, n)
    x = center.x + radius * np.cos(theta)
    y = center.y + radius * np.sin(theta)
    return np.column_stack([x, y])



def approx_circle_dxf(entity, arc_step_deg: float) -> Polyline:
    center = entity.dxf.center
    radius = float(entity.dxf.radius)
    step = math.radians(max(0.1, arc_step_deg))
    n = max(32, int(math.ceil((2 * math.pi) / step)) + 1)
    theta = np.linspace(0.0, 2 * math.pi, n)
    x = center.x + radius * np.cos(theta)
    y = center.y + radius * np.sin(theta)
    return np.column_stack([x, y])



def as_xy_array(vertices: Iterable) -> Optional[Polyline]:
    pts: List[Point] = []
    for v in vertices:
        if hasattr(v, "x") and hasattr(v, "y"):
            pts.append((float(v.x), float(v.y)))
        else:
            pts.append((float(v[0]), float(v[1])))
    if len(pts) < 2:
        return None
    return np.array(pts, dtype=float)



def dxf_entity_to_paths(entity, arc_step_deg: float, curve_flatten_dist: float) -> List[Polyline]:
    dxftype = entity.dxftype()

    if dxftype == "LINE":
        start = entity.dxf.start
        end = entity.dxf.end
        return [np.array([[start.x, start.y], [end.x, end.y]], dtype=float)]

    if dxftype == "ARC":
        return [approx_arc_dxf(entity, arc_step_deg)]

    if dxftype == "CIRCLE":
        return [approx_circle_dxf(entity, arc_step_deg)]

    if dxftype in {"LWPOLYLINE", "POLYLINE"}:
        if hasattr(entity, "virtual_entities"):
            paths: List[Polyline] = []
            try:
                for sub_entity in entity.virtual_entities():
                    paths.extend(dxf_entity_to_paths(sub_entity, arc_step_deg, curve_flatten_dist))
            except Exception:
                paths = []
            if paths:
                return paths

        try:
            if dxftype == "LWPOLYLINE":
                arr = as_xy_array(entity.get_points("xy"))
            else:
                arr = as_xy_array([v.dxf.location for v in entity.vertices])
            if arr is not None:
                return [arr]
        except Exception:
            pass
        return []

    if dxftype in {"ELLIPSE", "SPLINE"}:
        if hasattr(entity, "flattening"):
            try:
                arr = as_xy_array(entity.flattening(distance=curve_flatten_dist))
                if arr is not None:
                    return [arr]
            except Exception:
                pass
        return []

    if dxftype == "INSERT" and hasattr(entity, "virtual_entities"):
        paths: List[Polyline] = []
        try:
            for sub_entity in entity.virtual_entities():
                paths.extend(dxf_entity_to_paths(sub_entity, arc_step_deg, curve_flatten_dist))
        except Exception:
            pass
        return paths

    if hasattr(entity, "virtual_entities"):
        paths: List[Polyline] = []
        try:
            for sub_entity in entity.virtual_entities():
                paths.extend(dxf_entity_to_paths(sub_entity, arc_step_deg, curve_flatten_dist))
        except Exception:
            pass
        return paths

    return []



def parse_dxf_file(path: Path, arc_step_deg: float, curve_flatten_dist: float) -> List[Polyline]:
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    paths: List[Polyline] = []
    for entity in msp:
        paths.extend(dxf_entity_to_paths(entity, arc_step_deg, curve_flatten_dist))
    return paths



def all_points(*path_groups: Sequence[Polyline]) -> np.ndarray:
    arrays = [p for group in path_groups for p in group if len(p) > 0]
    if not arrays:
        return np.empty((0, 2), dtype=float)
    return np.vstack(arrays)



def plot_overlay(
    dxf_paths: Sequence[Polyline],
    g1_paths: Sequence[Polyline],
    g0_paths: Sequence[Polyline],
    out_png: Path,
    title: str,
    figsize: Tuple[float, float],
    dpi: int,
    pad_ratio: float,
    show_axes: bool,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    for p in dxf_paths:
        ax.plot(p[:, 0], p[:, 1], color="red", linewidth=0.8)

    for p in g1_paths:
        ax.plot(p[:, 0], p[:, 1], color="blue", linewidth=0.8)

    for p in g0_paths:
        ax.plot(p[:, 0], p[:, 1], color="blue", linewidth=0.6, linestyle="--", alpha=0.55)

    pts = all_points(dxf_paths, g1_paths, g0_paths)
    if len(pts):
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        dx = xmax - xmin
        dy = ymax - ymin
        span = max(dx, dy, 1.0)
        pad = span * max(0.0, pad_ratio)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)

    legend_handles = [
        Line2D([0], [0], color="red", lw=1.2, label="DXF"),
        Line2D([0], [0], color="blue", lw=1.2, label="G-code"),
    ]

    fig.subplots_adjust(right=0.80, top=0.92)
    ax.legend(
        handles=legend_handles,
        loc=LEGEND_LOCATION,
        bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
        borderaxespad=0.0,
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=0.9,
    )

    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=10)
    ax.set_aspect("equal", adjustable="box")
    if not show_axes:
        ax.axis("off")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)



def run_overlay_batch(
    dxf_dir: Path,
    gcode_dir: Path,
    out_dir: Path,
    arc_step_deg: float = ARC_STEP_DEG,
    curve_flatten_dist: float = CURVE_FLATTEN_DIST,
    figsize: Tuple[float, float] = FIGSIZE,
    dpi: int = DPI,
    pad_ratio: float = PAD_RATIO,
    show_g0: bool = SHOW_G0,
    show_axes: bool = SHOW_AXES,
) -> tuple[int, int]:
    if not dxf_dir.is_dir():
        raise FileNotFoundError(f"DXF folder not found: {dxf_dir}")
    if not gcode_dir.is_dir():
        raise FileNotFoundError(f"G-code folder not found: {gcode_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    dxf_map = list_files(dxf_dir, DXF_EXTS)
    gcode_map = list_files(gcode_dir, GCODE_EXTS)

    common_stems = sorted(set(dxf_map) & set(gcode_map))
    only_dxf = sorted(set(dxf_map) - set(gcode_map))
    only_gcode = sorted(set(gcode_map) - set(dxf_map))

    if not common_stems:
        raise RuntimeError(
            "No matching files found. Files are matched by filename stem, e.g. a.dxf <-> a.gcode"
        )

    print(f"[Overlay] Matched pairs: {len(common_stems)}")
    if only_dxf:
        print(f"[Overlay] DXF without G-code: {len(only_dxf)}")
    if only_gcode:
        print(f"[Overlay] G-code without DXF: {len(only_gcode)}")

    ok = 0
    fail = 0
    for stem in common_stems:
        dxf_path = dxf_map[stem]
        gcode_path = gcode_map[stem]
        out_png = out_dir / f"{stem}_overlay.png"

        try:
            dxf_paths = parse_dxf_file(dxf_path, arc_step_deg, curve_flatten_dist)
            g1_paths, g0_paths = parse_gcode_file(gcode_path, arc_step_deg, show_g0)
            plot_overlay(
                dxf_paths=dxf_paths,
                g1_paths=g1_paths,
                g0_paths=g0_paths,
                out_png=out_png,
                title=stem,
                figsize=figsize,
                dpi=dpi,
                pad_ratio=pad_ratio,
                show_axes=show_axes,
            )
            print(f"[Overlay][OK] {stem} -> {out_png}")
            ok += 1
        except Exception as exc:
            print(f"[Overlay][FAIL] {stem}: {exc}")
            fail += 1

    print("=" * 60)
    print(f"[Overlay] Done. OK={ok}, FAIL={fail}, OUT={out_dir}")
    return ok, fail



def main() -> None:
    run_overlay_batch(
        dxf_dir=DXF_DIR,
        gcode_dir=GCODE_DIR,
        out_dir=OUT_DIR,
        arc_step_deg=ARC_STEP_DEG,
        curve_flatten_dist=CURVE_FLATTEN_DIST,
        figsize=FIGSIZE,
        dpi=DPI,
        pad_ratio=PAD_RATIO,
        show_g0=SHOW_G0,
        show_axes=SHOW_AXES,
    )


if __name__ == "__main__":
    main()

