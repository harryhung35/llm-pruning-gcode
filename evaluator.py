from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import ezdxf
from ezdxf import path as ezpath
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import polygonize, unary_union


@dataclass
class MotionSegment:
    start: tuple[float, float]
    end: tuple[float, float]
    command: str
    line_no: int


@dataclass
class GCodeParseResult:
    cutting_segments: list[MotionSegment] = field(default_factory=list)
    travel_segments: list[MotionSegment] = field(default_factory=list)
    cutting_endpoints: list[tuple[float, float]] = field(default_factory=list)
    detected_groups: int = 0
    legal_groups: int = 0
    first_param_g1: dict[str, float] | None = None
    raw_lines: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _normalize_filename_key(name: str) -> str:
    return Path(name).stem.lower()


PARAM_ALIASES = {
    "filename": ["filename", "file", "name"],
    "s": ["s", "laser_power_s", "power_s"],
    "q": ["q", "laser_freq", "frequency_q"],
    "f": ["f", "scan_speed_f", "feed_f"],
    "line_distance": ["line_distance", "pitch", "scan_pitch", "hatch_spacing", "spacing"],
}


TIMING_ALIASES = {
    "filename": ["filename", "file", "name"],
    "inference_time": [
        "inference_time",
        "inference_time_s",
        "inference_time_sec",
        "total_time",
        "total_time_s",
        "total_time_sec",
    ],
    "first_token_latency": [
        "first_token_latency",
        "first_token_latency_s",
        "first_token_latency_sec",
        "latency",
        "latency_s",
        "latency_sec",
    ],
}


def _find_column(fieldnames: list[str], aliases: list[str]) -> str | None:
    lowered = {name.lower(): name for name in fieldnames}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def load_param_csv(csv_path: Path) -> dict[str, dict[str, float]]:
    mapping: dict[str, dict[str, float]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Parameter CSV has no header: {csv_path}")
        fn_col = _find_column(reader.fieldnames, PARAM_ALIASES["filename"])
        s_col = _find_column(reader.fieldnames, PARAM_ALIASES["s"])
        q_col = _find_column(reader.fieldnames, PARAM_ALIASES["q"])
        f_col = _find_column(reader.fieldnames, PARAM_ALIASES["f"])
        ld_col = _find_column(reader.fieldnames, PARAM_ALIASES["line_distance"])
        if not all([fn_col, s_col, q_col, f_col]):
            raise ValueError(
                f"Parameter CSV must contain filename/S/Q/F columns. Found: {reader.fieldnames}"
            )
        for row in reader:
            key = _normalize_filename_key(str(row[fn_col]))
            record = {
                "S": float(row[s_col]),
                "Q": float(row[q_col]),
                "F": float(row[f_col]),
            }
            if ld_col is not None and str(row.get(ld_col, "")).strip() != "":
                record["line_distance"] = float(row[ld_col])
            mapping[key] = record
    return mapping


def load_timing_csv(csv_path: Path) -> dict[str, dict[str, float]]:
    mapping: dict[str, dict[str, float]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Timing CSV has no header: {csv_path}")
        fn_col = _find_column(reader.fieldnames, TIMING_ALIASES["filename"])
        inf_col = _find_column(reader.fieldnames, TIMING_ALIASES["inference_time"])
        lat_col = _find_column(reader.fieldnames, TIMING_ALIASES["first_token_latency"])
        if not all([fn_col, inf_col, lat_col]):
            raise ValueError(
                f"Timing CSV must contain filename/inference_time/first_token_latency columns. Found: {reader.fieldnames}"
            )
        for row in reader:
            key = _normalize_filename_key(str(row[fn_col]))
            mapping[key] = {
                "inference_time": float(row[inf_col]),
                "first_token_latency": float(row[lat_col]),
            }
    return mapping


TOKEN_RE = re.compile(r"([A-Za-z])\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
COMMENT_PAREN_RE = re.compile(r"\([^)]*\)")


def strip_comments(line: str) -> str:
    line = COMMENT_PAREN_RE.sub("", line)
    if ";" in line:
        line = line.split(";", 1)[0]
    return line.strip()


def parse_tokens(line: str) -> dict[str, list[float]]:
    tokens: dict[str, list[float]] = {}
    for letter, number in TOKEN_RE.findall(line):
        tokens.setdefault(letter.upper(), []).append(float(number))
    return tokens


def _last(values: list[float] | None) -> float | None:
    if not values:
        return None
    return values[-1]


def parse_gcode_file(gcode_path: Path) -> GCodeParseResult:
    result = GCodeParseResult()
    x: float | None = None
    y: float | None = None
    laser_on = False
    pending_g0 = False
    active_group = False
    group_has_g0 = False
    group_g1_count = 0

    with gcode_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    result.raw_lines = [line.rstrip("\n") for line in lines]

    for line_no, raw in enumerate(lines, start=1):
        clean = strip_comments(raw)
        if not clean:
            continue
        tokens = parse_tokens(clean)
        if not tokens:
            continue

        g_cmd = _last(tokens.get("G"))
        m_cmd = _last(tokens.get("M"))
        new_x = _last(tokens.get("X"))
        new_y = _last(tokens.get("Y"))

        if g_cmd is not None and int(round(g_cmd)) in (0, 1):
            cmd = f"G{int(round(g_cmd))}"
            target_x = x if new_x is None else float(new_x)
            target_y = y if new_y is None else float(new_y)

            if x is not None and y is not None and target_x is not None and target_y is not None:
                seg = MotionSegment(start=(x, y), end=(target_x, target_y), command=cmd, line_no=line_no)
                if cmd == "G1" and laser_on:
                    result.cutting_segments.append(seg)
                    result.cutting_endpoints.extend([seg.start, seg.end])
                    group_g1_count += 1
                    if result.first_param_g1 is None:
                        s = _last(tokens.get("S"))
                        q = _last(tokens.get("Q"))
                        fval = _last(tokens.get("F"))
                        if s is not None or q is not None or fval is not None:
                            result.first_param_g1 = {
                                "S": float("nan") if s is None else float(s),
                                "Q": float("nan") if q is None else float(q),
                                "F": float("nan") if fval is None else float(fval),
                            }
                else:
                    result.travel_segments.append(seg)

            x, y = target_x, target_y
            if cmd == "G0":
                pending_g0 = True
                if not active_group:
                    group_has_g0 = True

        if m_cmd is not None:
            m_code = int(round(m_cmd))
            if m_code == 3:
                result.detected_groups += 1
                active_group = True
                laser_on = True
                group_has_g0 = pending_g0
                pending_g0 = False
                group_g1_count = 0
            elif m_code == 5:
                if active_group and group_has_g0 and group_g1_count > 0:
                    result.legal_groups += 1
                laser_on = False
                active_group = False
                group_has_g0 = False
                group_g1_count = 0

    if active_group:
        result.errors.append("File ended before M5 closed the last active group.")

    return result


def _points_to_linestring(points: list[tuple[float, float]], force_close: bool = False) -> LineString | None:
    if len(points) < 2:
        return None
    pts = [(float(x), float(y)) for x, y in points]
    if force_close and pts[0] != pts[-1]:
        pts.append(pts[0])
    cleaned = [pts[0]]
    for p in pts[1:]:
        if p != cleaned[-1]:
            cleaned.append(p)
    if len(cleaned) < 2:
        return None
    return LineString(cleaned)


def _extract_entity_lines(entity: Any, flatness: float = 0.005) -> list[LineString]:
    etype = entity.dxftype()
    lines: list[LineString] = []

    if etype == "LINE":
        start = entity.dxf.start
        end = entity.dxf.end
        ls = _points_to_linestring([(start.x, start.y), (end.x, end.y)])
        if ls is not None:
            lines.append(ls)
        return lines

    try:
        p = ezpath.make_path(entity)
        pts = [(v.x, v.y) for v in p.flattening(distance=flatness)]
        if not pts:
            return lines
        force_close = bool(getattr(entity, "closed", False))
        if hasattr(entity.dxf, "flags") and etype in {"LWPOLYLINE", "POLYLINE"}:
            force_close = force_close or bool(entity.closed)
        if etype in {"CIRCLE", "ELLIPSE"}:
            force_close = True
        ls = _points_to_linestring(pts, force_close=force_close)
        if ls is not None:
            lines.append(ls)
    except Exception:
        if etype == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in entity.get_points("xy")]
            ls = _points_to_linestring(pts, force_close=bool(entity.closed))
            if ls is not None:
                lines.append(ls)
        elif etype == "POLYLINE":
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
            ls = _points_to_linestring(pts, force_close=bool(entity.is_closed))
            if ls is not None:
                lines.append(ls)

    return lines


def dxf_to_legal_region(dxf_path: Path, flatness: float = 0.005) -> Polygon | MultiPolygon:
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    all_lines: list[LineString] = []

    for entity in msp:
        all_lines.extend(_extract_entity_lines(entity, flatness=flatness))

    if not all_lines:
        raise ValueError(f"No 2D linework could be extracted from DXF: {dxf_path}")

    merged = unary_union(all_lines)
    polygons = list(polygonize(merged))
    if not polygons:
        raise ValueError(
            f"DXF did not polygonize into a closed region. Check that contours are closed: {dxf_path}"
        )

    region = unary_union(polygons)
    if isinstance(region, (Polygon, MultiPolygon)):
        return region
    polys = [g for g in getattr(region, "geoms", []) if isinstance(g, Polygon)]
    if not polys:
        raise ValueError(f"DXF region could not be converted to Polygon/MultiPolygon: {dxf_path}")
    return unary_union(polys)


def infer_path_width_mm(segments: list[MotionSegment], fallback: float = 0.005) -> float:
    horizontal_y = []
    for seg in segments:
        dy = abs(seg.end[1] - seg.start[1])
        dx = abs(seg.end[0] - seg.start[0])
        if dx > 0 and dy <= 1e-6:
            horizontal_y.append(round(seg.start[1], 6))
    uniq = sorted(set(horizontal_y))
    if len(uniq) < 2:
        return fallback
    diffs = [b - a for a, b in zip(uniq, uniq[1:]) if (b - a) > 1e-9]
    if not diffs:
        return fallback
    width = median(diffs)
    return max(float(width), fallback)


def extract_predicted_row_ys(segments: list[MotionSegment], horiz_tol: float = 1e-6) -> list[float]:
    ys: list[float] = []
    for seg in segments:
        dy = abs(seg.end[1] - seg.start[1])
        dx = abs(seg.end[0] - seg.start[0])
        if dx > 0 and dy <= horiz_tol:
            ys.append(round(seg.start[1], 6))
    return sorted(set(ys))


def _collect_line_strings(geom) -> list[LineString]:
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom] if geom.length > 0 else []
    if isinstance(geom, MultiLineString):
        return [ls for ls in geom.geoms if ls.length > 0]
    if isinstance(geom, GeometryCollection):
        out: list[LineString] = []
        for g in geom.geoms:
            out.extend(_collect_line_strings(g))
        return out
    return []


def build_reference_fill_centerlines(
    legal_region: Polygon | MultiPolygon,
    line_distance_mm: float,
    tol_mm: float = 0.005,
    y_offset_mm: float | None = None,
) -> list[LineString]:
    if line_distance_mm <= 0:
        raise ValueError(f"line_distance_mm must be > 0, got {line_distance_mm}")

    minx, miny, maxx, maxy = legal_region.bounds
    height = maxy - miny
    if height <= 0:
        return []

    if y_offset_mm is None:
        y_offset_mm = line_distance_mm / 2.0

    y_offset_mm = float(y_offset_mm) % float(line_distance_mm)
    eps = max(tol_mm * 0.1, 1e-9)
    y_values: list[float] = []

    if height <= line_distance_mm:
        y_values = [(miny + maxy) / 2.0]
    else:
        y = miny + y_offset_mm
        while y <= maxy + eps:
            y_values.append(y)
            y += line_distance_mm
        if not y_values:
            y_values = [(miny + maxy) / 2.0]

    centerlines: list[LineString] = []
    pad = max(line_distance_mm, tol_mm, 1.0)
    for y in y_values:
        scan = LineString([(minx - pad, y), (maxx + pad, y)])
        inter = scan.intersection(legal_region)
        centerlines.extend(_collect_line_strings(inter))

    return centerlines


def segments_to_covered_region(
    segments: list[MotionSegment],
    path_width_mm: float,
    cap_style: int = 2,
    join_style: int = 2,
):
    if not segments:
        return GeometryCollection()
    half = path_width_mm / 2.0
    buffered = [
        LineString([seg.start, seg.end]).buffer(half, cap_style=cap_style, join_style=join_style)
        for seg in segments
    ]
    return unary_union(buffered)


def build_reference_fill_region(
    legal_region: Polygon | MultiPolygon,
    line_distance_mm: float,
    tol_mm: float = 0.005,
    y_offset_mm: float | None = None,
):
    centerlines = build_reference_fill_centerlines(
        legal_region=legal_region,
        line_distance_mm=line_distance_mm,
        tol_mm=tol_mm,
        y_offset_mm=y_offset_mm,
    )
    if not centerlines:
        return GeometryCollection()
    half = line_distance_mm / 2.0
    buffered = [ls.buffer(half, cap_style=2, join_style=2) for ls in centerlines]
    return unary_union(buffered)


def compute_iou(gt_fill_region, pred_fill_region) -> float:
    if gt_fill_region.is_empty and pred_fill_region.is_empty:
        return 1.0
    if gt_fill_region.is_empty or pred_fill_region.is_empty:
        return 0.0
    inter = gt_fill_region.intersection(pred_fill_region).area
    union = gt_fill_region.union(pred_fill_region).area
    if union <= 0:
        return 0.0
    return float(inter / union)


def choose_best_y_offset(
    legal_region: Polygon | MultiPolygon,
    pred_fill_region,
    predicted_row_ys: list[float],
    line_distance_mm: float,
    tol_mm: float = 0.005,
) -> tuple[float, Any, float]:
    miny = legal_region.bounds[1]
    default_offset = line_distance_mm / 2.0
    candidate_offsets = {round(default_offset % line_distance_mm, 9)}

    for y in predicted_row_ys:
        candidate_offsets.add(round((float(y) - miny) % line_distance_mm, 9))

    best_offset = default_offset % line_distance_mm
    best_region = build_reference_fill_region(
        legal_region=legal_region,
        line_distance_mm=line_distance_mm,
        tol_mm=tol_mm,
        y_offset_mm=best_offset,
    )
    best_iou = compute_iou(best_region, pred_fill_region)

    for candidate in sorted(candidate_offsets):
        region = build_reference_fill_region(
            legal_region=legal_region,
            line_distance_mm=line_distance_mm,
            tol_mm=tol_mm,
            y_offset_mm=float(candidate),
        )
        score = compute_iou(region, pred_fill_region)
        if score > best_iou:
            best_iou = score
            best_offset = float(candidate)
            best_region = region

    return float(best_offset), best_region, float(best_iou)


def compute_coordinate_mae(legal_region, endpoints: list[tuple[float, float]]) -> float:
    if not endpoints:
        return float("nan")
    boundary = legal_region.boundary
    errs = []
    for x, y in endpoints:
        p = Point(x, y)
        if legal_region.covers(p):
            errs.append(0.0)
        else:
            errs.append(float(boundary.distance(p)))
    return float(sum(errs) / len(errs)) if errs else float("nan")


def compute_param_mae(pred: dict[str, float] | None, gt: dict[str, float] | None) -> dict[str, float]:
    out = {"mae_s": float("nan"), "mae_q": float("nan"), "mae_f": float("nan")}
    if pred is None or gt is None:
        return out
    out["mae_s"] = abs(pred.get("S", float("nan")) - gt["S"]) if not math.isnan(pred.get("S", float("nan"))) else float("nan")
    out["mae_q"] = abs(pred.get("Q", float("nan")) - gt["Q"]) if not math.isnan(pred.get("Q", float("nan"))) else float("nan")
    out["mae_f"] = abs(pred.get("F", float("nan")) - gt["F"]) if not math.isnan(pred.get("F", float("nan"))) else float("nan")
    return out


def evaluate_one(
    dxf_path: Path,
    gcode_path: Path,
    gt_params: dict[str, float] | None,
    timing: dict[str, float] | None,
    tol_mm: float = 0.005,
    path_width_mm: float | None = None,
) -> dict[str, Any]:
    parse = parse_gcode_file(gcode_path)
    legal_region = dxf_to_legal_region(dxf_path, flatness=tol_mm)

    inferred_width = infer_path_width_mm(parse.cutting_segments, fallback=tol_mm)
    gt_line_distance = None if gt_params is None else gt_params.get("line_distance")
    effective_width = path_width_mm
    if effective_width is None:
        if gt_line_distance is not None and gt_line_distance > 0:
            effective_width = float(gt_line_distance)
        else:
            effective_width = inferred_width

    pred_fill_region = segments_to_covered_region(parse.cutting_segments, path_width_mm=effective_width)
    predicted_row_ys = extract_predicted_row_ys(parse.cutting_segments)
    _, _, best_iou = choose_best_y_offset(
        legal_region=legal_region,
        pred_fill_region=pred_fill_region,
        predicted_row_ys=predicted_row_ys,
        line_distance_mm=effective_width,
        tol_mm=tol_mm,
    )

    detected = parse.detected_groups
    legal = parse.legal_groups
    syntax_rate = float(legal / detected) if detected > 0 else float("nan")

    metrics = {
        "iou": best_iou,
        "group_syntax_success_rate": syntax_rate,
        "coordinate_mae": compute_coordinate_mae(legal_region, parse.cutting_endpoints),
        "inference_time": float("nan") if timing is None else timing.get("inference_time", float("nan")),
        "first_token_latency": float("nan") if timing is None else timing.get("first_token_latency", float("nan")),
    }
    metrics.update(compute_param_mae(parse.first_param_g1, gt_params))
    return metrics


def pair_files(dxf_dir: Path, gcode_dir: Path) -> list[tuple[Path, Path]]:
    dxf_map = {p.stem.lower(): p for p in dxf_dir.glob("*.dxf")}
    pairs: list[tuple[Path, Path]] = []
    missing_dxf = []
    for g in sorted(gcode_dir.glob("*.gcode")) + sorted(gcode_dir.glob("*.nc")) + sorted(gcode_dir.glob("*.txt")):
        key = g.stem.lower()
        dxf = dxf_map.get(key)
        if dxf is None:
            missing_dxf.append(g.name)
            continue
        pairs.append((dxf, g))
    if missing_dxf:
        print("[WARN] These G-code files had no matching DXF by stem name:")
        for name in missing_dxf:
            print(f"  - {name}")
    return pairs


def evaluate_all(
    dxf_dir: Path,
    gcode_dir: Path,
    param_csv: Path,
    timing_csv: Path | None,
    output_dir: Path,
    tol_mm: float = 0.005,
    path_width_mm: float | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    params = load_param_csv(param_csv)
    timings = load_timing_csv(timing_csv) if timing_csv is not None else {}
    pairs = pair_files(dxf_dir, gcode_dir)
    if not pairs:
        raise FileNotFoundError("No matching DXF/G-code pairs were found by filename stem.")

    metrics_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for dxf_path, gcode_path in pairs:
        key = _normalize_filename_key(gcode_path.name)
        gt_param = params.get(key)
        timing = timings.get(key)
        try:
            metrics = evaluate_one(
                dxf_path=dxf_path,
                gcode_path=gcode_path,
                gt_params=gt_param,
                timing=timing,
                tol_mm=tol_mm,
                path_width_mm=path_width_mm,
            )
            metrics["filename"] = gcode_path.name
            metrics_rows.append(metrics)
            print(f"[OK] {gcode_path.name}")
        except Exception as exc:
            failures.append({
                "filename": gcode_path.name,
                "dxf_file": dxf_path.name,
                "error": str(exc),
            })
            print(f"[FAIL] {gcode_path.name}: {exc}")

    metrics_csv = output_dir / "metrics.csv"
    metrics_fields = [
        "filename",
        "iou",
        "group_syntax_success_rate",
        "coordinate_mae",
        "mae_s",
        "mae_q",
        "mae_f",
        "inference_time",
        "first_token_latency",
    ]
    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_fields)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)

    summary = {
        "num_pairs": len(pairs),
        "num_success": len(metrics_rows),
        "num_failures": len(failures),
        "failures": failures,
    }
    if metrics_rows:
        numeric_cols = [
            "iou",
            "group_syntax_success_rate",
            "coordinate_mae",
            "mae_s",
            "mae_q",
            "mae_f",
            "inference_time",
            "first_token_latency",
        ]
        stats: dict[str, dict[str, float]] = {}
        for col in numeric_cols:
            vals = [float(r[col]) for r in metrics_rows if r[col] is not None and not math.isnan(float(r[col]))]
            if vals:
                stats[col] = {
                    "mean": sum(vals) / len(vals),
                    "min": min(vals),
                    "max": max(vals),
                }
        summary["stats"] = stats

    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics_csv, summary_json


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate generated G-code against DXF + parameter CSVs, with timing CSV. "
            "IoU is computed against a reference fill region generated from the DXF with best y-offset alignment."
        )
    )
    p.add_argument("--dxf-dir", required=True, type=Path)
    p.add_argument("--gcode-dir", required=True, type=Path)
    p.add_argument("--param-csv", required=True, type=Path)
    p.add_argument("--timing-csv", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--tol-mm", type=float, default=0.005)
    p.add_argument(
        "--path-width-mm",
        type=float,
        default=None,
        help=(
            "Optional override for the reference fill line_distance / effective scan width used in IoU. "
            "If omitted, the code uses line_distance from the parameter CSV when available, otherwise it infers it from the generated G-code."
        ),
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    metrics_csv, summary_json = evaluate_all(
        dxf_dir=args.dxf_dir,
        gcode_dir=args.gcode_dir,
        param_csv=args.param_csv,
        timing_csv=args.timing_csv,
        output_dir=args.output_dir,
        tol_mm=args.tol_mm,
        path_width_mm=args.path_width_mm,
    )
    print(f"\nMetrics written to: {metrics_csv}")
    print(f"Summary written to: {summary_json}")


if __name__ == "__main__":
    main()
