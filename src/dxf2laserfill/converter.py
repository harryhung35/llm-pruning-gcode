from __future__ import annotations

import csv
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from shapely.geometry import GeometryCollection, LineString, MultiLineString
from shapely.geometry.base import BaseGeometry

from .dxf_parser import DXFGeometryExtractor
from .gcode_writer import GCodeWriter
from .models import BatchConversionResult, ConversionItem, FillSegment, ParameterRow


class BatchConverter:
    XY_DECIMALS = 4
    XY_QUANT = Decimal("0.0001")

    def __init__(
        self,
        *,
        flatten_distance: float = 0.01,
        min_curve_segments: int = 8,
        snap_tolerance: float = 1e-4,
        start_offset_ratio: float = 0.5,
        serpentine: bool = True,
        force_close_open_contours: bool = True,
    ) -> None:
        self.flatten_distance = flatten_distance
        self.min_curve_segments = min_curve_segments
        self.snap_tolerance = snap_tolerance
        self.start_offset_ratio = start_offset_ratio
        self.serpentine = serpentine
        self.force_close_open_contours = force_close_open_contours

        self.extractor = DXFGeometryExtractor(
            flatten_distance=flatten_distance,
            min_curve_segments=min_curve_segments,
            snap_tolerance=snap_tolerance,
            force_close_open_contours=force_close_open_contours,
        )
        self.writer = GCodeWriter()

        self._hold_words = (
            "維持不變",
            "不做調整",
            "保持不變",
            "維持原設定",
            "保持原設定",
            "保持原本設定",
            "維持原值",
            "維持",
            "保持",
        )
        self._increase_words = (
            "逐步遞增至",
            "遞增至",
            "提高至",
            "逐步上修至",
            "上修至",
            "向上調整至",
            "上調至",
        )
        self._decrease_words = (
            "逐步遞減至",
            "遞減至",
            "降低至",
            "逐步下修至",
            "下修至",
            "向下調整至",
            "下調至",
        )
        self._param_aliases = {
            "S": ("S", "S(功率)", "功率S", "功率參數S", "功率"),
            "Q": ("Q", "Q(頻率)", "頻率Q", "頻率參數Q", "頻率"),
            "F": ("F", "F(速度)", "速度F", "速度參數F", "速度", "掃描速度"),
        }

    def convert_directory(
        self,
        *,
        dxf_dir: str | Path,
        parameter_csv: str | Path,
        output_dir: str | Path,
    ) -> BatchConversionResult:
        dxf_dir = Path(dxf_dir)
        parameter_csv = Path(parameter_csv)
        output_dir = Path(output_dir)

        if not dxf_dir.exists():
            raise FileNotFoundError(f"找不到 DXF 資料夾：{dxf_dir}")
        if not parameter_csv.exists():
            raise FileNotFoundError(f"找不到參數檔：{parameter_csv}")

        output_dir.mkdir(parents=True, exist_ok=True)

        parameter_map = self._load_parameters(parameter_csv)
        dxf_files = sorted(dxf_dir.glob("*.dxf"), key=self._natural_sort_key)

        result = BatchConversionResult()
        report_lines: list[str] = []

        for dxf_file in dxf_files:
            try:
                parameter = self._find_parameter_for_dxf(dxf_file, parameter_map)
                area, parse_report = self.extractor.load_fill_area(dxf_file)
                segments = self._generate_fill_segments(area, parameter.line_distance)
                parsed_strategy = self._apply_dynamic_parameters(segments, parameter)

                output_path = output_dir / f"{dxf_file.stem}.gcode"
                self.writer.write_file(
                    output_path,
                    segments=segments,
                    parameters=parameter,
                )

                message = "OK"
                extras: list[str] = []
                if parse_report.auto_closed_path_count > 0:
                    extras.append(f"path_auto_closed={parse_report.auto_closed_path_count}")
                if parse_report.auto_closed_gap_count > 0:
                    extras.append(f"gap_auto_closed={parse_report.auto_closed_gap_count}")
                if extras:
                    message = f"OK ({', '.join(extras)})"

                result.items.append(
                    ConversionItem(
                        dxf_file=dxf_file,
                        success=True,
                        message=message,
                    )
                )

                report_lines.extend(self._build_report_block(
                    dxf_file=dxf_file,
                    output_path=output_path,
                    parameter=parameter,
                    parse_report=parse_report,
                    segment_count=len(segments),
                    success=True,
                    error_message="",
                    parsed_strategy=parsed_strategy,
                ))
            except Exception as exc:
                result.items.append(
                    ConversionItem(
                        dxf_file=dxf_file,
                        success=False,
                        message=str(exc),
                    )
                )
                report_lines.extend(self._build_report_block(
                    dxf_file=dxf_file,
                    output_path=output_dir / f"{dxf_file.stem}.gcode",
                    parameter=None,
                    parse_report=None,
                    segment_count=0,
                    success=False,
                    error_message=str(exc),
                    parsed_strategy=None,
                ))

        report_path = output_dir.parent / "conversion_report.txt"
        report_path.write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")

        return result

    def _load_parameters(self, csv_path: Path) -> dict[str, ParameterRow]:
        rows: dict[str, ParameterRow] = {}

        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {
                "filename",
                "line_distance",
                "laser_power_s",
                "laser_freq",
                "scan_speed_f",
                "strategy_rule",
            }
            header = set(reader.fieldnames or [])
            missing = required - header
            if missing:
                raise ValueError(f"parameters.csv 缺少必要欄位：{', '.join(sorted(missing))}")

            for row in reader:
                raw_filename = str(row.get("filename") or "").strip()
                if not raw_filename:
                    continue

                key = self._normalize_filename(raw_filename)
                parameter = ParameterRow(
                    filename_key=key,
                    raw_filename=raw_filename,
                    process_mode=str(row.get("process_mode") or "").strip(),
                    line_distance=float(row["line_distance"]),
                    laser_power_s=float(row["laser_power_s"]),
                    laser_freq=int(round(float(row["laser_freq"]))),
                    scan_speed_f=int(round(float(row["scan_speed_f"]))),
                    strategy_rule=str(row.get("strategy_rule") or "").strip(),
                    source_row=dict(row),
                )
                rows[key] = parameter

        return rows

    def _find_parameter_for_dxf(
        self,
        dxf_file: Path,
        parameter_map: dict[str, ParameterRow],
    ) -> ParameterRow:
        key = self._normalize_filename(dxf_file.name)
        if key in parameter_map:
            return parameter_map[key]

        stem_only = self._normalize_filename(dxf_file.stem)
        if stem_only in parameter_map:
            return parameter_map[stem_only]

        raise KeyError(f"{dxf_file.name} 在 parameters.csv 找不到對應的 filename 參數。")

    def _normalize_filename(self, value: str) -> str:
        text = value.strip().replace("\\", "/")
        text = text.split("/")[-1]
        text = Path(text).stem if "." in text else text
        return text.strip().lower()

    def _natural_sort_key(self, path: Path) -> list[Any]:
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.stem)
            if part != ""
        ]

    def _generate_fill_segments(self, area: BaseGeometry, line_distance: float) -> list[FillSegment]:
        line_distance_dec = Decimal(str(line_distance))
        if line_distance_dec <= 0:
            raise ValueError("line_distance 必須大於 0。")
        if area.is_empty:
            return []

        min_x, min_y, max_x, max_y = area.bounds
        min_y_dec = Decimal(str(min_y))
        max_y_dec = Decimal(str(max_y))
        start_offset_dec = line_distance_dec * Decimal(str(self.start_offset_ratio))
        y_dec = self._quantize_xy(min_y_dec + start_offset_dec)

        eps_dec = max(Decimal("0.0000001"), line_distance_dec * Decimal("0.000001"))
        pad = max(1.0, (max_x - min_x), (max_y - min_y), float(line_distance_dec) * 10.0)

        segments: list[FillSegment] = []
        active_row_index = 0

        while y_dec <= max_y_dec + eps_dec:
            y_float = float(y_dec)
            scanline = LineString([(min_x - pad, y_float), (max_x + pad, y_float)])
            hit = area.intersection(scanline)

            raw_row_segments = self._extract_scan_segments(hit)
            rounded_row_segments = self._round_and_filter_row_segments(raw_row_segments)

            if rounded_row_segments:
                if self.serpentine and active_row_index % 2 == 1:
                    ordered = list(reversed(rounded_row_segments))
                    for x0_dec, x1_dec in ordered:
                        segments.append(
                            FillSegment(
                                x_start=float(x1_dec),
                                y_start=float(y_dec),
                                x_end=float(x0_dec),
                                y_end=float(y_dec),
                                row_index=active_row_index,
                            )
                        )
                else:
                    for x0_dec, x1_dec in rounded_row_segments:
                        segments.append(
                            FillSegment(
                                x_start=float(x0_dec),
                                y_start=float(y_dec),
                                x_end=float(x1_dec),
                                y_end=float(y_dec),
                                row_index=active_row_index,
                            )
                        )
                active_row_index += 1

            y_dec += line_distance_dec

        return segments

    def _extract_scan_segments(self, geom: BaseGeometry) -> list[tuple[float, float]]:
        segments: list[tuple[float, float]] = []

        if geom.is_empty:
            return segments

        if isinstance(geom, LineString):
            coords = list(geom.coords)
            if len(coords) >= 2:
                x0 = float(coords[0][0])
                x1 = float(coords[-1][0])
                if x0 != x1:
                    segments.append((min(x0, x1), max(x0, x1)))
            return segments

        if isinstance(geom, MultiLineString):
            for item in geom.geoms:
                segments.extend(self._extract_scan_segments(item))
            return self._merge_overlapping_segments(segments)

        if isinstance(geom, GeometryCollection):
            for item in geom.geoms:
                if isinstance(item, (LineString, MultiLineString, GeometryCollection)):
                    segments.extend(self._extract_scan_segments(item))
            return self._merge_overlapping_segments(segments)

        if hasattr(geom, "geoms"):
            for item in geom.geoms:
                if isinstance(item, (LineString, MultiLineString, GeometryCollection)):
                    segments.extend(self._extract_scan_segments(item))
            return self._merge_overlapping_segments(segments)

        return self._merge_overlapping_segments(segments)

    def _merge_overlapping_segments(self, segments: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not segments:
            return []

        segments = sorted((min(a, b), max(a, b)) for a, b in segments if a != b)
        merged: list[list[float]] = [[segments[0][0], segments[0][1]]]
        tol = max(1e-9, self.snap_tolerance)

        for start, end in segments[1:]:
            last = merged[-1]
            if start <= last[1] + tol:
                if end > last[1]:
                    last[1] = end
            else:
                merged.append([start, end])

        return [(start, end) for start, end in merged]

    def _round_and_filter_row_segments(
        self,
        segments: list[tuple[float, float]],
    ) -> list[tuple[Decimal, Decimal]]:
        rounded: list[tuple[Decimal, Decimal]] = []

        for start, end in segments:
            start_dec = self._quantize_xy(Decimal(str(min(start, end))))
            end_dec = self._quantize_xy(Decimal(str(max(start, end))))
            if start_dec == end_dec:
                continue
            rounded.append((start_dec, end_dec))

        if not rounded:
            return []

        rounded.sort(key=lambda item: (item[0], item[1]))
        merged: list[list[Decimal]] = [[rounded[0][0], rounded[0][1]]]

        for start_dec, end_dec in rounded[1:]:
            last = merged[-1]
            if start_dec <= last[1]:
                if end_dec > last[1]:
                    last[1] = end_dec
            else:
                merged.append([start_dec, end_dec])

        return [(start_dec, end_dec) for start_dec, end_dec in merged]

    def _quantize_xy(self, value: Decimal) -> Decimal:
        return value.quantize(self.XY_QUANT, rounding=ROUND_HALF_UP)

    def _apply_dynamic_parameters(
        self,
        segments: list[FillSegment],
        parameter: ParameterRow,
    ) -> dict[str, str]:
        if not segments:
            return {"S": "none", "Q": "none", "F": "none"}

        row_ids = sorted({segment.row_index for segment in segments})
        row_map = {row_id: idx for idx, row_id in enumerate(row_ids)}
        total_rows = len(row_ids)

        s_plan_info = self._parse_strategy_for_parameter(
            rule=parameter.strategy_rule,
            parameter_name="S",
            base_value=parameter.laser_power_s,
            total_rows=total_rows,
            as_int=False,
            decimals=2,
        )
        q_plan_info = self._parse_strategy_for_parameter(
            rule=parameter.strategy_rule,
            parameter_name="Q",
            base_value=float(parameter.laser_freq),
            total_rows=total_rows,
            as_int=True,
            decimals=0,
        )
        f_plan_info = self._parse_strategy_for_parameter(
            rule=parameter.strategy_rule,
            parameter_name="F",
            base_value=float(parameter.scan_speed_f),
            total_rows=total_rows,
            as_int=True,
            decimals=0,
        )

        s_values = s_plan_info["values"]
        q_values = q_plan_info["values"]
        f_values = f_plan_info["values"]

        for segment in segments:
            idx = row_map[segment.row_index]
            segment.s_value = float(s_values[idx])
            segment.q_value = int(q_values[idx])
            segment.f_value = int(f_values[idx])

        return {
            "S": s_plan_info["summary"],
            "Q": q_plan_info["summary"],
            "F": f_plan_info["summary"],
        }

    def _parse_strategy_for_parameter(
        self,
        *,
        rule: str,
        parameter_name: str,
        base_value: float,
        total_rows: int,
        as_int: bool,
        decimals: int,
    ) -> dict[str, Any]:
        chunks = [chunk for chunk in re.split(r"[，。；;]", rule) if chunk.strip()]
        target_chunk = ""

        for chunk in chunks:
            if self._chunk_contains_parameter(chunk, parameter_name):
                target_chunk = chunk.strip()
                break

        if not target_chunk:
            values = self._build_value_schedule(
                base_value=base_value,
                target_value=base_value,
                mode="hold",
                total_rows=total_rows,
                as_int=as_int,
                decimals=decimals,
            )
            return {
                "mode": "hold",
                "target": base_value,
                "values": values,
                "summary": f"hold:{self._fmt_value(base_value, as_int, decimals)}",
            }

        normalized_chunk = self._normalize_rule_text(target_chunk)

        if any(word in normalized_chunk for word in map(self._normalize_rule_text, self._hold_words)):
            values = self._build_value_schedule(
                base_value=base_value,
                target_value=base_value,
                mode="hold",
                total_rows=total_rows,
                as_int=as_int,
                decimals=decimals,
            )
            return {
                "mode": "hold",
                "target": base_value,
                "values": values,
                "summary": f"hold:{self._fmt_value(base_value, as_int, decimals)}",
            }

        target_value = self._extract_last_number(target_chunk)
        if target_value is None:
            values = self._build_value_schedule(
                base_value=base_value,
                target_value=base_value,
                mode="hold",
                total_rows=total_rows,
                as_int=as_int,
                decimals=decimals,
            )
            return {
                "mode": "hold",
                "target": base_value,
                "values": values,
                "summary": f"hold:{self._fmt_value(base_value, as_int, decimals)}",
            }

        increase_hit = any(word in normalized_chunk for word in map(self._normalize_rule_text, self._increase_words))
        decrease_hit = any(word in normalized_chunk for word in map(self._normalize_rule_text, self._decrease_words))

        if increase_hit or decrease_hit:
            values = self._build_value_schedule(
                base_value=base_value,
                target_value=target_value,
                mode="ramp",
                total_rows=total_rows,
                as_int=as_int,
                decimals=decimals,
            )
            direction = "ramp_up" if target_value >= base_value else "ramp_down"
            return {
                "mode": "ramp",
                "target": target_value,
                "values": values,
                "summary": (
                    f"{direction}:{self._fmt_value(base_value, as_int, decimals)}"
                    f"->{self._fmt_value(target_value, as_int, decimals)}"
                ),
            }

        if target_value != base_value:
            values = self._build_value_schedule(
                base_value=base_value,
                target_value=target_value,
                mode="ramp",
                total_rows=total_rows,
                as_int=as_int,
                decimals=decimals,
            )
            direction = "ramp_up" if target_value >= base_value else "ramp_down"
            return {
                "mode": "ramp",
                "target": target_value,
                "values": values,
                "summary": (
                    f"{direction}:{self._fmt_value(base_value, as_int, decimals)}"
                    f"->{self._fmt_value(target_value, as_int, decimals)}"
                ),
            }

        values = self._build_value_schedule(
            base_value=base_value,
            target_value=base_value,
            mode="hold",
            total_rows=total_rows,
            as_int=as_int,
            decimals=decimals,
        )
        return {
            "mode": "hold",
            "target": base_value,
            "values": values,
            "summary": f"hold:{self._fmt_value(base_value, as_int, decimals)}",
        }

    def _build_value_schedule(
        self,
        *,
        base_value: float,
        target_value: float,
        mode: str,
        total_rows: int,
        as_int: bool,
        decimals: int,
    ) -> list[int] | list[float]:
        if total_rows <= 0:
            return []

        if total_rows == 1 or mode == "hold":
            if as_int:
                return [int(round(base_value)) for _ in range(total_rows)]
            return [round(base_value, decimals) for _ in range(total_rows)]

        values: list[float] = []
        for i in range(total_rows):
            ratio = i / (total_rows - 1)
            value = base_value + (target_value - base_value) * ratio
            values.append(value)

        values[0] = base_value
        values[-1] = target_value

        if as_int:
            return [int(round(v)) for v in values]
        return [round(v, decimals) for v in values]

    def _normalize_rule_text(self, text: str) -> str:
        return (
            str(text)
            .replace(" ", "")
            .replace("　", "")
            .replace("（", "(")
            .replace("）", ")")
            .upper()
        )

    def _chunk_contains_parameter(self, chunk: str, parameter_name: str) -> bool:
        normalized = self._normalize_rule_text(chunk)
        aliases = self._param_aliases[parameter_name]
        return any(self._normalize_rule_text(alias) in normalized for alias in aliases)

    def _extract_last_number(self, text: str) -> float | None:
        matches = re.findall(r"-?\d+(?:\.\d+)?", text)
        if not matches:
            return None
        return float(matches[-1])

    def _fmt_value(self, value: float, as_int: bool, decimals: int) -> str:
        if as_int:
            return str(int(round(value)))
        return f"{value:.{decimals}f}"

    def _build_report_block(
        self,
        *,
        dxf_file: Path,
        output_path: Path,
        parameter: ParameterRow | None,
        parse_report: Any,
        segment_count: int,
        success: bool,
        error_message: str,
        parsed_strategy: dict[str, str] | None,
    ) -> list[str]:
        lines: list[str] = []
        lines.append(f"DXF: {dxf_file}")
        lines.append(f"Output: {output_path}")
        lines.append(f"Success: {success}")
        if parameter is not None:
            lines.append(f"line_distance: {parameter.line_distance}")
            lines.append(f"laser_power_s: {parameter.laser_power_s}")
            lines.append(f"laser_freq: {parameter.laser_freq}")
            lines.append(f"scan_speed_f: {parameter.scan_speed_f}")
            lines.append(f"strategy_rule: {parameter.strategy_rule}")
            lines.append(f"segments: {segment_count}")
        if parsed_strategy is not None:
            lines.append(f"parsed_strategy_S: {parsed_strategy['S']}")
            lines.append(f"parsed_strategy_Q: {parsed_strategy['Q']}")
            lines.append(f"parsed_strategy_F: {parsed_strategy['F']}")
        if parse_report is not None:
            lines.append(f"entity_count: {parse_report.entity_count}")
            lines.append(f"line_count: {parse_report.line_count}")
            lines.append(f"auto_closed_path_count: {parse_report.auto_closed_path_count}")
            lines.append(f"auto_closed_gap_count: {parse_report.auto_closed_gap_count}")
            if parse_report.warnings:
                lines.append("warnings:")
                lines.extend(f"  - {item}" for item in parse_report.warnings)
            if parse_report.skipped_entities:
                lines.append("skipped_entities:")
                lines.extend(f"  - {item}" for item in parse_report.skipped_entities)
        if error_message:
            lines.append(f"error: {error_message}")
        lines.append("")
        return lines