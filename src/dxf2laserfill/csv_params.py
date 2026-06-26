from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from .models import ParameterRow


CONTOUR_FILL_LABELS = {
    "輪廓填充",
    "contour_fill",
    "contour fill",
    "laser_fill",
    "laser fill",
    "fill",
}


@dataclass(slots=True)
class ParameterTable:
    rows: list[ParameterRow]

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "ParameterTable":
        path = Path(csv_path)
        rows: list[ParameterRow] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "filename",
                "process_mode",
                "line_distance",
                "laser_power_s",
                "laser_freq",
                "scan_speed_f",
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"parameters.csv 缺少必要欄位: {', '.join(sorted(missing))}")

            for index, row in enumerate(reader, start=2):
                try:
                    rows.append(
                        ParameterRow(
                            filename=str(row["filename"]).strip(),
                            process_mode=str(row["process_mode"]).strip(),
                            line_distance=float(row["line_distance"]),
                            laser_power_s=float(row["laser_power_s"]),
                            laser_freq=float(row["laser_freq"]),
                            scan_speed_f=float(row["scan_speed_f"]),
                            row_number=index,
                            source_csv=path,
                        )
                    )
                except Exception as exc:  # pragma: no cover - defensive parsing
                    raise ValueError(f"CSV 第 {index} 列解析失敗: {exc}") from exc
        return cls(rows=rows)

    def match_for_dxf(self, dxf_path: str | Path) -> ParameterRow:
        dxf_file = Path(dxf_path)
        stem = self._normalize_name(dxf_file.stem)
        digits = self._digits_only(dxf_file.stem)

        contour_rows = [row for row in self.rows if self._is_contour_fill(row.process_mode)]
        search_rows = contour_rows if contour_rows else self.rows

        exact = [row for row in search_rows if self._normalize_name(row.filename) == stem]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise ValueError(f"DXF {dxf_file.name} 在 CSV 中找到多筆同名參數列。")

        if digits:
            digit_match = [row for row in search_rows if self._digits_only(row.filename) == digits]
            if len(digit_match) == 1:
                return digit_match[0]
            if len(digit_match) > 1:
                raise ValueError(f"DXF {dxf_file.name} 以數字匹配到多筆參數列。")

        raise ValueError(
            f"找不到 {dxf_file.name} 對應的參數列。請確認 DXF 檔名（不含副檔名）與 parameters.csv 的 filename 欄位一致。"
        )

    @staticmethod
    def _normalize_name(value: str) -> str:
        return value.strip().lower()

    @staticmethod
    def _digits_only(value: str) -> str:
        return "".join(re.findall(r"\d+", value))

    @staticmethod
    def _is_contour_fill(value: str) -> bool:
        return value.strip().lower() in CONTOUR_FILL_LABELS
