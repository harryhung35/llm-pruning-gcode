from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry


@dataclass(slots=True)
class FillCut:
    start_x: float
    end_x: float
    y: float


class ScanlineFillGenerator:
    def __init__(
        self,
        line_distance: float,
        start_offset_ratio: float = 0.5,
        serpentine: bool = True,
        min_cut_length: float = 1e-7,
    ) -> None:
        if line_distance <= 0:
            raise ValueError("line_distance 必須大於 0")
        self.line_distance = line_distance
        self.start_offset_ratio = start_offset_ratio
        self.serpentine = serpentine
        self.min_cut_length = min_cut_length

    def generate(self, area: BaseGeometry) -> list[FillCut]:
        if area.is_empty:
            return []

        min_x, min_y, max_x, max_y = area.bounds
        padding = max(self.line_distance * 2.0, 1e-4)
        start_y = min_y + (self.line_distance * self.start_offset_ratio)

        if start_y > max_y:
            start_y = (min_y + max_y) / 2.0

        total_rows = int(math.floor((max_y - start_y) / self.line_distance)) + 1
        cuts: list[FillCut] = []

        for row_index in range(max(total_rows, 1)):
            y = start_y + row_index * self.line_distance
            if y > max_y + 1e-12:
                break
            scanline = LineString([(min_x - padding, y), (max_x + padding, y)])
            intersections = self._extract_segments(area.intersection(scanline), y)
            if not intersections:
                continue

            ordered = sorted(intersections, key=lambda item: (item[0], item[1]))
            if self.serpentine and row_index % 2 == 1:
                ordered = [(end_x, start_x) for start_x, end_x in reversed(ordered)]

            for start_x, end_x in ordered:
                if abs(end_x - start_x) <= self.min_cut_length:
                    continue
                cuts.append(FillCut(start_x=start_x, end_x=end_x, y=y))
        return cuts

    def _extract_segments(self, geometry: BaseGeometry, y: float) -> list[tuple[float, float]]:
        segments: list[tuple[float, float]] = []
        geom_type = geometry.geom_type

        if geom_type == "LineString":
            coords = list(geometry.coords)
            if len(coords) >= 2:
                x1 = float(coords[0][0])
                x2 = float(coords[-1][0])
                segments.append((min(x1, x2), max(x1, x2)))
            return segments

        if geom_type == "MultiLineString":
            for geom in geometry.geoms:
                segments.extend(self._extract_segments(geom, y))
            return segments

        if geom_type == "GeometryCollection":
            for geom in geometry.geoms:
                segments.extend(self._extract_segments(geom, y))
            return segments

        if geom_type in {"Point", "MultiPoint"}:
            return segments

        if geom_type == "LinearRing":
            coords = list(geometry.coords)
            if len(coords) >= 2:
                x_values = sorted(float(point[0]) for point in coords)
                if x_values:
                    segments.append((x_values[0], x_values[-1]))
            return segments

        raise ValueError(f"不支援的交集幾何型別: {geom_type} at y={y}")
