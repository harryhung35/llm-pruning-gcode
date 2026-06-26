from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParameterRow:
    filename_key: str
    raw_filename: str
    process_mode: str
    line_distance: float
    laser_power_s: float
    laser_freq: int
    scan_speed_f: int
    source_row: dict[str, str] = field(default_factory=dict)


@dataclass
class DXFParseReport:
    source_file: Path
    entity_count: int = 0
    line_count: int = 0
    auto_closed_path_count: int = 0
    auto_closed_gap_count: int = 0
    warnings: list[str] = field(default_factory=list)
    skipped_entities: list[str] = field(default_factory=list)


@dataclass
class FillSegment:
    x_start: float
    y_start: float
    x_end: float
    y_end: float
    row_index: int = 0
    s_value: float | None = None
    q_value: int | None = None
    f_value: int | None = None


@dataclass
class ConversionItem:
    dxf_file: Path
    success: bool
    message: str


@dataclass
class BatchConversionResult:
    items: list[ConversionItem] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for item in self.items if item.success)

    @property
    def fail_count(self) -> int:
        return sum(1 for item in self.items if not item.success)
