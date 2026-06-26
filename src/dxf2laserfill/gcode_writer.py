from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from .models import FillSegment, ParameterRow


class GCodeWriter:
    XY_DECIMALS = 4
    S_DECIMALS = 2

    def write_file(
        self,
        output_path: str | Path,
        *,
        segments: list[FillSegment],
        parameters: ParameterRow,
    ) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = [
            "G21",
            "G90",
        ]

        for segment in segments:
            x_start = self._round_xy(segment.x_start)
            y_start = self._round_xy(segment.y_start)
            x_end = self._round_xy(segment.x_end)
            y_end = self._round_xy(segment.y_end)

            if self._is_zero_length_after_format(x_start, y_start, x_end, y_end):
                continue

            s_value = parameters.laser_power_s if segment.s_value is None else segment.s_value
            q_value = parameters.laser_freq if segment.q_value is None else segment.q_value
            f_value = parameters.scan_speed_f if segment.f_value is None else segment.f_value

            lines.append(f"G0 X{x_start} Y{y_start}")
            lines.append("M3")
            lines.append(
                "G1 "
                f"X{x_end} "
                f"Y{y_end} "
                f"S{self._fmt_s(s_value)} "
                f"Q{self._fmt_int(q_value)} "
                f"F{self._fmt_int(f_value)}"
            )
            lines.append("M5")

        text = "\n".join(lines)
        if text:
            text += "\n"
        output.write_text(text, encoding="utf-8")

    def _round_xy(self, value: float) -> str:
        quant = Decimal("1." + ("0" * self.XY_DECIMALS))
        return format(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP), f".{self.XY_DECIMALS}f")

    def _fmt_s(self, value: float) -> str:
        quant = Decimal("1." + ("0" * self.S_DECIMALS))
        return format(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP), f".{self.S_DECIMALS}f")

    @staticmethod
    def _fmt_int(value: int | float) -> str:
        return str(int(round(value)))

    @staticmethod
    def _is_zero_length_after_format(
        x_start: str,
        y_start: str,
        x_end: str,
        y_end: str,
    ) -> bool:
        return x_start == x_end and y_start == y_end