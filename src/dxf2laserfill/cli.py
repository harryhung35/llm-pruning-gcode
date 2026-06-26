from __future__ import annotations

import argparse
from pathlib import Path

from .converter import BatchConverter



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="批量把 DXF 轉成雷射輪廓填充 G-code，並將 S/Q/F 與 line_distance 套用自 parameters.csv"
    )
    parser.add_argument("--dxf-dir", default="./dxf", help="放置 DXF 檔案的資料夾")
    parser.add_argument("--csv", default="./parameters.csv", help="參數 CSV 路徑")
    parser.add_argument("--out-dir", default="./gcode_out", help="輸出 G-code 的資料夾")
    parser.add_argument(
        "--flatten-distance",
        type=float,
        default=0.01,
        help="曲線近似為折線時的最大誤差（drawing units）",
    )
    parser.add_argument(
        "--min-curve-segments",
        type=int,
        default=8,
        help="每段曲線至少切成幾段折線",
    )
    parser.add_argument(
        "--snap-tolerance",
        type=float,
        default=1e-5,
        help="封閉邊界補縫容差",
    )
    parser.add_argument(
        "--start-offset-ratio",
        type=float,
        default=0.5,
        help="第一條掃描線相對於 line_distance 的偏移比例，預設 0.5 代表從半個 pitch 開始掃",
    )
    parser.add_argument(
        "--no-serpentine",
        action="store_true",
        help="關閉蛇形掃描，所有填充線皆由左往右",
    )
    parser.add_argument(
        "--no-force-close",
        action="store_true",
        help="關閉 DXF 開放輪廓自動強制封閉功能",
    )
    return parser



def main() -> int:
    args = build_parser().parse_args()

    converter = BatchConverter(
        flatten_distance=args.flatten_distance,
        min_curve_segments=args.min_curve_segments,
        snap_tolerance=args.snap_tolerance,
        start_offset_ratio=args.start_offset_ratio,
        serpentine=not args.no_serpentine,
        force_close_open_contours=not args.no_force_close,
    )
    result = converter.convert_directory(
        dxf_dir=Path(args.dxf_dir),
        parameter_csv=Path(args.csv),
        output_dir=Path(args.out_dir),
    )

    print(f"Total: {len(result.items)}")
    print(f"Success: {result.success_count}")
    print(f"Failed: {result.fail_count}")
    for item in result.items:
        status = "OK" if item.success else "FAIL"
        print(f"[{status}] {item.dxf_file.name} -> {item.message}")
    return 0 if result.fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
