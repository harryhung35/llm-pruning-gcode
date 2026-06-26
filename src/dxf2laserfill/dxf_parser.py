from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from shapely import build_area
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, snap, unary_union

try:
    from shapely import line_merge as _line_merge
except ImportError:
    from shapely.ops import linemerge as _line_merge

from .models import DXFParseReport

COMPLEX_VIRTUAL_ENTITY_TYPES = {
    "INSERT",
    "DIMENSION",
    "LEADER",
    "MULTILEADER",
    "MLINE",
    "POINT",
    "ACAD_PROXY_ENTITY",
}


class DXFDependencyError(RuntimeError):
    pass


class DXFGeometryExtractor:
    def __init__(
        self,
        flatten_distance: float = 0.01,
        min_curve_segments: int = 8,
        snap_tolerance: float = 1e-4,
        min_segment_length: float = 1e-9,
        force_close_open_contours: bool = True,
    ) -> None:
        self.flatten_distance = flatten_distance
        self.min_curve_segments = min_curve_segments
        self.snap_tolerance = snap_tolerance
        self.min_segment_length = min_segment_length
        self.force_close_open_contours = force_close_open_contours

    def load_fill_area(self, dxf_path: str | Path) -> tuple[BaseGeometry, DXFParseReport]:
        ezdxf, ezpath, text2path = _import_ezdxf_dependencies()

        source = Path(dxf_path)
        doc = ezdxf.readfile(source)
        msp = doc.modelspace()
        report = DXFParseReport(source_file=source)

        linework: list[LineString] = []
        for entity in msp:
            report.entity_count += 1
            for path_obj in self._entity_to_paths(entity, ezpath=ezpath, text2path=text2path, report=report):
                linework.extend(self._path_to_linework(path_obj, report=report))

        if not linework:
            raise ValueError(f"{source.name} 解析後沒有得到任何可用的封閉邊界。")

        linework, snapped_endpoint_count = _snap_linework_endpoints(linework, tolerance=self.snap_tolerance)
        if snapped_endpoint_count > 0:
            report.warnings.append(f"已對齊 {snapped_endpoint_count} 個端點，以修正微小座標落差。")

        area = _build_fill_area(linework, tolerance=self.snap_tolerance)
        if not area.is_empty:
            report.line_count = len(linework)
            return area, report

        if self.force_close_open_contours:
            repaired_linework, added_gap_count = _force_close_linework(linework, tolerance=self.snap_tolerance)
            report.auto_closed_gap_count += added_gap_count
            if added_gap_count > 0:
                report.warnings.append(
                    f"偵測到未封閉邊界，已自動補上 {added_gap_count} 段缺口並強制封閉。"
                )
            area = _build_fill_area(repaired_linework, tolerance=self.snap_tolerance)
            if not area.is_empty:
                report.line_count = len(repaired_linework)
                return area, report
            linework = repaired_linework

        report.line_count = len(linework)
        raise ValueError(
            f"{source.name} 無法從 DXF 邊界建立封閉面域。即使自動補縫後仍失敗，請確認圖形是否可形成有效封閉區域。"
        )

    def _entity_to_paths(
        self,
        entity: Any,
        *,
        ezpath: Any,
        text2path: Any,
        report: DXFParseReport,
    ) -> Iterator[Any]:
        dxftype = entity.dxftype()

        if dxftype == "INSERT":
            inserts = list(entity.multi_insert()) if getattr(entity, "mcount", 1) > 1 else [entity]
            for insert in inserts:
                for virtual_entity in insert.virtual_entities(
                    skipped_entity_callback=lambda skipped, reason: report.skipped_entities.append(
                        f"INSERT child {skipped.dxftype()}: {reason}"
                    )
                ):
                    yield from self._entity_to_paths(
                        virtual_entity,
                        ezpath=ezpath,
                        text2path=text2path,
                        report=report,
                    )
            return

        if dxftype in COMPLEX_VIRTUAL_ENTITY_TYPES and hasattr(entity, "virtual_entities"):
            try:
                for virtual_entity in entity.virtual_entities():
                    yield from self._entity_to_paths(
                        virtual_entity,
                        ezpath=ezpath,
                        text2path=text2path,
                        report=report,
                    )
            except Exception as exc:
                report.warnings.append(f"{dxftype} virtual_entities() 失敗: {exc}")
            return

        if dxftype in {"TEXT", "ATTRIB"}:
            try:
                for path_obj in text2path.make_paths_from_entity(entity):
                    yield path_obj
            except Exception as exc:
                report.warnings.append(f"{dxftype} 轉 path 失敗: {exc}")
            return

        if dxftype in {"HATCH", "MPOLYGON"}:
            try:
                for path_obj in ezpath.from_hatch(entity):
                    yield path_obj
            except Exception as exc:
                report.warnings.append(f"{dxftype} boundary path 解析失敗: {exc}")
            return

        if dxftype in {"XLINE", "RAY", "UNDERLAY", "PDFUNDERLAY", "DWFUNDERLAY", "DGNUNDERLAY", "MTEXT"}:
            report.skipped_entities.append(f"{dxftype}: 不適合作為封閉填充邊界，已略過")
            return

        try:
            path_obj = ezpath.make_path(entity)
        except TypeError:
            report.skipped_entities.append(f"{dxftype}: ezdxf.path.make_path 不支援")
            return
        except Exception as exc:
            report.warnings.append(f"{dxftype} make_path 失敗: {exc}")
            return

        if getattr(path_obj, "has_sub_paths", False):
            yield from path_obj.sub_paths()
        else:
            yield path_obj

    def _path_to_linework(self, path_obj: Any, *, report: DXFParseReport) -> list[LineString]:
        sub_paths: Iterable[Any]
        if getattr(path_obj, "has_sub_paths", False):
            sub_paths = path_obj.sub_paths()
        else:
            sub_paths = [path_obj]

        linework: list[LineString] = []
        for sub_path in sub_paths:
            vertices = [
                (float(vertex.x), float(vertex.y))
                for vertex in sub_path.flattening(self.flatten_distance, self.min_curve_segments)
            ]
            vertices = _remove_consecutive_duplicates(vertices, self.min_segment_length)
            if len(vertices) < 2:
                continue

            is_closed = getattr(sub_path, "is_closed", False)
            should_close = is_closed or _points_close(vertices[0], vertices[-1], self.snap_tolerance)
            if should_close and not _points_close(vertices[0], vertices[-1], self.min_segment_length):
                vertices.append(vertices[0])
                if not is_closed:
                    report.auto_closed_path_count += 1

            for start, end in zip(vertices, vertices[1:]):
                if _segment_length(start, end) <= self.min_segment_length:
                    continue
                linework.append(LineString([start, end]))
        return linework


def _remove_consecutive_duplicates(
    vertices: list[tuple[float, float]],
    tolerance: float,
) -> list[tuple[float, float]]:
    if not vertices:
        return []
    cleaned = [vertices[0]]
    for vertex in vertices[1:]:
        if not _points_close(cleaned[-1], vertex, tolerance):
            cleaned.append(vertex)
    return cleaned


def _points_close(a: tuple[float, float], b: tuple[float, float], tolerance: float) -> bool:
    return abs(a[0] - b[0]) <= tolerance and abs(a[1] - b[1]) <= tolerance


def _segment_length(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def _point_key(point: tuple[float, float], tolerance: float) -> tuple[int, int]:
    scale = max(tolerance, 1e-9)
    return (round(point[0] / scale), round(point[1] / scale))


def _build_fill_area(linework: list[LineString], *, tolerance: float) -> BaseGeometry:
    if not linework:
        return MultiLineString([])

    if len(linework) == 1:
        merged: BaseGeometry = linework[0]
    else:
        merged = unary_union(linework)

    if tolerance > 0:
        merged = snap(merged, merged, tolerance)

    polygons = list(polygonize(merged))
    if polygons:
        area = unary_union(polygons)
        if not area.is_empty:
            return area

    area = build_area(merged)
    if not area.is_empty:
        return area

    if tolerance > 0:
        buffered = merged.buffer(tolerance, join_style=2, cap_style=2)
        if not buffered.is_empty:
            healed = buffered.buffer(-tolerance, join_style=2, cap_style=2)
            if not healed.is_empty:
                return healed

    return area


def _snap_linework_endpoints(
    linework: list[LineString],
    *,
    tolerance: float,
) -> tuple[list[LineString], int]:
    if tolerance <= 0 or not linework:
        return linework, 0

    endpoints: list[tuple[float, float]] = []
    endpoint_refs: list[tuple[int, int]] = []
    for line_index, line in enumerate(linework):
        coords = list(line.coords)
        if len(coords) < 2:
            continue
        endpoints.append((float(coords[0][0]), float(coords[0][1])))
        endpoint_refs.append((line_index, 0))
        endpoints.append((float(coords[-1][0]), float(coords[-1][1])))
        endpoint_refs.append((line_index, 1))

    if not endpoints:
        return linework, 0

    parent = list(range(len(endpoints)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, point in enumerate(endpoints):
        cell = (int(round(point[0] / tolerance)), int(round(point[1] / tolerance)))
        buckets[cell].append(index)

    for cell, indexes in buckets.items():
        neighbor_indexes: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbor_indexes.extend(buckets.get((cell[0] + dx, cell[1] + dy), []))
        for index in indexes:
            point = endpoints[index]
            for neighbor in neighbor_indexes:
                if neighbor <= index:
                    continue
                if _segment_length(point, endpoints[neighbor]) <= tolerance:
                    union(index, neighbor)

    clusters: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for index, point in enumerate(endpoints):
        clusters[find(index)].append(point)

    centroid_map: dict[int, tuple[float, float]] = {}
    for root, points in clusters.items():
        centroid_map[root] = (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )

    snapped_line_endpoints: dict[int, list[tuple[float, float] | None]] = {
        index: [None, None] for index in range(len(linework))
    }
    moved_points = 0
    for index, point in enumerate(endpoints):
        centroid = centroid_map[find(index)]
        if not _points_close(point, centroid, 0.0):
            moved_points += 1
        line_index, endpoint_position = endpoint_refs[index]
        snapped_line_endpoints[line_index][endpoint_position] = centroid

    snapped_lines: list[LineString] = []
    for line_index, line in enumerate(linework):
        coords = list(line.coords)
        if len(coords) < 2:
            continue
        start = snapped_line_endpoints[line_index][0]
        end = snapped_line_endpoints[line_index][1]
        if start is None:
            start = (float(coords[0][0]), float(coords[0][1]))
        if end is None:
            end = (float(coords[-1][0]), float(coords[-1][1]))
        if _segment_length(start, end) > 0:
            snapped_lines.append(LineString([start, end]))
    return snapped_lines, moved_points


def _force_close_linework(
    linework: list[LineString],
    *,
    tolerance: float,
) -> tuple[list[LineString], int]:
    if not linework:
        return linework, 0

    repaired, _ = _snap_linework_endpoints(linework, tolerance=tolerance)
    added_segments: list[LineString] = []

    component_lines = _split_linework_components(repaired, tolerance=tolerance)
    for lines in component_lines:
        added_segments.extend(_close_component_open_chains(lines, tolerance=tolerance))

    repaired = repaired + added_segments
    repaired, _ = _snap_linework_endpoints(repaired, tolerance=tolerance)

    component_segments, endpoint_lookup, component_map = _build_endpoint_graph(repaired, tolerance=tolerance)
    bridging_segments: list[LineString] = []
    for _, keys in component_map.items():
        odd_keys = [key for key in keys if len(component_segments[key]) % 2 == 1]
        bridging_segments.extend(_pair_keys(odd_keys, endpoint_lookup))

    repaired = repaired + bridging_segments
    repaired, _ = _snap_linework_endpoints(repaired, tolerance=tolerance)

    if _build_fill_area(repaired, tolerance=tolerance).is_empty:
        global_odd_keys = _collect_global_odd_keys(repaired, tolerance=tolerance)
        global_segments = _pair_keys(global_odd_keys, endpoint_lookup)
        if global_segments:
            repaired = repaired + global_segments
            repaired, _ = _snap_linework_endpoints(repaired, tolerance=tolerance)
            bridging_segments.extend(global_segments)

    total_added = len(added_segments) + len(bridging_segments)
    return repaired, total_added


def _split_linework_components(
    linework: list[LineString],
    *,
    tolerance: float,
) -> list[list[LineString]]:
    key_to_lines: dict[tuple[int, int], set[int]] = defaultdict(set)
    line_neighbors: dict[int, set[int]] = defaultdict(set)

    for index, line in enumerate(linework):
        coords = list(line.coords)
        if len(coords) < 2:
            continue
        start = (float(coords[0][0]), float(coords[0][1]))
        end = (float(coords[-1][0]), float(coords[-1][1]))
        start_key = _point_key(start, tolerance)
        end_key = _point_key(end, tolerance)
        key_to_lines[start_key].add(index)
        key_to_lines[end_key].add(index)

    for indexes in key_to_lines.values():
        indexes = list(indexes)
        for i in range(len(indexes)):
            for j in range(i + 1, len(indexes)):
                a = indexes[i]
                b = indexes[j]
                line_neighbors[a].add(b)
                line_neighbors[b].add(a)

    components: list[list[LineString]] = []
    visited: set[int] = set()
    for index in range(len(linework)):
        if index in visited:
            continue
        stack = [index]
        component_indexes: list[int] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component_indexes.append(current)
            stack.extend(neighbor for neighbor in line_neighbors[current] if neighbor not in visited)
        components.append([linework[i] for i in component_indexes])
    return components


def _close_component_open_chains(
    linework: list[LineString],
    *,
    tolerance: float,
) -> list[LineString]:
    if not linework:
        return []

    if len(linework) == 1:
        merged: BaseGeometry = linework[0]
    else:
        merged = unary_union(linework)

    merged = _line_merge(merged)

    chains: list[LineString] = []
    if merged.geom_type == "LineString":
        chains = [merged]
    elif merged.geom_type == "MultiLineString":
        chains = [geom for geom in merged.geoms if len(geom.coords) >= 2]
    else:
        return []

    added_segments: list[LineString] = []
    for chain in chains:
        coords = list(chain.coords)
        if len(coords) < 2:
            continue
        start = (float(coords[0][0]), float(coords[0][1]))
        end = (float(coords[-1][0]), float(coords[-1][1]))
        if _segment_length(start, end) > tolerance:
            added_segments.append(LineString([start, end]))
    return added_segments


def _build_endpoint_graph(
    linework: list[LineString],
    *,
    tolerance: float,
) -> tuple[
    dict[tuple[int, int], list[tuple[int, int]]],
    dict[tuple[int, int], tuple[float, float]],
    dict[int, set[tuple[int, int]]],
]:
    adjacency: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    degrees: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    endpoint_lookup: dict[tuple[int, int], tuple[float, float]] = {}

    for line in linework:
        coords = list(line.coords)
        if len(coords) < 2:
            continue
        start = (float(coords[0][0]), float(coords[0][1]))
        end = (float(coords[-1][0]), float(coords[-1][1]))
        start_key = _point_key(start, tolerance)
        end_key = _point_key(end, tolerance)
        endpoint_lookup.setdefault(start_key, start)
        endpoint_lookup.setdefault(end_key, end)
        adjacency[start_key].add(end_key)
        adjacency[end_key].add(start_key)
        degrees[start_key].append(end_key)
        degrees[end_key].append(start_key)

    visited: set[tuple[int, int]] = set()
    component_map: dict[int, set[tuple[int, int]]] = {}
    component_id = 0
    for key in adjacency:
        if key in visited:
            continue
        stack = [key]
        component_nodes: set[tuple[int, int]] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component_nodes.add(current)
            stack.extend(neighbor for neighbor in adjacency[current] if neighbor not in visited)
        component_map[component_id] = component_nodes
        component_id += 1

    return degrees, endpoint_lookup, component_map


def _collect_global_odd_keys(
    linework: list[LineString],
    *,
    tolerance: float,
) -> list[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for line in linework:
        coords = list(line.coords)
        if len(coords) < 2:
            continue
        start = (float(coords[0][0]), float(coords[0][1]))
        end = (float(coords[-1][0]), float(coords[-1][1]))
        counts[_point_key(start, tolerance)] += 1
        counts[_point_key(end, tolerance)] += 1
    return [key for key, count in counts.items() if count % 2 == 1]


def _pair_keys(
    keys: list[tuple[int, int]],
    endpoint_lookup: dict[tuple[int, int], tuple[float, float]],
) -> list[LineString]:
    unused = list(dict.fromkeys(keys))
    added_segments: list[LineString] = []
    while len(unused) >= 2:
        base_key = unused.pop(0)
        base_point = endpoint_lookup[base_key]
        nearest_index = min(
            range(len(unused)),
            key=lambda idx: _segment_length(base_point, endpoint_lookup[unused[idx]]),
        )
        match_key = unused.pop(nearest_index)
        match_point = endpoint_lookup[match_key]
        if _segment_length(base_point, match_point) > 0:
            added_segments.append(LineString([base_point, match_point]))
    if unused:
        base_key = unused.pop()
        base_point = endpoint_lookup[base_key]
        nearest_key = min(
            endpoint_lookup.keys(),
            key=lambda key: _segment_length(base_point, endpoint_lookup[key]) if key != base_key else float("inf"),
        )
        nearest_point = endpoint_lookup[nearest_key]
        if _segment_length(base_point, nearest_point) > 0:
            added_segments.append(LineString([base_point, nearest_point]))
    return added_segments


def _import_ezdxf_dependencies() -> tuple[Any, Any, Any]:
    try:
        import ezdxf
        from ezdxf import path as ezpath
        from ezdxf.addons import text2path
    except ImportError as exc:
        raise DXFDependencyError(
            "缺少 ezdxf。請先在 VS Code 專案中執行: pip install -r requirements.txt"
        ) from exc
    return ezdxf, ezpath, text2path