import gc
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

try:
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box as shapely_box
    from shapely.ops import unary_union
    from shapely.validation import make_valid
except ImportError:
    GeometryCollection = None
    MultiPolygon = None
    Polygon = None
    shapely_box = None
    unary_union = None
    make_valid = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def load_coco(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def split_values(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(";") if item.strip()]


def build_category_id_filter(categories, include_category_names=None, include_category_ids=None):
    names = set(split_values(include_category_names))
    ids = {int(value) for value in split_values(include_category_ids)}
    if not names and not ids:
        return None
    for category in categories:
        if category.get("name") in names:
            ids.add(int(category["id"]))
    return ids


def image_search_dirs(image_dir):
    dirs = [Path(path) for path in split_values(image_dir)]
    if not dirs:
        raise ValueError("at least one image directory is required")
    return dirs


def find_image_path(image_dirs, file_name):
    file_name = Path(file_name)
    candidates = []
    if file_name.is_absolute():
        candidates.append(file_name)
    else:
        candidates.extend(directory / file_name for directory in image_dirs)
        candidates.extend(directory / file_name.name for directory in image_dirs)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_excluded_file_names(exclude_file_names=None, exclude_list_path=None):
    excluded = {Path(name).name for name in split_values(exclude_file_names)}
    if exclude_list_path:
        path = Path(exclude_list_path)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    excluded.add(Path(line).name)
    return excluded


def bbox_to_rect(bbox):
    if bbox is None or len(bbox) < 4:
        return None
    x, y, width, height = [float(value) for value in bbox[:4]]
    if width <= 0 or height <= 0:
        return None
    return {
        "x1": x,
        "y1": y,
        "x2": x + width,
        "y2": y + height,
        "width": width,
        "height": height,
        "cx": x + width / 2.0,
        "cy": y + height / 2.0,
    }


def points_to_rect(points):
    if not points:
        return None
    clean = []
    for point in points:
        try:
            x, y = point[:2]
            clean.append((float(x), float(y)))
        except (TypeError, ValueError, IndexError):
            continue
    if len(clean) < 3:
        return None
    xs = [x for x, _ in clean]
    ys = [y for _, y in clean]
    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": width,
        "height": height,
        "cx": x1 + width / 2.0,
        "cy": y1 + height / 2.0,
    }


def normalize_polygon_points(values):
    if not values:
        return []
    if all(isinstance(item, (int, float)) for item in values):
        coords = [float(value) for value in values]
        if len(coords) < 6:
            return []
        if len(coords) % 2 == 1:
            coords = coords[:-1]
        return [(coords[index], coords[index + 1]) for index in range(0, len(coords), 2)]
    points = []
    for item in values:
        try:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                first, second = item[0], item[1]
                if isinstance(first, (int, float)) and isinstance(second, (int, float)):
                    points.append((float(first), float(second)))
        except (TypeError, ValueError):
            continue
    return points if len(points) >= 3 else []


def segmentation_to_points(segmentation):
    if not segmentation or isinstance(segmentation, dict):
        return []
    if isinstance(segmentation, (list, tuple)):
        if segmentation and all(isinstance(item, (int, float)) for item in segmentation):
            return normalize_polygon_points(segmentation)
        points = []
        for polygon in segmentation:
            points.extend(normalize_polygon_points(polygon))
        return points
    return []


def annotation_group_id(annotation):
    for key in ("group_id", "groupId", "group", "line_id", "lineId"):
        value = annotation.get(key)
        if value is not None:
            return str(value)
    attributes = annotation.get("attributes") or {}
    for key in ("group_id", "groupId", "group", "line_id", "lineId"):
        value = attributes.get(key)
        if value is not None:
            return str(value)
    return None


def annotation_to_item(annotation):
    points = segmentation_to_points(annotation.get("segmentation"))
    rect = bbox_to_rect(annotation.get("bbox")) or points_to_rect(points)
    if rect is None:
        return None
    if not points:
        points = rect_points(rect)
    item = dict(rect)
    item["points"] = clean_polygon_points(points)
    item["group_id"] = annotation_group_id(annotation)
    item["annotation_id"] = annotation.get("id")
    return item


def rect_union(rects):
    return {
        "x1": min(rect["x1"] for rect in rects),
        "y1": min(rect["y1"] for rect in rects),
        "x2": max(rect["x2"] for rect in rects),
        "y2": max(rect["y2"] for rect in rects),
    }


def rect_center_y(rect):
    return (rect["y1"] + rect["y2"]) / 2.0


def rect_height(rect):
    return rect["y2"] - rect["y1"]


def vertical_overlap_ratio(first, second):
    overlap = min(first["y2"], second["y2"]) - max(first["y1"], second["y1"])
    if overlap <= 0:
        return 0.0
    return overlap / max(1e-6, min(rect_height(first), rect_height(second)))


def can_join_line(box, line_rect, min_y_overlap, center_y_tolerance):
    overlap = vertical_overlap_ratio(box, line_rect)
    if overlap >= min_y_overlap:
        return True
    avg_height = (rect_height(box) + rect_height(line_rect)) / 2.0
    return abs(rect_center_y(box) - rect_center_y(line_rect)) <= center_y_tolerance * avg_height


def group_boxes_into_lines(boxes, min_y_overlap=0.35, center_y_tolerance=0.6):
    lines = []
    for box in sorted(boxes, key=lambda item: (item["cy"], item["x1"])):
        best_index = None
        best_score = -1.0
        for index, line in enumerate(lines):
            line_rect = rect_union(line)
            if not can_join_line(box, line_rect, min_y_overlap, center_y_tolerance):
                continue
            score = vertical_overlap_ratio(box, line_rect) - abs(box["cy"] - rect_center_y(line_rect)) * 1e-4
            if score > best_score:
                best_index = index
                best_score = score
        if best_index is None:
            lines.append([box])
        else:
            lines[best_index].append(box)

    for line in lines:
        line.sort(key=lambda item: item["x1"])
    lines.sort(key=lambda line: (rect_union(line)["y1"], rect_union(line)["x1"]))
    return lines


def group_items_into_lines(items, min_y_overlap=0.35, center_y_tolerance=0.6, line_group_mode="auto"):
    if line_group_mode not in {"auto", "group-id", "geometry"}:
        raise ValueError(f"unknown line_group_mode: {line_group_mode}")

    has_group_ids = any(item.get("group_id") is not None for item in items)
    use_group_ids = line_group_mode == "group-id" or (line_group_mode == "auto" and has_group_ids)
    stats = {
        "group_id_lines": 0,
        "geometry_lines": 0,
        "annotations_with_group_id": sum(1 for item in items if item.get("group_id") is not None),
    }

    if not use_group_ids:
        lines = group_boxes_into_lines(items, min_y_overlap=min_y_overlap, center_y_tolerance=center_y_tolerance)
        stats["geometry_lines"] = len(lines)
        return lines, stats

    grouped = defaultdict(list)
    missing_group = []
    for item in items:
        group_id = item.get("group_id")
        if group_id is None:
            missing_group.append(item)
        else:
            grouped[group_id].append(item)

    lines = list(grouped.values())
    stats["group_id_lines"] = len(lines)
    if missing_group:
        geometry_lines = group_boxes_into_lines(
            missing_group,
            min_y_overlap=min_y_overlap,
            center_y_tolerance=center_y_tolerance,
        )
        lines.extend(geometry_lines)
        stats["geometry_lines"] = len(geometry_lines)

    for line in lines:
        line.sort(key=lambda item: item["x1"])
    lines.sort(key=lambda line: (rect_union(line)["y1"], rect_union(line)["x1"]))
    return lines, stats


def convex_hull(points):
    points = sorted(set((float(x), float(y)) for x, y in points))
    if len(points) <= 1:
        return points

    def cross(origin, first, second):
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def triangle_area(previous, point, next_point):
    return abs(
        (previous[0] * (point[1] - next_point[1])
        + point[0] * (next_point[1] - previous[1])
        + next_point[0] * (previous[1] - point[1]))
        / 2.0
    )


def reduce_convex_polygon_points(points, max_points=None):
    points = clean_polygon_points(points)
    if not max_points or len(points) <= max_points:
        return points
    max_points = max(3, int(max_points))
    reduced = list(points)
    while len(reduced) > max_points:
        index = min(
            range(len(reduced)),
            key=lambda item: triangle_area(reduced[item - 1], reduced[item], reduced[(item + 1) % len(reduced)]),
        )
        del reduced[index]
    return reduced


def clean_polygon_points(points):
    cleaned = []
    for x, y in points:
        point = (float(x), float(y))
        if cleaned and cleaned[-1] == point:
            continue
        cleaned.append(point)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return cleaned


def limit_hull_points(points, max_hull_points=None):
    points = clean_polygon_points(points)
    if not max_hull_points or len(points) <= max_hull_points:
        return points

    mandatory = convex_hull(points)
    if len(mandatory) > int(max_hull_points):
        return reduce_convex_polygon_points(mandatory, max_hull_points)

    seen = set(mandatory)
    remaining = [point for point in points if point not in seen]
    budget = max(0, int(max_hull_points) - len(mandatory))
    if budget <= 0:
        return mandatory

    if len(remaining) <= budget:
        sampled = remaining
    else:
        step = len(remaining) / budget
        sampled = [remaining[int(index * step)] for index in range(budget)]
    return clean_polygon_points(mandatory + sampled)


def rect_points(box, padding=0.0, image_size=None):
    x1 = box["x1"] - padding
    y1 = box["y1"] - padding
    x2 = box["x2"] + padding
    y2 = box["y2"] + padding
    if image_size is not None:
        width, height = image_size
        x1 = min(max(0.0, x1), float(width - 1))
        y1 = min(max(0.0, y1), float(height - 1))
        x2 = min(max(0.0, x2), float(width - 1))
        y2 = min(max(0.0, y2), float(height - 1))
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def line_to_points(line_boxes, padding=0.0, image_size=None):
    points = []
    for box in line_boxes:
        if box.get("points") and padding == 0:
            item_points = box["points"]
            if image_size is not None:
                width, height = image_size
                item_points = [
                    (
                        min(max(0.0, float(x)), float(width - 1)),
                        min(max(0.0, float(y)), float(height - 1)),
                    )
                    for x, y in item_points
                ]
            points.extend(item_points)
        else:
            points.extend(rect_points(box, padding=padding, image_size=image_size))
    return points


def require_shapely():
    if Polygon is None:
        raise ImportError("shapely is required for this operation. Install it with: pip install shapely")


def geometry_polygon_parts(geometry):
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        parts = []
        for item in geometry.geoms:
            parts.extend(geometry_polygon_parts(item))
        return parts
    return []


def valid_polygon_geometry(points):
    require_shapely()
    points = clean_polygon_points(points)
    if len(points) < 3:
        return None
    geometry = Polygon(points)
    if not geometry.is_valid:
        geometry = make_valid(geometry)
    parts = [part for part in geometry_polygon_parts(geometry) if part.area > 0]
    if not parts:
        return None
    return max(parts, key=lambda item: item.area)


def gap_buffer(geometry, distance):
    if distance <= 0:
        return geometry
    return geometry.buffer(distance, quad_segs=1, join_style=2)


def reference_geometry_for_line(line_boxes, image_size=None):
    require_shapely()
    parts = []
    for box in line_boxes:
        if box.get("points"):
            geometry = valid_polygon_geometry(box["points"])
            if geometry is not None:
                parts.append(geometry)
                continue
        points = rect_points(box, image_size=image_size)
        x1, y1 = points[0]
        x2, y2 = points[2]
        parts.append(shapely_box(x1, y1, x2, y2))
    if not parts:
        return None
    return unary_union(parts)


def pick_best_polygon_part(geometry, reference_geometry=None):
    parts = [part for part in geometry_polygon_parts(geometry) if part.area > 0]
    if not parts:
        return None
    if reference_geometry is None or reference_geometry.is_empty:
        return max(parts, key=lambda item: item.area)
    return max(parts, key=lambda item: (item.intersection(reference_geometry).area, item.area))


def exterior_point_count(geometry):
    if geometry is None or geometry.is_empty:
        return 0
    return len(clean_polygon_points(list(geometry.exterior.coords)))


def simplify_geometry_to_max_points(geometry, max_points, reference_geometry=None):
    max_points = max(3, int(max_points))
    if exterior_point_count(geometry) <= max_points:
        return geometry

    minx, miny, maxx, maxy = geometry.bounds
    span = max(maxx - minx, maxy - miny, 1.0)
    low = 0.0
    high = 0.25
    best = None

    while high <= span * 2:
        candidate = pick_best_polygon_part(
            geometry.simplify(high, preserve_topology=True),
            reference_geometry=reference_geometry,
        )
        if candidate is not None and exterior_point_count(candidate) <= max_points:
            best = candidate
            break
        low = high
        high *= 2.0

    if best is None:
        return geometry

    for _ in range(16):
        middle = (low + high) / 2.0
        candidate = pick_best_polygon_part(
            geometry.simplify(middle, preserve_topology=True),
            reference_geometry=reference_geometry,
        )
        if candidate is not None and exterior_point_count(candidate) <= max_points:
            best = candidate
            high = middle
        else:
            low = middle
    return best


def geometry_to_polygon_points(geometry, reference_geometry=None, simplify_tolerance=0.0, max_points=None):
    geometry = pick_best_polygon_part(geometry, reference_geometry=reference_geometry)
    if geometry is None or geometry.is_empty:
        return []
    if simplify_tolerance > 0:
        geometry = geometry.simplify(simplify_tolerance, preserve_topology=True)
        geometry = pick_best_polygon_part(geometry, reference_geometry=reference_geometry)
        if geometry is None or geometry.is_empty:
            return []
    if max_points and exterior_point_count(geometry) > int(max_points):
        geometry = simplify_geometry_to_max_points(
            geometry,
            max_points=max_points,
            reference_geometry=reference_geometry,
        )
    return clean_polygon_points(list(geometry.exterior.coords))


def concave_hull_points(points, concavity=2.0, length_threshold=0.0):
    try:
        from concave_hull import concave_hull
    except ImportError as exc:
        raise ImportError("concave-hull is required for --hull-method concave. Install it with: pip install concave-hull") from exc
    return clean_polygon_points(concave_hull(points, concavity=concavity, length_threshold=length_threshold))


def alpha_shape_points(points, alpha=1.5, reference_geometry=None, simplify_tolerance=0.0):
    try:
        import alphashape
    except ImportError as exc:
        raise ImportError("alphashape is required for --hull-method alpha. Install it with: pip install alphashape") from exc
    geometry = alphashape.alphashape(points, alpha)
    return geometry_to_polygon_points(geometry, reference_geometry=reference_geometry, simplify_tolerance=simplify_tolerance)


def line_to_polygon(
    line_boxes,
    padding=0.0,
    image_size=None,
    hull_method="convex",
    concavity=2.0,
    length_threshold=0.0,
    alpha=1.5,
    simplify_tolerance=0.0,
    max_hull_points=None,
):
    points = line_to_points(line_boxes, padding=padding, image_size=image_size)
    if len(points) < 3:
        return []
    points = limit_hull_points(points, max_hull_points=max_hull_points)
    if hull_method == "convex":
        polygon = reduce_convex_polygon_points(convex_hull(points), max_hull_points)
    elif hull_method == "concave":
        polygon = concave_hull_points(points, concavity=concavity, length_threshold=length_threshold)
    elif hull_method == "alpha":
        reference_geometry = reference_geometry_for_line(line_boxes, image_size=image_size)
        polygon = alpha_shape_points(
            points,
            alpha=alpha,
            reference_geometry=reference_geometry,
            simplify_tolerance=simplify_tolerance,
        )
    else:
        raise ValueError(f"unknown hull_method: {hull_method}")

    geometry = valid_polygon_geometry(polygon)
    if geometry is None:
        polygon = reduce_convex_polygon_points(convex_hull(points), max_hull_points)
        geometry = valid_polygon_geometry(polygon)
    return geometry_to_polygon_points(
        geometry,
        simplify_tolerance=simplify_tolerance,
        max_points=max_hull_points,
    )


def make_non_overlapping_polygons(
    polygons,
    line_boxes,
    image_size,
    min_polygon_area=4.0,
    non_overlap_gap=2.0,
    simplify_tolerance=0.0,
    max_hull_points=None,
):
    require_shapely()
    width, height = image_size
    image_bounds = shapely_box(0, 0, max(0, width - 1), max(0, height - 1))
    occupied = None
    kept_polygons = []
    kept_lines = []
    stats = {
        "intersections_removed": 0,
        "invalid_polygons": 0,
        "dropped_polygons": 0,
    }

    for polygon, boxes in zip(polygons, line_boxes):
        geometry = valid_polygon_geometry(polygon)
        if geometry is None:
            stats["invalid_polygons"] += 1
            stats["dropped_polygons"] += 1
            continue

        reference_geometry = reference_geometry_for_line(boxes, image_size=image_size)
        geometry = pick_best_polygon_part(geometry.intersection(image_bounds), reference_geometry)
        if geometry is None or geometry.area < min_polygon_area:
            stats["dropped_polygons"] += 1
            continue

        if occupied is not None and not occupied.is_empty:
            blocker = gap_buffer(occupied, non_overlap_gap)
            overlap_area = geometry.intersection(blocker).area
            if overlap_area > 1e-6:
                stats["intersections_removed"] += 1
                geometry = pick_best_polygon_part(geometry.difference(blocker), reference_geometry)

        if geometry is None or geometry.area < min_polygon_area:
            stats["dropped_polygons"] += 1
            continue

        points = geometry_to_polygon_points(
            geometry,
            reference_geometry=reference_geometry,
            simplify_tolerance=simplify_tolerance,
            max_points=max_hull_points,
        )
        final_geometry = valid_polygon_geometry(points)
        if final_geometry is None or final_geometry.area < min_polygon_area:
            stats["dropped_polygons"] += 1
            continue

        if occupied is not None and not occupied.is_empty:
            overlap_area = final_geometry.intersection(occupied).area
            if overlap_area > 1e-6:
                stats["intersections_removed"] += 1
                final_geometry = pick_best_polygon_part(
                    final_geometry.difference(gap_buffer(occupied, max(non_overlap_gap, 0.01))),
                    reference_geometry,
                )
                points = geometry_to_polygon_points(
                    final_geometry,
                    reference_geometry=reference_geometry,
                    simplify_tolerance=simplify_tolerance,
                    max_points=max_hull_points,
                )
                final_geometry = valid_polygon_geometry(points)
                if final_geometry is None or final_geometry.area < min_polygon_area:
                    stats["dropped_polygons"] += 1
                    continue
                if final_geometry.intersection(occupied).area > 1e-6:
                    stats["dropped_polygons"] += 1
                    continue

        kept_polygons.append(points)
        kept_lines.append(boxes)
        occupied = final_geometry if occupied is None else unary_union([occupied, final_geometry]).buffer(0)

    return kept_polygons, kept_lines, stats


def count_polygon_overlaps(polygons):
    require_shapely()
    geometries = []
    for polygon in polygons:
        geometry = valid_polygon_geometry(polygon)
        if geometry is not None:
            geometries.append(geometry)
    overlaps = 0
    max_overlap_area = 0.0
    for index, first in enumerate(geometries):
        for second in geometries[index + 1:]:
            area = first.intersection(second).area
            if area > 1e-6:
                overlaps += 1
                max_overlap_area = max(max_overlap_area, area)
    return overlaps, max_overlap_area


def build_line_polygons(
    lines,
    image_size,
    polygon_padding=0.0,
    hull_method="convex",
    concavity=2.0,
    length_threshold=0.0,
    alpha=1.5,
    prevent_intersections=True,
    non_overlap_gap=0.0,
    min_polygon_area=4.0,
    simplify_tolerance=0.0,
    max_hull_points=None,
):
    raw_polygons = []
    raw_lines = []
    for line in lines:
        polygon = line_to_polygon(
            line,
            padding=polygon_padding,
            image_size=image_size,
            hull_method=hull_method,
            concavity=concavity,
            length_threshold=length_threshold,
            alpha=alpha,
            simplify_tolerance=simplify_tolerance,
            max_hull_points=max_hull_points,
        )
        if len(polygon) >= 3:
            raw_polygons.append(polygon)
            raw_lines.append(line)

    stats = {
        "intersections_removed": 0,
        "invalid_polygons": 0,
        "dropped_polygons": len(lines) - len(raw_polygons),
        "remaining_intersections": 0,
        "max_remaining_overlap_area": 0.0,
    }
    if prevent_intersections:
        polygons, kept_lines, non_overlap_stats = make_non_overlapping_polygons(
            raw_polygons,
            raw_lines,
            image_size,
            min_polygon_area=min_polygon_area,
            non_overlap_gap=non_overlap_gap,
            simplify_tolerance=simplify_tolerance,
            max_hull_points=max_hull_points,
        )
        stats.update({key: stats.get(key, 0) + value for key, value in non_overlap_stats.items()})
    else:
        polygons, kept_lines = raw_polygons, raw_lines

    if Polygon is not None:
        overlaps, max_overlap_area = count_polygon_overlaps(polygons)
        stats["remaining_intersections"] = overlaps
        stats["max_remaining_overlap_area"] = max_overlap_area
    return polygons, kept_lines, stats


def round_polygon_points(points, digits=2):
    return clean_polygon_points((round(float(x), digits), round(float(y), digits)) for x, y in points)


def polygon_to_gt_line(polygon, max_points=None):
    polygon = round_polygon_points(polygon)
    if len(polygon) < 3:
        return None
    if Polygon is not None:
        geometry = valid_polygon_geometry(polygon)
        if geometry is None:
            return None
        polygon = geometry_to_polygon_points(geometry, max_points=max_points)
        polygon = round_polygon_points(polygon)
        geometry = valid_polygon_geometry(polygon)
        if geometry is None:
            return None
        polygon = round_polygon_points(list(geometry.exterior.coords))
        if len(polygon) < 3:
            return None

    values = []
    for x, y in polygon:
        values.append(f"{x:.2f}")
        values.append(f"{y:.2f}")
    return ",".join(values)


def group_annotations_by_image(annotations, ignore_crowd=True, category_ids=None):
    grouped = defaultdict(list)
    for annotation in annotations:
        if ignore_crowd and annotation.get("iscrowd"):
            continue
        if category_ids is not None and int(annotation.get("category_id", -1)) not in category_ids:
            continue
        grouped[annotation["image_id"]].append(annotation)
    return grouped


def annotation_items_for_image(annotations, min_box_area=1.0):
    items = []
    skipped = 0
    for annotation in annotations:
        item = annotation_to_item(annotation)
        if item is None or item["width"] * item["height"] < min_box_area:
            skipped += 1
            continue
        items.append(item)
    return items, skipped


def build_annotations_by_image(annotations, ignore_crowd=True, min_box_area=1.0, category_ids=None):
    grouped = defaultdict(list)
    raw_grouped = group_annotations_by_image(annotations, ignore_crowd=ignore_crowd, category_ids=category_ids)
    for image_id, image_annotations in raw_grouped.items():
        grouped[image_id], _ = annotation_items_for_image(image_annotations, min_box_area=min_box_area)
    return grouped


def copy_image(source, destination, copy_mode):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "none":
        return
    try:
        if source.resolve() == destination.resolve():
            return
    except FileNotFoundError:
        pass
    if destination.exists():
        destination.unlink()
    if copy_mode == "copy":
        shutil.copy2(source, destination)
    elif copy_mode == "hardlink":
        destination.hardlink_to(source)
    elif copy_mode == "symlink":
        destination.symlink_to(source)
    else:
        raise ValueError(f"unknown copy_mode: {copy_mode}")


def draw_line_preview(image_path, line_polygons, line_boxes, output_path, max_side=1600):
    image = Image.open(image_path).convert("RGB")
    scale = 1.0
    if max_side and max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((int(round(image.width * scale)), int(round(image.height * scale))), BILINEAR)

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")

    for boxes in line_boxes:
        for box in boxes:
            if box.get("points") and len(box["points"]) >= 3:
                points = [(x * scale, y * scale) for x, y in box["points"]]
                draw.line(points + [points[0]], fill=(80, 140, 255, 140), width=1)
            else:
                rect = [box["x1"] * scale, box["y1"] * scale, box["x2"] * scale, box["y2"] * scale]
                draw.rectangle(rect, outline=(80, 140, 255, 130), width=1)

    for index, polygon in enumerate(line_polygons, start=1):
        points = [(x * scale, y * scale) for x, y in polygon]
        if len(points) < 3:
            continue
        draw.polygon(points, fill=(40, 220, 80, 45))
        draw.line(points + [points[0]], fill=(40, 220, 80, 230), width=2)
        x_min = min(x for x, _ in points)
        y_min = min(y for _, y in points)
        draw.rectangle([x_min, y_min, x_min + 44, y_min + 18], fill=(40, 220, 80, 210))
        draw.text((x_min + 4, y_min + 1), str(index), fill=(0, 0, 0, 255))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)


def prepare_coco_lines(
    coco_json,
    image_dir,
    output_img_dir,
    output_gt_dir,
    preview_dir=None,
    preview_count=0,
    preview_max_side=1600,
    copy_mode="copy",
    min_y_overlap=0.35,
    center_y_tolerance=0.6,
    line_group_mode="auto",
    polygon_padding=0.0,
    hull_method="convex",
    concavity=2.0,
    length_threshold=0.0,
    alpha=1.5,
    prevent_intersections=True,
    non_overlap_gap=0.0,
    min_polygon_area=4.0,
    simplify_tolerance=0.0,
    max_hull_points=None,
    min_box_area=1.0,
    include_category_names=None,
    include_category_ids=None,
    exclude_file_names=None,
    exclude_list_path=None,
    start_index=0,
    max_images=None,
    ignore_crowd=True,
    progress_interval=0,
):
    coco = load_coco(coco_json)
    image_dirs = image_search_dirs(image_dir)
    output_img_dir = Path(output_img_dir)
    output_gt_dir = Path(output_gt_dir)
    category_ids = build_category_id_filter(
        coco.get("categories", []),
        include_category_names=include_category_names,
        include_category_ids=include_category_ids,
    )
    excluded_file_names = load_excluded_file_names(exclude_file_names, exclude_list_path)
    annotations_by_image = group_annotations_by_image(
        coco.get("annotations", []),
        ignore_crowd=ignore_crowd,
        category_ids=category_ids,
    )

    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_gt_dir.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        Path(preview_dir).mkdir(parents=True, exist_ok=True)

    summary = {
        "images": 0,
        "missing_images": 0,
        "excluded_images": 0,
        "start_index": int(start_index or 0),
        "requested_images": None,
        "boxes": 0,
        "lines": 0,
        "gt_files": 0,
        "previews": 0,
        "group_id_lines": 0,
        "geometry_lines": 0,
        "annotations_with_group_id": 0,
        "skipped_annotations": 0,
        "intersections_removed": 0,
        "invalid_polygons": 0,
        "dropped_polygons": 0,
        "remaining_intersections": 0,
        "max_remaining_overlap_area": 0.0,
        "category_filter_ids": sorted(category_ids) if category_ids is not None else None,
    }
    previews_written = 0
    images = coco.get("images", [])
    if start_index:
        images = images[int(start_index):]
    if max_images is not None:
        images = images[: int(max_images)]
        summary["requested_images"] = int(max_images)

    for image_index, image_info in enumerate(images, start=1):
        if progress_interval and (image_index == 1 or image_index % progress_interval == 0 or image_index == len(images)):
            print(f"prepare {Path(coco_json).name}: {image_index}/{len(images)} {image_info['file_name']}", flush=True)
        file_name = image_info["file_name"]
        if Path(file_name).name in excluded_file_names:
            summary["excluded_images"] += 1
            continue
        image_path = find_image_path(image_dirs, file_name)
        if not image_path.exists():
            summary["missing_images"] += 1
            print(f"missing image: {image_path}")
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        items, skipped_annotations = annotation_items_for_image(
            annotations_by_image.get(image_info["id"], []),
            min_box_area=min_box_area,
        )
        lines, group_stats = group_items_into_lines(
            items,
            min_y_overlap=min_y_overlap,
            center_y_tolerance=center_y_tolerance,
            line_group_mode=line_group_mode,
        )
        image_size = (int(image_info.get("width") or 0), int(image_info.get("height") or 0))
        if image_size[0] <= 0 or image_size[1] <= 0:
            with Image.open(image_path) as image:
                image_size = image.size
        polygons, preview_lines, polygon_stats = build_line_polygons(
            lines,
            image_size,
            polygon_padding=polygon_padding,
            hull_method=hull_method,
            concavity=concavity,
            length_threshold=length_threshold,
            alpha=alpha,
            prevent_intersections=prevent_intersections,
            non_overlap_gap=non_overlap_gap,
            min_polygon_area=min_polygon_area,
            simplify_tolerance=simplify_tolerance,
            max_hull_points=max_hull_points,
        )

        destination_image = output_img_dir / Path(file_name).name
        copy_image(image_path, destination_image, copy_mode)

        gt_path = output_gt_dir / f"gt_{Path(file_name).stem}.txt"
        gt_lines = []
        for polygon in polygons:
            gt_line = polygon_to_gt_line(polygon, max_points=max_hull_points)
            if gt_line:
                gt_lines.append(gt_line)
        gt_path.write_text("\n".join(gt_lines), encoding="utf-8")

        if preview_dir is not None and previews_written < preview_count:
            preview_path = Path(preview_dir) / f"{Path(file_name).stem}_lines.png"
            draw_line_preview(image_path, polygons, preview_lines, preview_path, max_side=preview_max_side)
            previews_written += 1
            summary["previews"] += 1

        summary["images"] += 1
        summary["boxes"] += len(items)
        summary["lines"] += len(polygons)
        summary["gt_files"] += 1
        summary["group_id_lines"] += group_stats["group_id_lines"]
        summary["geometry_lines"] += group_stats["geometry_lines"]
        summary["annotations_with_group_id"] += group_stats["annotations_with_group_id"]
        summary["skipped_annotations"] += skipped_annotations
        summary["intersections_removed"] += polygon_stats["intersections_removed"]
        summary["invalid_polygons"] += polygon_stats["invalid_polygons"]
        summary["dropped_polygons"] += polygon_stats["dropped_polygons"]
        summary["remaining_intersections"] += polygon_stats["remaining_intersections"]
        summary["max_remaining_overlap_area"] = max(
            summary["max_remaining_overlap_area"],
            polygon_stats["max_remaining_overlap_area"],
        )
        if image_index % 50 == 0:
            gc.collect()

    return summary
