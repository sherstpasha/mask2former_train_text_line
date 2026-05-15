import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.coco_lines import prepare_coco_lines


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert COCO word/box annotations into line-level polygon gt files."
    )
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", default="coco_lines_prepared")
    parser.add_argument("--split-name", default="train")
    parser.add_argument("--output-img-dir", default=None)
    parser.add_argument("--output-gt-dir", default=None)
    parser.add_argument("--preview-dir", default=None)
    parser.add_argument("--preview-count", type=int, default=4)
    parser.add_argument("--preview-max-side", type=int, default=1600)
    parser.add_argument("--copy-mode", choices=["copy", "hardlink", "symlink", "none"], default="copy")
    parser.add_argument("--min-y-overlap", type=float, default=0.35)
    parser.add_argument("--center-y-tolerance", type=float, default=0.6)
    parser.add_argument("--line-group-mode", choices=["auto", "group-id", "geometry"], default="auto")
    parser.add_argument("--polygon-padding", type=float, default=0.0)
    parser.add_argument("--hull-method", choices=["convex", "concave", "alpha"], default="convex")
    parser.add_argument("--concavity", type=float, default=2.0)
    parser.add_argument("--length-threshold", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--allow-intersections", action="store_true")
    parser.add_argument("--non-overlap-gap", type=float, default=2.0)
    parser.add_argument("--min-polygon-area", type=float, default=4.0)
    parser.add_argument("--simplify-tolerance", type=float, default=0.0)
    parser.add_argument("--max-hull-points", type=int, default=32)
    parser.add_argument("--min-box-area", type=float, default=1.0)
    parser.add_argument("--include-category-names", default=None)
    parser.add_argument("--include-category-ids", default=None)
    parser.add_argument("--exclude-file-names", default=None)
    parser.add_argument("--exclude-list", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--include-crowd", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_img_dir = Path(args.output_img_dir) if args.output_img_dir else output_dir / f"{args.split_name}_img"
    output_gt_dir = Path(args.output_gt_dir) if args.output_gt_dir else output_dir / f"{args.split_name}_gt"
    preview_dir = Path(args.preview_dir) if args.preview_dir else output_dir / "previews"

    summary = prepare_coco_lines(
        coco_json=args.coco_json,
        image_dir=args.image_dir,
        output_img_dir=output_img_dir,
        output_gt_dir=output_gt_dir,
        preview_dir=preview_dir,
        preview_count=args.preview_count,
        preview_max_side=args.preview_max_side,
        copy_mode=args.copy_mode,
        min_y_overlap=args.min_y_overlap,
        center_y_tolerance=args.center_y_tolerance,
        line_group_mode=args.line_group_mode,
        polygon_padding=args.polygon_padding,
        hull_method=args.hull_method,
        concavity=args.concavity,
        length_threshold=args.length_threshold,
        alpha=args.alpha,
        prevent_intersections=not args.allow_intersections,
        non_overlap_gap=args.non_overlap_gap,
        min_polygon_area=args.min_polygon_area,
        simplify_tolerance=args.simplify_tolerance,
        max_hull_points=args.max_hull_points,
        min_box_area=args.min_box_area,
        include_category_names=args.include_category_names,
        include_category_ids=args.include_category_ids,
        exclude_file_names=args.exclude_file_names,
        exclude_list_path=args.exclude_list,
        start_index=args.start_index,
        max_images=args.max_images,
        ignore_crowd=not args.include_crowd,
        progress_interval=args.progress_interval,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"images: {output_img_dir}")
    print(f"gt: {output_gt_dir}")
    print(f"previews: {preview_dir}")


if __name__ == "__main__":
    main()
