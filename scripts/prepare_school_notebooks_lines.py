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
        description="Convert school_notebooks_RU train/val/test COCO annotations into line-level gt folders."
    )
    parser.add_argument("--dataset-dir", default=r"C:\shared\data0205\data02065\school_notebooks_RU")
    parser.add_argument("--output-dir", default="school_notebooks_RU_lines")
    parser.add_argument("--copy-mode", choices=["copy", "hardlink", "symlink", "none"], default="hardlink")
    parser.add_argument("--line-group-mode", choices=["auto", "group-id", "geometry"], default="auto")
    parser.add_argument("--include-category-names", default="pupil_text;pupil_comment")
    parser.add_argument("--include-category-ids", default=None)
    parser.add_argument("--min-y-overlap", type=float, default=0.35)
    parser.add_argument("--center-y-tolerance", type=float, default=0.6)
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
    parser.add_argument("--preview-count", type=int, default=3)
    parser.add_argument("--preview-max-side", type=int, default=1600)
    parser.add_argument("--splits", default="train;val;test")
    parser.add_argument("--exclude-file-names", default=None)
    parser.add_argument("--exclude-list", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--include-crowd", action="store_true")
    return parser.parse_args()


def convert_split(args, split_name):
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    summary = prepare_coco_lines(
        coco_json=dataset_dir / f"annotations_{split_name}.json",
        image_dir=f"{dataset_dir / 'train_images'};{dataset_dir / 'test_images'}",
        output_img_dir=output_dir / f"{split_name}_img",
        output_gt_dir=output_dir / f"{split_name}_gt",
        preview_dir=output_dir / "previews" / split_name,
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
    print(f"{split_name}:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main():
    args = parse_args()
    split_names = [item.strip() for item in args.splits.split(";") if item.strip()]
    summaries = {split_name: convert_split(args, split_name) for split_name in split_names}
    print("output:")
    print(Path(args.output_dir))
    print("combined:")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
