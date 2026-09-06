#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create one horizontal Input | Prediction | Ground Truth image per slice.

Example:
    python make_cine_pred_lge_triptychs.py \
        --pred-dir /mnt/vd-r5/data1/data1/Weiren/data_all/CMR_HKU/ALL \
        --input-dir /mnt/vd-r5/data1/data1/Weiren/data_all/CMR_HKU/test/Cine \
        --gt-dir /mnt/vd-r5/data1/data1/Weiren/data_all/CMR_HKU/test/LGE \
        --output-dir /mnt/vd-r5/data1/data1/Weiren/data_all/CMR_HKU/triptychs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# Pillow < 9.1 has Image.BILINEAR but no Image.Resampling namespace.
BILINEAR = (
    Image.Resampling.BILINEAR
    if hasattr(Image, "Resampling")
    else Image.BILINEAR
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Horizontally arrange cardiac MRI Input, Prediction, and GT images."
    )
    parser.add_argument("--pred-dir", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--gt-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*.png")
    parser.add_argument(
        "--panel-size",
        type=int,
        default=512,
        help="Width and height of each square panel (default: 512)",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=0,
        help="Black gap in pixels between panels (default: 0)",
    )
    parser.add_argument(
        "--labels",
        action="store_true",
        help="Add Input, Prediction, and Ground Truth labels above the images",
    )
    parser.add_argument(
        "--label-height",
        type=int,
        default=40,
        help="Label area height when --labels is enabled (default: 40)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any filename is not present in all three directories",
    )
    parser.add_argument(
        "--max-error-examples",
        type=int,
        default=20,
        help="Print the first N processing errors immediately (default: 20)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first image-processing error and show its traceback",
    )
    return parser.parse_args()


def index_images(folder: Path, pattern: str):
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {folder}")
    indexed = {}
    for path in sorted(folder.glob(pattern)):
        if not path.is_file():
            continue
        if path.name in indexed:
            raise ValueError(f"Duplicate filename: {path.name}")
        indexed[path.name] = path
    return indexed


def load_panel(path: Path, size: int):
    """Load as grayscale and resize without changing intensity normalization."""
    with Image.open(path) as image:
        panel = image.convert("L")
        if panel.size != (size, size):
            panel = panel.resize((size, size), BILINEAR)
        else:
            panel = panel.copy()
    return panel


def make_triptych(input_path, pred_path, gt_path, panel_size, gap, labels, label_height):
    panels = [
        load_panel(input_path, panel_size),
        load_panel(pred_path, panel_size),
        load_panel(gt_path, panel_size),
    ]
    top = label_height if labels else 0
    width = panel_size * 3 + gap * 2
    height = panel_size + top
    canvas = Image.new("L", (width, height), color=0)
    x_positions = [0, panel_size + gap, panel_size * 2 + gap * 2]

    for panel, x in zip(panels, x_positions):
        canvas.paste(panel, (x, top))

    if labels:
        draw = ImageDraw.Draw(canvas)
        # No size argument keeps compatibility with older Pillow versions.
        font = ImageFont.load_default()
        for label, x in zip(("Input Cine", "Prediction", "GT LGE"), x_positions):
            box = draw.textbbox((0, 0), label, font=font)
            text_width = box[2] - box[0]
            draw.text(
                (x + (panel_size - text_width) // 2, max(0, (label_height - 22) // 2)),
                label,
                fill=255,
                font=font,
            )
    return canvas


def main():
    args = parse_args()
    if args.panel_size <= 0 or args.gap < 0 or args.label_height < 0:
        raise ValueError("panel-size must be positive; gap and label-height cannot be negative")

    pred_dir = args.pred_dir.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve()
    gt_dir = args.gt_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pred = index_images(pred_dir, args.pattern)
    input_images = index_images(input_dir, args.pattern)
    gt = index_images(gt_dir, args.pattern)
    pred_names, input_names, gt_names = set(pred), set(input_images), set(gt)
    paired_names = sorted(pred_names & input_names & gt_names)

    missing_pred = sorted((input_names & gt_names) - pred_names)
    missing_input = sorted((pred_names & gt_names) - input_names)
    missing_gt = sorted((pred_names & input_names) - gt_names)
    if args.strict and (missing_pred or missing_input or missing_gt):
        raise RuntimeError(
            f"Unpaired files: missing_pred={len(missing_pred)}, "
            f"missing_input={len(missing_input)}, missing_gt={len(missing_gt)}"
        )

    failed = []
    written = 0
    for index, name in enumerate(paired_names, start=1):
        try:
            triptych = make_triptych(
                input_images[name],
                pred[name],
                gt[name],
                args.panel_size,
                args.gap,
                args.labels,
                args.label_height,
            )
            triptych.save(output_dir / name, format="PNG")
            written += 1
        except Exception as exc:
            failed.append({"filename": name, "error": repr(exc)})
            if len(failed) <= args.max_error_examples:
                print(f"[ERROR {len(failed)}] {name}: {exc!r}", flush=True)
            if args.fail_fast:
                raise
        if index % 500 == 0 or index == len(paired_names):
            print(f"[PROGRESS] {index}/{len(paired_names)}; written={written}; failed={len(failed)}")

    summary = {
        "order": ["Input Cine", "Prediction", "GT LGE"],
        "input_dir": str(input_dir),
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir),
        "output_dir": str(output_dir),
        "input_images": len(input_images),
        "prediction_images": len(pred),
        "gt_images": len(gt),
        "matched_filenames": len(paired_names),
        "written": written,
        "failed": failed,
        "missing_prediction_count": len(missing_pred),
        "missing_input_count": len(missing_input),
        "missing_gt_count": len(missing_gt),
        "missing_prediction_examples": missing_pred[:50],
        "missing_input_examples": missing_input[:50],
        "missing_gt_examples": missing_gt[:50],
        "panel_size": args.panel_size,
        "gap": args.gap,
        "labels": args.labels,
    }
    (output_dir / "triptych_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[DONE] Wrote {written} triptychs to {output_dir}")


if __name__ == "__main__":
    main()
