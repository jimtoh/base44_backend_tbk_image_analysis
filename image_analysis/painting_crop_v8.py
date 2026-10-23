#!/usr/bin/env python3
"""Find a painting in a photo, crop it out, and flatten its perspective.

A central box prompt is passed to Segment Anything (ViT-B), which returns the
painting region. A quadrilateral is fitted to that region; each side is a line
fitted to the region's hull points, and the four lines are intersected to give
the corners. Each fit is checked against its mask before it is accepted. Corners
can also be given as an override.

Given a photo, writes <photo>_painting_v8.<format> next to the photo.
Given a folder, processes every image in it (not subfolders) and writes the
crops to <folder>/output/. The SAM model is loaded once for the whole folder.
The output is rectified to face the camera and saved losslessly as TIFF or BMP.

Usage:
    python painting_crop_v8.py PHOTO_OR_FOLDER [--format tiff|bmp] [corners]

TMS: Use this
    python3 image_analysis/painting_crop_v8.py client_pre_images

    For BMP output:

    ```bash
    python3 image_analysis/painting_crop_v8.py client_pre_images --format bmp

"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "sam_vit_b_01ec64.pth"
BOX_FRACTIONS = [(0.2, 0.2, 0.8, 0.8), (0.1, 0.1, 0.9, 0.9), (0.3, 0.3, 0.7, 0.7)]  # central box prompts, tried in turn
MIN_AREA_FRACTION = 0.10  # painting must cover at least 10% of the photo
MAX_AREA_FRACTION = 0.98  # ...and not be the whole photo
MIN_MASK_IOU = 0.80  # the fitted quadrilateral must overlap the mask at least this much
APPROX_EPSILONS = np.linspace(0.005, 0.10, 20)  # polygon simplification sweep, relative to perimeter
REFINE_ITERATIONS = 5  # rounds of re-assigning hull points to sides and refitting
TIFF_LZW = 5  # lossless TIFF compression
CORNER_NAMES = ["top_left", "top_right", "bottom_right", "bottom_left"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
OUTPUT_DIRNAME = "output"  # created inside the input folder in folder mode


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def order_corners(points):
    """Return corners as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = pts[:, 0] - pts[:, 1]
    return np.array(
        [
            pts[np.argmin(sums)],  # top-left: smallest x + y
            pts[np.argmax(diffs)],  # top-right: largest x - y
            pts[np.argmax(sums)],  # bottom-right: largest x + y
            pts[np.argmin(diffs)],  # bottom-left: smallest x - y
        ],
        dtype=np.float32,
    )


def parse_corners(text):
    """Parse 'x1,y1 x2,y2 x3,y3 x4,y4' into a (4, 2) array in the given order."""
    pairs = text.split()
    if len(pairs) != 4:
        raise ValueError("corners need four points: 'x,y x,y x,y x,y'")
    return np.array([[float(v) for v in pair.split(",")] for pair in pairs], dtype=np.float32)


def validate_quad(corners, image_shape):
    """Reject corners that are off-image or not convex."""
    height, width = image_shape[:2]
    if np.any(corners < 0) or np.any(corners[:, 0] >= width) or np.any(corners[:, 1] >= height):
        raise ValueError("corners must lie inside the image")
    if not cv2.isContourConvex(corners.reshape(-1, 1, 2).astype(np.float32)):
        raise ValueError("corners do not form a convex quadrilateral; check the order (TL, TR, BR, BL)")


def line_intersection(line_a, line_b):
    """Intersection of two (point, direction) lines, or None if they are parallel."""
    (p1, d1), (p2, d2) = line_a, line_b
    matrix = np.column_stack([d1, -d2])
    if abs(np.linalg.det(matrix)) < 1e-6:
        return None
    t, _ = np.linalg.solve(matrix, p2 - p1)
    return p1 + t * d1


def distance_to_line(points, line):
    point, direction = line
    normal = np.array([-direction[1], direction[0]])
    return np.abs((points - point) @ normal)


def initial_quad(hull):
    """Start from the simplest four-vertex polygon around the hull, else its minimum-area rectangle."""
    perimeter = cv2.arcLength(hull, True)
    for epsilon in APPROX_EPSILONS:
        approx = cv2.approxPolyDP(hull, epsilon * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)


def fit_quad(mask):
    """Fit a four-sided outline to a binary mask; return corners in TL, TR, BR, BL order, or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea)).reshape(-1, 2).astype(np.float32)
    corners = order_corners(initial_quad(hull.reshape(-1, 1, 2)))

    for _ in range(REFINE_ITERATIONS):
        # Edge i joins corner i and corner i+1 (mod 4).
        edges = [(corners[i], corners[(i + 1) % 4] - corners[i]) for i in range(4)]
        distances = np.column_stack([distance_to_line(hull, (p, d / np.linalg.norm(d))) for p, d in edges])
        owner = np.argmin(distances, axis=1)
        lines = []
        for i in range(4):
            pts = hull[owner == i]
            if len(pts) < 2:
                return None
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
            lines.append((np.array([x0, y0], dtype=np.float64), np.array([vx, vy], dtype=np.float64)))
        # Corner i is where edge i-1 meets edge i.
        new = [line_intersection(lines[(i - 1) % 4], lines[i]) for i in range(4)]
        if any(c is None for c in new):
            return None
        corners = np.array(new, dtype=np.float32)

    return corners


def quad_is_acceptable(corners, mask):
    """Check that the quadrilateral is convex, plausibly sized, and overlaps the mask well."""
    height, width = mask.shape
    if not cv2.isContourConvex(corners.reshape(-1, 1, 2)):
        return False
    area_fraction = cv2.contourArea(corners) / (width * height)
    if not MIN_AREA_FRACTION <= area_fraction <= MAX_AREA_FRACTION:
        return False
    quad_mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(quad_mask, [np.round(corners).astype(np.int32)], 1)
    intersection = np.logical_and(quad_mask, mask).sum()
    union = np.logical_or(quad_mask, mask).sum()
    return union > 0 and intersection / union >= MIN_MASK_IOU


def segment_candidates(predictor, image_shape):
    """Yield (box, mask, score) for each box prompt and SAM mask, best score first within each box."""
    height, width = image_shape[:2]
    for fx0, fy0, fx1, fy1 in BOX_FRACTIONS:
        box = np.array([width * fx0, height * fy0, width * fx1, height * fy1])
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
        for index in np.argsort(scores)[::-1]:
            yield box, masks[index].astype(np.uint8), float(scores[index])


def detect_painting(image, predictor):
    """Return (corners, method) for the painting in full-resolution coordinates.

    Runs SAM on the image once, then tries each box prompt and mask in turn
    until one gives a quadrilateral that passes the checks.
    """
    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for box, mask, score in segment_candidates(predictor, image.shape):
        corners = fit_quad(mask)
        if corners is not None and quad_is_acceptable(corners, mask):
            return order_corners(corners), f"sam_box_{score:.2f}"
    raise ValueError("no painting outline passed the checks; the photo may be too unusual for automatic detection")


def rectify(image, corners):
    """Warp the quadrilateral to an upright rectangle."""
    tl, tr, br, bl = corners
    width = round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    target = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, target)
    return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_CUBIC)


def write_lossless(image, path, fmt):
    params = [cv2.IMWRITE_TIFF_COMPRESSION, TIFF_LZW] if fmt == "tiff" else []
    if not cv2.imwrite(str(path), image, params):
        raise OSError(f"failed to write {path}")


def build_predictor(model_path):
    if not model_path.exists():
        raise ValueError(f"model not found: {model_path}")
    sam = sam_model_registry["vit_b"](checkpoint=str(model_path))
    sam.to("cpu").eval()
    return SamPredictor(sam)


def crop_painting(input_path, output_path, fmt, predictor, manual_corners=None):
    image = load_image(input_path)
    if manual_corners is not None:
        validate_quad(manual_corners, image.shape)
        corners, method = order_corners(manual_corners), "manual"
    else:
        with torch.no_grad():
            corners, method = detect_painting(image, predictor)
    rectified = rectify(image, corners)
    write_lossless(rectified, output_path, fmt)

    x_min, y_min = np.floor(corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(corners.max(axis=0)).astype(int)
    height, width = rectified.shape[:2]
    return {
        "input": str(input_path),
        "output": str(output_path),
        "format": fmt,
        "method": method,
        "corners": {
            "order": CORNER_NAMES,
            "points": corners.round(1).tolist(),
        },
        "bbox": {"x": int(x_min), "y": int(y_min), "width": int(x_max - x_min), "height": int(y_max - y_min)},
        "output_size": {"width": int(width), "height": int(height)},
    }


def image_paths(folder):
    """Image files directly inside the folder, sorted by name. Subfolders are skipped."""
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def crop_folder(folder, fmt, predictor):
    """Crop every image in the folder into folder/output. Returns (reports, failures)."""
    paths = image_paths(folder)
    if not paths:
        raise ValueError(f"no images found in folder: {folder}")
    output_dir = folder / OUTPUT_DIRNAME
    output_dir.mkdir(exist_ok=True)

    reports, failures = [], []
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path.name}", file=sys.stderr)
        output = output_dir / f"{path.stem}_painting_v8.{fmt}"
        try:
            reports.append(crop_painting(path, output, fmt, predictor))
        except (ValueError, OSError) as error:
            failures.append({"input": str(path), "error": str(error)})
            print(f"error: {path.name}: {error}", file=sys.stderr)
    return reports, failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="photo of the painting, or a folder of photos")
    parser.add_argument("--format", choices=["tiff", "bmp"], default="tiff", help="lossless output format")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="SAM ViT-B checkpoint")
    parser.add_argument(
        "corners",
        nargs="?",
        help="optional override for a single photo: outer corners in input pixels, TL TR BR BL; omit for automatic detection",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    try:
        if args.input.is_dir():
            if args.corners:
                raise ValueError("corner override works on a single photo, not a folder")
            reports, failures = crop_folder(args.input, args.format, build_predictor(args.model))
            print(json.dumps({"reports": reports, "failures": failures}, indent=2))
            return 1 if failures else 0

        output = args.input.with_name(f"{args.input.stem}_painting_v8.{args.format}")
        manual = parse_corners(args.corners) if args.corners else None
        predictor = None if manual is not None else build_predictor(args.model)
        report = crop_painting(args.input, output, args.format, predictor, manual)
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
