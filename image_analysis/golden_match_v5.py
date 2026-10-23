#!/usr/bin/env python3
"""Find which golden reference, if any, a candidate painting photo matches.

The vault holds SIFT features for each golden, computed on several channels:
grey, red, green, blue, and a red chromaticity channel that is less sensitive
to white balance. Each channel has its own FLANN index.

For a candidate, every channel is queried. A golden's fused score is its best
RANSAC inlier count across channels. A golden is accepted only when its fused
score clears MIN_INLIERS and stands clear of the runner-up. A template score
(normalised cross-correlation of the golden inside the candidate) is reported
for the top golden, for display only.

Several shots of the same artwork can be searched together. The golden
accepted on at least half of the shots is reported as the match.

Usage:
    python golden_match_v5.py build-vault GOLDEN_DIR VAULT_DIR [--json]
    python golden_match_v5.py search VAULT_DIR CANDIDATE_OR_FOLDER [...] [--json]
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys
from collections import Counter

import cv2
import numpy as np

IMAGE_EXTENSIONS = ("*.bmp", "*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")
CHANNELS = ("grey", "red", "green", "blue", "chroma")
WORK_SIDE = 1000  # longer side of every image before feature extraction
GOLDEN_FEATURES = 2000
CANDIDATE_FEATURES = 4000
CIRCLE_RADIUS = 0.48  # golden feature mask radius, as a fraction of the side
RATIO = 0.8  # Lowe ratio test for descriptor matches
TOP_VOTES = 10  # goldens per channel whose votes are checked geometrically
MIN_VOTES = 8  # a golden needs at least this many ratio-test matches
RANSAC_PX = 5.0
MIN_INLIERS = 50  # fused inlier count a golden needs to be accepted
MARGIN = 1.5  # top golden must have this many times the runner-up's inliers
SHOT_FRACTION = 0.5  # a golden must win on at least this share of shots
FLANN_PARAMS = {"algorithm": 1, "trees": 4}  # KD-tree index
FLANN_CHECKS = {"checks": 128}
VAULT_FILE = "vault.npz"
TEMPLATE_BLUR = (5, 5)
TEMPLATE_SCALES = (0.4, 1.3)  # golden diameter as a fraction of the candidate's short side
TEMPLATE_STEPS = 19
TEMPLATE_PAD = 0.6  # padding on each side, as a fraction of the candidate's short side
TEMPLATE_MIN_PX = 80
NCC_FLOOR = 0.35  # template NCC at or below this scores 0
NCC_CEILING = 0.90  # template NCC at or above this scores 90


def index_file(vault_dir, channel):
    return os.path.join(vault_dir, f"vault_{channel}.flann")


def read_bgr(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def channel_images(bgr):
    b, g, r = cv2.split(bgr)
    # Red share of total intensity: depends on hue, not on overall brightness or white balance
    total = bgr.astype(np.float32).sum(axis=2) + 1.0
    chroma = cv2.normalize(bgr[:, :, 2] / total, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return {
        "grey": cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
        "red": r,
        "green": g,
        "blue": b,
        "chroma": chroma,
    }


def resize_longer_side(bgr, side):
    h, w = bgr.shape[:2]
    factor = side / max(h, w)
    return cv2.resize(bgr, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)


def circle_mask(shape):
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(CIRCLE_RADIUS * min(h, w)), 255, -1)
    return mask


def sift_features(grey, mask, n_features):
    kps, desc = cv2.SIFT_create(nfeatures=n_features).detectAndCompute(grey, mask)
    if desc is None or len(kps) < 2:
        return np.empty((0, 2), np.float32), np.empty((0, 128), np.float32)
    # RootSIFT: L1-normalise, then square-root, which suits Euclidean matching
    desc = desc / (desc.sum(axis=1, keepdims=True) + 1e-7)
    desc = np.sqrt(desc).astype(np.float32)
    return np.float32([k.pt for k in kps]), desc


def unique_golden_paths(golden_dir):
    paths = []
    for pattern in IMAGE_EXTENSIONS:
        paths += glob.glob(os.path.join(golden_dir, pattern))
    kept, seen = [], set()
    for path in sorted(paths):
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        if digest in seen:
            print(f"skipped (byte-identical duplicate): {path}", file=sys.stderr)
            continue
        seen.add(digest)
        kept.append(os.path.abspath(path))
    return kept


def build_vault(golden_dir, vault_dir, progress=True):
    paths = unique_golden_paths(golden_dir)
    if not paths:
        raise ValueError(f"No golden images found in {golden_dir}")

    parts = {ch: {"xy": [], "desc": [], "owner": []} for ch in CHANNELS}
    kept = []
    for path in paths:
        img = resize_longer_side(read_bgr(path), WORK_SIDE)
        mask = circle_mask(img.shape[:2])
        owner_id = len(kept)
        kept.append(path)
        for ch, grey in channel_images(img).items():
            xy, desc = sift_features(grey, mask, GOLDEN_FEATURES)
            parts[ch]["xy"].append(xy)
            parts[ch]["desc"].append(desc)
            parts[ch]["owner"].append(np.full(len(desc), owner_id, np.int32))
        if progress:
            print(f"indexed: {os.path.basename(path)}", file=sys.stderr)

    os.makedirs(vault_dir, exist_ok=True)
    arrays = {"paths": np.array(kept)}
    for ch in CHANNELS:
        xy = np.vstack(parts[ch]["xy"])
        desc = np.vstack(parts[ch]["desc"])
        owner = np.concatenate(parts[ch]["owner"])
        arrays[f"{ch}_xy"], arrays[f"{ch}_desc"], arrays[f"{ch}_owner"] = xy, desc, owner
        cv2.flann_Index(desc, FLANN_PARAMS).save(index_file(vault_dir, ch))
        if progress:
            print(f"channel {ch}: {len(desc)} features", file=sys.stderr)
    np.savez(os.path.join(vault_dir, VAULT_FILE), **arrays)
    return {
        "vault_dir": os.path.abspath(vault_dir),
        "goldens": [os.path.basename(p) for p in kept],
        "features": {ch: int(sum(len(d) for d in parts[ch]["desc"])) for ch in CHANNELS},
    }


def load_vault(vault_dir):
    path = os.path.join(vault_dir, VAULT_FILE)
    if not os.path.exists(path):
        raise ValueError(f"No vault found in {vault_dir}: run build-vault first")
    data = np.load(path)
    indexes = {}
    for ch in CHANNELS:
        index = cv2.flann_Index()
        index.load(data[f"{ch}_desc"], index_file(vault_dir, ch))
        indexes[ch] = index
    return data, indexes


def channel_inliers(data, index, ch, c_xy, c_desc):
    """Return an inlier count per golden for one channel, zero where a golden is not shortlisted."""
    n_goldens = len(data["paths"])
    inliers = np.zeros(n_goldens, np.int32)
    if len(c_desc) < 2:
        return inliers
    idx, dist = index.knnSearch(c_desc, 2, params=FLANN_CHECKS)
    good = np.sqrt(dist[:, 0]) < RATIO * np.sqrt(dist[:, 1])
    q_idx = np.nonzero(good)[0]
    t_idx = idx[good, 0]
    owners = data[f"{ch}_owner"][t_idx]
    votes = np.bincount(owners, minlength=n_goldens)

    for g in np.argsort(votes)[::-1][:TOP_VOTES]:
        if votes[g] < MIN_VOTES:
            break
        sel = owners == g
        _, mask = cv2.findHomography(data[f"{ch}_xy"][t_idx[sel]], c_xy[q_idx[sel]], cv2.RANSAC, RANSAC_PX)
        inliers[g] = int(mask.sum()) if mask is not None else 0
    return inliers


def shot_features(path):
    img = resize_longer_side(read_bgr(path), WORK_SIDE)
    return {ch: sift_features(grey, None, CANDIDATE_FEATURES) for ch, grey in channel_images(img).items()}


def preprocess_template(bgr, test):
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if test == "edge":
        blurred = cv2.GaussianBlur(grey, TEMPLATE_BLUR, 0).astype(np.float32)
        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if test == "clahe":
        grey = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grey)
    return cv2.GaussianBlur(grey, TEMPLATE_BLUR, 0)


def circular_template(img, side):
    # Resize the golden to a square and fill everything outside its inscribed circle with the mean
    tpl = cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA)
    mask = np.zeros((side, side), np.uint8)
    cv2.circle(mask, (side // 2, side // 2), side // 2, 255, -1)
    tpl[mask == 0] = int(tpl[mask > 0].mean())
    return tpl


def template_ncc(golden_bgr, candidate_bgr):
    """Median over three preprocessings of the best NCC of the golden as a circular template in the padded candidate."""
    h, w = candidate_bgr.shape[:2]
    factor = 800 / max(h, w)
    candidate_bgr = cv2.resize(candidate_bgr, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    short_side = min(candidate_bgr.shape[:2])
    pad = int(TEMPLATE_PAD * short_side)
    per_test = []
    for test in ("grey", "edge", "clahe"):
        gp = preprocess_template(golden_bgr, test)
        cp = preprocess_template(candidate_bgr, test)
        padded = cv2.copyMakeBorder(cp, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=int(cp.mean()))
        best = -1.0
        lo, hi = TEMPLATE_SCALES
        for frac in np.linspace(lo, hi, TEMPLATE_STEPS):
            side = int(round(frac * short_side))
            if side < TEMPLATE_MIN_PX:
                continue
            tpl = circular_template(gp, side)
            ncc_map = cv2.matchTemplate(padded, tpl, cv2.TM_CCOEFF_NORMED)
            best = max(best, float(cv2.minMaxLoc(ncc_map)[1]))
        per_test.append(best)
    return float(np.median(per_test))


def template_score_percent(golden_path, candidate_path):
    fraction = (template_ncc(read_bgr(golden_path), read_bgr(candidate_path)) - NCC_FLOOR) / (NCC_CEILING - NCC_FLOOR)
    return round(90.0 * min(1.0, max(0.0, fraction)), 1)


def search_shot(data, indexes, candidate_path):
    features = shot_features(candidate_path)
    per_channel = {ch: channel_inliers(data, indexes[ch], ch, *features[ch]) for ch in CHANNELS}
    fused = np.max(np.vstack(list(per_channel.values())), axis=0)

    order = np.argsort(fused)[::-1]
    top = int(order[0])
    top_in = int(fused[top])
    second_in = int(fused[int(order[1])]) if len(order) > 1 else 0
    accepted = top_in >= MIN_INLIERS and top_in >= MARGIN * max(second_in, 1)

    top_golden = str(data["paths"][top])
    return {
        "candidate": os.path.basename(candidate_path),
        "match": os.path.basename(top_golden) if accepted else None,
        "golden": top_golden,
        "inliers": top_in,
        "runner_up_inliers": second_in,
        "template_score_percent": template_score_percent(top_golden, candidate_path),
        "channel_inliers": {ch: int(per_channel[ch][top]) for ch in CHANNELS},
    }


def expand_candidates(paths):
    """Expand any folder arguments to the image files inside them, sorted by name."""
    shots = []
    for path in paths:
        if os.path.isdir(path):
            found = []
            for pattern in IMAGE_EXTENSIONS:
                found += glob.glob(os.path.join(path, pattern))
            shots += sorted(found)
        else:
            shots.append(path)
    if not shots:
        raise ValueError("No candidate images found")
    return shots


def search(vault_dir, candidate_paths):
    data, indexes = load_vault(vault_dir)
    shots = [search_shot(data, indexes, p) for p in expand_candidates(candidate_paths)]

    wins = Counter(s["match"] for s in shots if s["match"])
    needed = math.ceil(len(shots) * SHOT_FRACTION)
    consensus = None
    if wins:
        name, count = wins.most_common(1)[0]
        if count >= needed:
            consensus = name
    return {"match": consensus, "shots": shots}


def print_result(result):
    print(f"match: {result['match'] or 'none'}")
    for s in result["shots"]:
        status = s["match"] or "no match"
        print(f"  {s['candidate']}: {status}")
        print(f"    top golden: {os.path.basename(s['golden'])}")
        print(f"    inliers: {s['inliers']} (runner-up {s['runner_up_inliers']}) "
              f"template score: {s['template_score_percent']}%")
        channels = " ".join(f"{ch}={n}" for ch, n in s["channel_inliers"].items())
        print(f"    per channel: {channels}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Match candidate painting photos against golden references.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-vault", help="index a folder of golden images")
    p_build.add_argument("golden_dir")
    p_build.add_argument("vault_dir")
    p_build.add_argument("--json", action="store_true", help="print the build summary as JSON, progress hidden")

    p_search = sub.add_parser("search", help="find the golden one or more candidate shots match")
    p_search.add_argument("vault_dir")
    p_search.add_argument("candidates", nargs="+", help="image files and/or folders of images")
    p_search.add_argument("--json", action="store_true", help="print the result as JSON")

    args = parser.parse_args(argv)
    try:
        if args.command == "build-vault":
            summary = build_vault(args.golden_dir, args.vault_dir, progress=not args.json)
            if args.json:
                print(json.dumps(summary, indent=2))
            else:
                print(f"vault: {len(summary['goldens'])} goldens -> {summary['vault_dir']}")
            return 0
        result = search(args.vault_dir, args.candidates)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
