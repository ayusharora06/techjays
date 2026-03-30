"""
HVAC Duct Detection — Morphological Close + Contour Detection.

Pipeline:
  1. Render PDF → full-res image, crop title block
  2. Threshold darkest pixels → duct outlines only
  3. Directional morph close (fill duct gaps) + open (keep elongated strips)
  4. findContours → filter elongated rectangles → duct centerlines
  5. GPT-4o grid → pressure classification
  6. Draw colored centerlines on original image
"""

import logging
import os
import cv2
import numpy as np
import fitz
import math
import json
import re
import base64
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class DuctDetectorConfig:
    """All tunable parameters for duct detection, calibrated for 200dpi."""

    # PDF rendering
    dpi: int = 200
    crop_fraction: float = 0.65        # top portion of PDF that contains the drawing
    title_block_fraction: float = 0.15  # right portion excluded (title block)

    # Thresholding
    dark_threshold: int = 50            # only darkest pixels (duct outlines)

    # Morphological kernels (close fills gaps, open keeps elongated strips)
    morph_close_size: int = 70          # kernel length for closing gaps between walls
    morph_open_size: int = 80           # kernel length for keeping elongated strips

    # Contour filters
    min_contour_area: int = 1500
    min_aspect_ratio: float = 2.5
    min_duct_width: int = 10            # min short side (px)
    max_duct_width: int = 160           # max short side (px)
    min_duct_length: int = 80           # min long side (px)

    # Angle tolerance (degrees from axis-aligned)
    angle_tolerance: float = 15.0
    min_fill_ratio: float = 0.4         # min contour area / bounding rect area

    # Centerline
    centerline_shrink: float = 0.10     # shrink each end by this fraction
    line_thickness: int = 2             # drawn line thickness (px)

    # Deduplication
    dedup_proximity: int = 30           # max px distance for same-band check
    dedup_overlap_ratio: float = 0.5    # min overlap fraction to consider duplicate

    # GPT-4o grid
    grid_cols: int = 8
    grid_rows: int = 4
    grid_image_max_px: int = 2000       # downscale grid image for API
    gpt_model: str = "gpt-4o"
    gpt_max_tokens: int = 2048

    # Metadata matching
    metadata_max_distance_ratio: float = 0.3  # max dist as fraction of image size

    # Pressure classification thresholds (inches)
    pressure_high_min: float = 18.0
    pressure_medium_min: float = 12.0

    # Pixel-width → dimension estimation (at 200dpi)
    dim_thresholds: list = field(default_factory=lambda: [
        (120, '22"x14"'),
        (65,  '18"ø'),
        (40,  '14"ø'),
        (25,  '12"'),
        (0,   '8"ø'),
    ])


DEFAULT_CONFIG = DuctDetectorConfig()

PRESSURE_BGR = {
    "high": (0, 0, 255),       # Red
    "medium": (0, 140, 255),   # Orange
    "low": (255, 136, 68),     # Blue
}

PRESSURE_HEX = {
    "high": "#FF0000",
    "medium": "#FF8C00",
    "low": "#4488FF",
}

TYPE_LABELS = {
    "supply": "Supply Air",
    "return": "Return Air",
    "exhaust": "Exhaust",
    "grease": "Grease Exhaust",
}


# ─── Main entry point ────────────────────────────────────────────────────────

def detect_ducts(file_path: str, dpi: int = 200, config: DuctDetectorConfig = None):
    cfg = config or DuctDetectorConfig(dpi=dpi)

    is_pdf = file_path.lower().endswith(".pdf")
    if is_pdf:
        full_img = _render_pdf(file_path, cfg.dpi)
    else:
        full_img = cv2.imread(file_path)
        if full_img is None:
            raise ValueError(f"Cannot read: {file_path}")

    h, w = full_img.shape[:2]
    crop_h = int(h * cfg.crop_fraction) if is_pdf else h
    drawing = full_img[:crop_h]

    logger.info("Detecting duct rectangles...")
    ducts = _find_duct_rectangles(drawing, cfg)
    logger.info(f"Found {len(ducts)} ducts")

    metadata = []
    if os.getenv("OPENAI_API_KEY", ""):
        logger.info("Getting metadata from GPT-4o...")
        metadata = _get_duct_metadata(drawing, cfg)
        logger.info(f"Got {len(metadata)} metadata entries")

    result = full_img.copy()
    duct_data = []
    for i, d in enumerate(ducts):
        pressure = _match_pressure(d, metadata, crop_h, w, cfg)
        color_bgr = PRESSURE_BGR.get(pressure, PRESSURE_BGR["low"])
        color_hex = PRESSURE_HEX.get(pressure, PRESSURE_HEX["low"])
        cv2.line(result, (d["x1"], d["y1"]), (d["x2"], d["y2"]),
                 color_bgr, cfg.line_thickness)

        dimension, duct_type = _get_nearest_metadata(d, metadata, crop_h, w, cfg)

        duct_data.append({
            "id": i,
            "coordinates": [
                {"x": int(d["x1"]), "y": int(d["y1"])},
                {"x": int(d["x2"]), "y": int(d["y2"])},
            ],
            "dimension": dimension,
            "duct_type": duct_type,
            "pressure_class": pressure,
            "color": color_hex,
            "description": f"{TYPE_LABELS.get(duct_type, duct_type)} duct — {pressure} pressure",
        })

    return result, w, h, duct_data


# ─── Morphological detection ─────────────────────────────────────────────────

def _find_duct_rectangles(drawing, cfg: DuctDetectorConfig = DEFAULT_CONFIG):
    """Detect duct rectangles using directional morph close + open + contours."""
    gray = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
    dh, dw = gray.shape

    _, dark = cv2.threshold(gray, cfg.dark_threshold, 255, cv2.THRESH_BINARY_INV)

    title_x = int(dw * (1 - cfg.title_block_fraction))
    dark[:, title_x:] = 0

    # H ducts: close vertically, open horizontally
    k_close_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, cfg.morph_close_size))
    closed_h = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k_close_v)
    k_open_h = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.morph_open_size, 1))
    h_strips = cv2.morphologyEx(closed_h, cv2.MORPH_OPEN, k_open_h)

    # V ducts: close horizontally, open vertically
    k_close_h = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.morph_close_size, 1))
    closed_v = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k_close_h)
    k_open_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, cfg.morph_open_size))
    v_strips = cv2.morphologyEx(closed_v, cv2.MORPH_OPEN, k_open_v)

    ducts = []

    for strips, orient in [(h_strips, "H"), (v_strips, "V")]:
        contours, _ = cv2.findContours(strips, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < cfg.min_contour_area:
                continue

            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), angle = rect
            if rw < rh:
                rw, rh = rh, rw
                angle += 90

            aspect = rw / max(rh, 1)
            if (aspect < cfg.min_aspect_ratio or rh < cfg.min_duct_width
                    or rh > cfg.max_duct_width or rw < cfg.min_duct_length):
                continue

            # Reject slanted contours
            norm_angle = angle % 180
            tol = cfg.angle_tolerance
            if orient == "H" and not (norm_angle < tol or norm_angle > 180 - tol):
                continue
            if orient == "V" and not (90 - tol < norm_angle < 90 + tol):
                continue

            # Fill ratio check
            bx, by, bbw, bbh = cv2.boundingRect(cnt)
            bbox_area = bbw * bbh
            cnt_area = cv2.contourArea(cnt)
            fill_ratio = cnt_area / max(bbox_area, 1)
            if fill_ratio < cfg.min_fill_ratio:
                continue

            # Centerline endpoints
            shrink = cfg.centerline_shrink
            if orient == "H":
                margin = int(bbw * shrink)
                x1, x2 = bx + margin, bx + bbw - margin
                y1 = y2 = by + bbh // 2
            else:
                margin = int(bbh * shrink)
                y1, y2 = by + margin, by + bbh - margin
                x1 = x2 = bx + bbw // 2

            ducts.append({
                "orientation": orient,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": cx, "cy": cy, "length": rw, "width": rh,
            })

    return _deduplicate(ducts, cfg)


def _deduplicate(ducts, cfg: DuctDetectorConfig = DEFAULT_CONFIG):
    ducts.sort(key=lambda d: d["length"], reverse=True)
    keep = []
    for d in ducts:
        dup = False
        for k in keep:
            if d["orientation"] != k["orientation"]:
                continue
            if d["orientation"] == "H":
                if abs(d["y1"] - k["y1"]) < cfg.dedup_proximity:
                    overlap = min(d["x2"], k["x2"]) - max(d["x1"], k["x1"])
                    if overlap > d["length"] * cfg.dedup_overlap_ratio:
                        dup = True
                        break
            else:
                if abs(d["x1"] - k["x1"]) < cfg.dedup_proximity:
                    overlap = min(d["y2"], k["y2"]) - max(d["y1"], k["y1"])
                    if overlap > d["length"] * cfg.dedup_overlap_ratio:
                        dup = True
                        break
        if not dup:
            keep.append(d)
    return keep


# ─── GPT-4o metadata ─────────────────────────────────────────────────────────

def _get_duct_metadata(drawing, cfg: DuctDetectorConfig = DEFAULT_CONFIG):
    dh, dw = drawing.shape[:2]
    cell_w, cell_h = dw // cfg.grid_cols, dh // cfg.grid_rows

    max_px = cfg.grid_image_max_px
    scale = min(max_px / dw, max_px / dh, 1.0)
    grid_img = cv2.resize(drawing, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA) if scale < 1 else drawing.copy()
    gh, gw = grid_img.shape[:2]
    gcw, gch = gw // cfg.grid_cols, gh // cfg.grid_rows

    for c in range(1, cfg.grid_cols):
        cv2.line(grid_img, (c * gcw, 0), (c * gcw, gh), (0, 0, 255), 2)
    for r in range(1, cfg.grid_rows):
        cv2.line(grid_img, (0, r * gch), (gw, r * gch), (0, 0, 255), 2)
    for r in range(cfg.grid_rows):
        for c in range(cfg.grid_cols):
            cv2.putText(grid_img, f"{chr(65 + r)}{c + 1}",
                        (c * gcw + 5, r * gch + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    _, buf = cv2.imencode('.jpg', grid_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf).decode()

    try:
        client = OpenAI()
        row_labels = ", ".join(
            f"{chr(65 + r)}1-{chr(65 + r)}{cfg.grid_cols}" for r in range(cfg.grid_rows))
        resp = client.chat.completions.create(
            model=cfg.gpt_model, temperature=0, max_tokens=cfg.gpt_max_tokens,
            messages=[
                {"role": "system",
                 "content": "Expert HVAC engineer. Identify duct dimensions in grid cells."},
                {"role": "user", "content": [
                    {"type": "text", "text": (
                        f"This HVAC floor plan has a red grid: cells {row_labels}.\n"
                        "For each cell with ductwork, provide: cell label, duct dimension "
                        "if readable, duct_type (supply/return/exhaust).\n"
                        "Return JSON array: "
                        "[{\"cell\":\"B3\",\"dimension\":\"14\\\"\",\"duct_type\":\"supply\"}]"
                    )},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
                ]},
            ],
        )
        raw = resp.choices[0].message.content or ""
        cells = _parse_json_response(raw)

        result = []
        for cell in cells:
            label = cell.get("cell", "")
            if len(label) < 2:
                continue
            row = ord(label[0].upper()) - 65
            col = int(label[1:]) - 1
            if 0 <= row < cfg.grid_rows and 0 <= col < cfg.grid_cols:
                result.append({
                    "cx": (col + 0.5) * cell_w,
                    "cy": (row + 0.5) * cell_h,
                    "dimension": cell.get("dimension", "Unknown"),
                    "duct_type": cell.get("duct_type", "supply"),
                })
        return result
    except Exception as e:
        logger.warning(f"GPT-4o metadata call failed: {e}")
        return []


# ─── Metadata matching & pressure classification ─────────────────────────────

def _get_nearest_metadata(duct, metadata, crop_h, img_w,
                          cfg: DuctDetectorConfig = DEFAULT_CONFIG):
    """Find the nearest GPT metadata cell and return (dimension, duct_type)."""
    cx, cy = float(duct["cx"]), float(duct["cy"])
    if metadata:
        best, best_dist = None, float("inf")
        for m in metadata:
            dist = math.hypot(cx - m["cx"], cy - m["cy"])
            if dist < best_dist:
                best_dist = dist
                best = m
        if best and best_dist < max(crop_h, img_w) * cfg.metadata_max_distance_ratio:
            return best.get("dimension", "Unknown"), best.get("duct_type", "supply")

    return _estimate_dimension(duct, cfg), "supply"


def _estimate_dimension(duct, cfg: DuctDetectorConfig = DEFAULT_CONFIG):
    """Estimate duct dimension from detected pixel width."""
    px_w = float(duct.get("width", 0))
    for threshold, label in cfg.dim_thresholds:
        if px_w >= threshold:
            return label
    return '8"ø'


def _match_pressure(duct, metadata, crop_h, img_w,
                    cfg: DuctDetectorConfig = DEFAULT_CONFIG):
    cx, cy = float(duct["cx"]), float(duct["cy"])
    if metadata:
        best, best_dist = None, float("inf")
        for m in metadata:
            dist = math.hypot(cx - m["cx"], cy - m["cy"])
            if dist < best_dist:
                best_dist = dist
                best = m
        if best and best_dist < max(crop_h, img_w) * cfg.metadata_max_distance_ratio:
            return _classify_pressure(best.get("dimension", ""), cfg)

    return _classify_pressure(_estimate_dimension(duct, cfg), cfg)


def _classify_pressure(dimension, cfg: DuctDetectorConfig = DEFAULT_CONFIG):
    if not dimension or dimension == "Unknown":
        return "low"
    nums = re.findall(r'(\d+(?:\.\d+)?)', dimension)
    if nums:
        max_dim = max(float(n) for n in nums)
        if max_dim >= cfg.pressure_high_min:
            return "high"
        elif max_dim >= cfg.pressure_medium_min:
            return "medium"
    return "low"


# ─── Utilities ────────────────────────────────────────────────────────────────

def _parse_json_response(raw):
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


def _render_pdf(pdf_path, dpi):
    doc = fitz.open(pdf_path)
    page = doc[0]
    z = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(z, z))
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 3:
        img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif pix.n == 4:
        img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    elif pix.n == 1:
        img = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    else:
        doc.close()
        raise ValueError(f"Unsupported PDF pixel format: {pix.n} channels")
    doc.close()
    return img


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    f = sys.argv[1] if len(sys.argv) > 1 else "testset2.pdf"
    print(f"Detecting ducts in: {f}")
    img, w, h, _ = detect_ducts(f)
    print(f"Image: {w}x{h}")
    Path("static/rendered").mkdir(parents=True, exist_ok=True)
    cv2.imwrite("static/rendered/detected_ducts.png", img)
    s = 2000 / w
    cv2.imwrite("static/rendered/detected_sm.png",
                cv2.resize(img, (2000, int(h * s))))
    print("Saved: static/rendered/detected_ducts.png")
