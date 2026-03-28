"""
HVAC Duct Detection — Morphological Close + Contour Detection.

Pipeline:
  1. Render PDF → full-res image, crop title block
  2. Threshold darkest pixels (<50) → duct outlines only
  3. Directional morph close (fill duct gaps) + open (keep elongated strips)
  4. findContours → filter elongated rectangles → duct centerlines
  5. GPT-4o grid → pressure classification
  6. Draw colored centerlines on original image
"""

import os
import cv2
import numpy as np
import fitz
import math
import json
import re
import base64
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

PRESSURE_BGR = {
    "high": (0, 0, 255),       # Red
    "medium": (0, 140, 255),   # Orange
    "low": (255, 136, 68),     # Blue
}

GRID_COLS = 8
GRID_ROWS = 4


def detect_ducts(file_path: str, dpi: int = 200):
    is_pdf = file_path.lower().endswith(".pdf")
    if is_pdf:
        full_img = _render_pdf(file_path, dpi)
    else:
        full_img = cv2.imread(file_path)
        if full_img is None:
            raise ValueError(f"Cannot read: {file_path}")

    h, w = full_img.shape[:2]
    crop_h = int(h * 0.65) if is_pdf else h
    drawing = full_img[:crop_h]

    print("  Detecting duct rectangles...")
    ducts = _find_duct_rectangles(drawing)
    print(f"  Found {len(ducts)} ducts")

    metadata = []
    if os.getenv("OPENAI_API_KEY", ""):
        print("  Getting metadata from GPT-4o...")
        metadata = _get_duct_metadata(drawing)
        print(f"  Got {len(metadata)} metadata entries")

    result = full_img.copy()
    for d in ducts:
        pressure = _match_pressure(d, metadata, crop_h, w)
        color = PRESSURE_BGR.get(pressure, PRESSURE_BGR["low"])
        cv2.line(result, (d["x1"], d["y1"]), (d["x2"], d["y2"]), color, 3)

    return result, w, h, []


def _find_duct_rectangles(drawing):
    """Detect duct rectangles using directional morph close + open + contours."""
    gray = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
    dh, dw = gray.shape

    # Only darkest pixels — duct outlines, not gray building walls
    _, dark = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Exclude right 15% (title block area)
    title_x = int(dw * 0.85)
    dark[:, title_x:] = 0

    # H ducts: close vertically (fill gap between top/bottom walls)
    #          then open horizontally (keep only wide horizontal strips)
    k_close_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 70))
    closed_h = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k_close_v)
    k_open_h = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    h_strips = cv2.morphologyEx(closed_h, cv2.MORPH_OPEN, k_open_h)

    # V ducts: close horizontally, then open vertically
    k_close_h = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 1))
    closed_v = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k_close_h)
    k_open_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
    v_strips = cv2.morphologyEx(closed_v, cv2.MORPH_OPEN, k_open_v)

    ducts = []

    for strips, orient in [(h_strips, "H"), (v_strips, "V")]:
        contours, _ = cv2.findContours(strips, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 1500:
                continue

            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), angle = rect
            if rw < rh:
                rw, rh = rh, rw
                angle += 90

            aspect = rw / max(rh, 1)
            if aspect < 2.5 or rh < 10 or rh > 160 or rw < 80:
                continue

            # Centerline endpoints — use boundingRect, shrink by 10% each end
            bx, by, bbw, bbh = cv2.boundingRect(cnt)
            shrink = 0.10
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

    # Deduplicate overlapping detections
    return _deduplicate(ducts)


def _deduplicate(ducts):
    ducts.sort(key=lambda d: d["length"], reverse=True)
    keep = []
    for d in ducts:
        dup = False
        for k in keep:
            if d["orientation"] != k["orientation"]:
                continue
            if d["orientation"] == "H":
                if abs(d["y1"] - k["y1"]) < 30:
                    overlap = min(d["x2"], k["x2"]) - max(d["x1"], k["x1"])
                    if overlap > d["length"] * 0.5:
                        dup = True
                        break
            else:
                if abs(d["x1"] - k["x1"]) < 30:
                    overlap = min(d["y2"], k["y2"]) - max(d["y1"], k["y1"])
                    if overlap > d["length"] * 0.5:
                        dup = True
                        break
        if not dup:
            keep.append(d)
    return keep


# ═══════════════════════════════════════════════════════════════════════════════
# GPT-4o metadata
# ═══════════════════════════════════════════════════════════════════════════════

def _get_duct_metadata(drawing):
    dh, dw = drawing.shape[:2]
    cell_w, cell_h = dw // GRID_COLS, dh // GRID_ROWS

    scale = min(2000 / dw, 2000 / dh, 1.0)
    grid_img = cv2.resize(drawing, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA) if scale < 1 else drawing.copy()
    gh, gw = grid_img.shape[:2]
    gcw, gch = gw // GRID_COLS, gh // GRID_ROWS

    for c in range(1, GRID_COLS):
        cv2.line(grid_img, (c * gcw, 0), (c * gcw, gh), (0, 0, 255), 2)
    for r in range(1, GRID_ROWS):
        cv2.line(grid_img, (0, r * gch), (gw, r * gch), (0, 0, 255), 2)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cv2.putText(grid_img, f"{chr(65 + r)}{c + 1}",
                        (c * gcw + 5, r * gch + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    _, buf = cv2.imencode('.jpg', grid_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf).decode()

    try:
        client = OpenAI()
        row_labels = ", ".join(
            f"{chr(65 + r)}1-{chr(65 + r)}{GRID_COLS}" for r in range(GRID_ROWS))
        resp = client.chat.completions.create(
            model="gpt-4o", temperature=0, max_tokens=2048,
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
            if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
                result.append({
                    "cx": (col + 0.5) * cell_w,
                    "cy": (row + 0.5) * cell_h,
                    "dimension": cell.get("dimension", "Unknown"),
                    "duct_type": cell.get("duct_type", "supply"),
                })
        return result
    except Exception as e:
        print(f"  Metadata call failed: {e}")
        return []


def _match_pressure(duct, metadata, crop_h, img_w):
    if not metadata:
        return "low"
    cx, cy = duct["cx"], duct["cy"]
    best, best_dist = None, float("inf")
    for m in metadata:
        dist = math.hypot(cx - m["cx"], cy - m["cy"])
        if dist < best_dist:
            best_dist = dist
            best = m
    if best and best_dist < max(crop_h, img_w) * 0.3:
        return _classify_pressure(best.get("dimension", ""))
    return "low"


def _classify_pressure(dimension):
    if not dimension or dimension == "Unknown":
        return "low"
    nums = re.findall(r'(\d+(?:\.\d+)?)', dimension)
    if nums:
        max_dim = max(float(n) for n in nums)
        if max_dim >= 18:
            return "high"
        elif max_dim >= 12:
            return "medium"
    return "low"


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
    img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if pix.n == 3 else cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    doc.close()
    return img


if __name__ == "__main__":
    import sys
    f = sys.argv[1] if len(sys.argv) > 1 else "testset2.pdf"
    print(f"Detecting ducts in: {f}")
    img, w, h, _ = detect_ducts(f)
    print(f"Image: {w}x{h}")
    Path("static/rendered").mkdir(parents=True, exist_ok=True)
    cv2.imwrite("static/rendered/detected_ducts.png", img)
    # Save smaller version for quick viewing
    s = 2000 / w
    cv2.imwrite("static/rendered/detected_sm.png",
                cv2.resize(img, (2000, int(h * s))))
    print("Saved: static/rendered/detected_ducts.png")
