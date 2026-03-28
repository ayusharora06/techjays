"""
HVAC Duct Detection — GPT Image Edit + CV Extraction.

Pipeline:
  1. Render image, crop title block
  2. gpt-image-1.5 fills duct channels with bright red
  3. CV extracts red mask from annotated image
  4. Skeletonize → duct centerlines
  5. GPT-4o grid call → dimensions + metadata
  6. Fallback to grid+CV if image edit fails
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

PRESSURE_COLORS = {"high": "#FF4444", "medium": "#FF8C00", "low": "#4488FF"}

IMAGE_EDIT_PROMPT = (
    "This is an HVAC mechanical floor plan. "
    "Identify ALL duct channels — the straight rectangular/round air channels shown as two parallel lines. "
    "Ducts run in straight lines: horizontally, vertically, or at an angle.\n\n"
    "For each duct, draw a THIN 1-pixel centerline through the middle of the duct channel, colored by pressure class:\n"
    "- RED line for HIGH pressure ducts (18 inches or larger)\n"
    "- ORANGE line for MEDIUM pressure ducts (12 to 17 inches)\n"
    "- BLUE line for LOW pressure ducts (smaller than 12 inches)\n"
    "- If dimension is unreadable, use BLUE.\n\n"
    "Also write the duct dimension in small text near each duct (e.g. 14\"⌀, 12\"x8\"). Keep text small so it doesn't obscure the drawing.\n\n"
    "IMPORTANT: Only mark DUCTS — the straight air channels between equipment and diffusers. "
    "Do NOT mark pipes, connectors, fittings, elbows, dampers, equipment outlines, walls, text, or symbols. "
    "Only the straight duct runs."
)

GRID_COLS = 8
GRID_ROWS = 4


def detect_ducts(file_path: str, dpi: int = 200) -> tuple[np.ndarray, int, int, list[dict]]:
    """Main entry: detect ducts in HVAC drawing."""
    is_pdf = file_path.lower().endswith(".pdf")
    if is_pdf:
        full_img = _render_pdf(file_path, dpi)
    else:
        full_img = cv2.imread(file_path)
        if full_img is None:
            raise ValueError(f"Cannot read: {file_path}")

    h, w = full_img.shape[:2]
    crop_h = int(h * 0.65) if is_pdf else h

    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        try:
            result = _pipeline_image_edit(full_img, crop_h, file_path, dpi)
            if result is not None:
                print("  Image edit pipeline: success")
                # Return annotated image directly — no centerlines needed
                return result, w, h, []
            print("  Image edit returned None, trying fallback")
        except Exception as e:
            print(f"  Image edit pipeline failed: {e}, trying fallback")

    ducts = _fallback_detect(full_img, crop_h)
    return full_img, w, h, ducts


# ── Main Pipeline: GPT Image Edit + CV Extraction ────────────────────────────

def _pipeline_image_edit(full_img: np.ndarray, crop_h: int, file_path: str, dpi: int) -> np.ndarray | None:
    """Send to GPT image edit, return annotated image with colored ducts + labels."""
    drawing = full_img[:crop_h]

    annotated = _get_annotated_image(drawing)
    if annotated is None:
        return None

    # Place annotated image into full-size canvas (with title block from original)
    h, w = full_img.shape[:2]
    result = full_img.copy()
    resized = cv2.resize(annotated, (w, crop_h), interpolation=cv2.INTER_LANCZOS4)
    result[:crop_h] = resized

    return result


# ── Step 2: GPT Image Edit ───────────────────────────────────────────────────

def _get_annotated_image(drawing: np.ndarray) -> np.ndarray | None:
    """Send drawing to gpt-image-1.5, get back image with red-filled ducts."""
    # Save as PNG for API
    tmp_path = "/tmp/hvac_duct_input.png"
    cv2.imwrite(tmp_path, drawing)

    client = OpenAI()
    print(f"  Sending to gpt-image-1.5: {drawing.shape[1]}x{drawing.shape[0]}")

    result = client.images.edit(
        model="gpt-image-1.5",
        image=open(tmp_path, "rb"),
        prompt=IMAGE_EDIT_PROMPT,
        size="1536x1024",
    )

    img_data = base64.b64decode(result.data[0].b64_json)
    arr = np.frombuffer(img_data, dtype=np.uint8)
    annotated = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if annotated is None:
        print("  Failed to decode annotated image")
        return None

    print(f"  Annotated image: {annotated.shape[1]}x{annotated.shape[0]}")

    # Save for debugging
    Path("static/rendered").mkdir(parents=True, exist_ok=True)
    cv2.imwrite("static/rendered/gpt_annotated.png", annotated)

    return annotated


# ── Step 3: Red Mask Extraction ──────────────────────────────────────────────

def _extract_red_mask(annotated: np.ndarray) -> np.ndarray:
    """Extract bright red regions from the annotated image."""
    hsv = cv2.cvtColor(annotated, cv2.COLOR_BGR2HSV)

    # Red in HSV wraps around 0/180
    mask1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([12, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([168, 80, 80]), np.array([180, 255, 255]))
    red_mask = mask1 | mask2

    # Also catch via BGR directly — high R, low G, low B
    b, g, r = cv2.split(annotated)
    bgr_mask = (r > 150) & (g < 100) & (b < 100)
    red_mask = red_mask | (bgr_mask.astype(np.uint8) * 255)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return red_mask


# ── Step 4: Skeletonize → Centerlines ────────────────────────────────────────

def _mask_to_centerlines(mask: np.ndarray) -> list[dict]:
    """Convert binary duct mask to centerline polylines."""
    # Thin the mask to 1-pixel skeleton
    skeleton = cv2.ximgproc.thinning(mask) if hasattr(cv2, 'ximgproc') else _simple_thin(mask)

    # Find connected components in skeleton
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton, connectivity=8)

    paths = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < 15:  # skip tiny fragments
            continue

        # Get skeleton pixels for this component
        ys, xs = np.where(labels == label)
        if len(xs) < 2:
            continue

        # Order pixels into a polyline
        points = _trace_skeleton(xs, ys)

        # Simplify polyline (reduce point count)
        points_arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        epsilon = max(3.0, len(points) * 0.02)
        simplified = cv2.approxPolyDP(points_arr, epsilon, closed=False)
        coords = [{"x": round(float(p[0][0]), 1), "y": round(float(p[0][1]), 1)}
                   for p in simplified]

        if len(coords) >= 2:
            total_len = sum(math.hypot(coords[i+1]["x"] - coords[i]["x"],
                                       coords[i+1]["y"] - coords[i]["y"])
                           for i in range(len(coords) - 1))
            if total_len > 20:
                paths.append({"coords": coords, "total_length": total_len})

    paths.sort(key=lambda p: -p["total_length"])
    return paths


def _trace_skeleton(xs, ys):
    """Order skeleton pixels into a sequential path using nearest-neighbor."""
    points = list(zip(xs.tolist(), ys.tolist()))
    if len(points) <= 2:
        return points

    # Start from one endpoint (pixel with fewest neighbors)
    # For simplicity, use nearest-neighbor traversal from first point
    ordered = [points[0]]
    remaining = set(range(1, len(points)))

    for _ in range(len(points) - 1):
        if not remaining:
            break
        last = ordered[-1]
        best_idx = min(remaining, key=lambda i: (points[i][0] - last[0])**2 + (points[i][1] - last[1])**2)
        ordered.append(points[best_idx])
        remaining.discard(best_idx)

    return ordered


def _simple_thin(mask):
    """Simple thinning fallback if cv2.ximgproc not available."""
    # Use distance transform + threshold as approximation
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, skeleton = cv2.threshold(dist, dist.max() * 0.3, 255, cv2.THRESH_BINARY)
    return skeleton.astype(np.uint8)


# ── Step 5: Metadata via GPT-4o Grid ─────────────────────────────────────────

def _get_duct_metadata(drawing: np.ndarray) -> list[dict]:
    """Get duct dimensions/types via GPT-4o grid classification."""
    dh, dw = drawing.shape[:2]
    cell_w = dw // GRID_COLS
    cell_h = dh // GRID_ROWS

    # Create grid overlay
    scale = min(2000 / dw, 2000 / dh, 1.0)
    if scale < 1:
        grid_img = cv2.resize(drawing, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        grid_img = drawing.copy()
    gh, gw = grid_img.shape[:2]
    gcw, gch = gw // GRID_COLS, gh // GRID_ROWS

    for c in range(1, GRID_COLS):
        cv2.line(grid_img, (c * gcw, 0), (c * gcw, gh), (0, 0, 255), 2)
    for r in range(1, GRID_ROWS):
        cv2.line(grid_img, (0, r * gch), (gw, r * gch), (0, 0, 255), 2)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            label = f"{chr(65 + r)}{c + 1}"
            cv2.putText(grid_img, label, (c * gcw + 5, r * gch + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    _, buf = cv2.imencode('.jpg', grid_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf).decode()

    try:
        client = OpenAI()
        row_labels = ", ".join(f"{chr(65 + r)}1-{chr(65 + r)}{GRID_COLS}" for r in range(GRID_ROWS))
        prompt = (
            f"This HVAC floor plan has a red grid: cells {row_labels}.\n"
            "For each cell with ductwork, provide: cell label, duct dimension if readable, duct_type (supply/return/exhaust).\n"
            "Return JSON array: [{\"cell\":\"B3\",\"dimension\":\"14\\\"\",\"duct_type\":\"supply\"}]"
        )

        resp = client.chat.completions.create(
            model="gpt-4o", temperature=0, max_tokens=2048,
            messages=[
                {"role": "system", "content": "Expert HVAC engineer. Identify duct dimensions in grid cells."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
                ]},
            ],
        )
        raw = resp.choices[0].message.content or ""
        cells = _parse_json_response(raw)

        # Convert cell labels to pixel regions
        result = []
        for cell in cells:
            label = cell.get("cell", "")
            if len(label) < 2:
                continue
            row = ord(label[0].upper()) - 65
            col = int(label[1:]) - 1
            if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
                cx = (col + 0.5) * cell_w
                cy = (row + 0.5) * cell_h
                result.append({
                    "cx": cx, "cy": cy,
                    "dimension": cell.get("dimension", "Unknown"),
                    "duct_type": cell.get("duct_type", "supply"),
                })
        return result
    except Exception as e:
        print(f"  Metadata call failed: {e}")
        return []


def _match_metadata(path: dict, metadata: list[dict]) -> tuple[str, str, str]:
    """Find the closest metadata entry for a duct path."""
    if not metadata:
        return "Unknown", "supply", "low"

    # Use path midpoint
    coords = path["coords"]
    mid = coords[len(coords) // 2]
    mx, my = mid["x"], mid["y"]

    best = None
    best_dist = float("inf")
    for m in metadata:
        d = math.hypot(mx - m["cx"], my - m["cy"])
        if d < best_dist:
            best_dist = d
            best = m

    if best and best_dist < 1000:
        dim = best.get("dimension", "Unknown")
        dtype = best.get("duct_type", "supply")
        pressure = _classify_pressure(dim)
        return dim, dtype, pressure

    return "Unknown", "supply", "low"


def _classify_pressure(dimension: str) -> str:
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


def _parse_json_response(raw: str) -> list[dict]:
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


# ── Fallback ─────────────────────────────────────────────────────────────────

def _fallback_detect(img: np.ndarray, crop_h: int) -> list[dict]:
    """Basic CV fallback."""
    print("  Using fallback CV detection")
    gray = cv2.cvtColor(img[:crop_h], cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mean_val = np.mean(gray)
    thresh_val = int(mean_val * 0.5)
    thresh_val = max(70, min(thresh_val, 150))
    _, dark = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    scale = max(w, 1) / 7200
    edges = cv2.Canny(dark, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(20, int(40 * scale)),
                            minLineLength=max(15, int(40 * scale)), maxLineGap=max(5, int(12 * scale)))
    if lines is None:
        return []

    segments = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        length = math.hypot(x2 - x1, y2 - y1)
        if length > max(12, int(35 * scale)):
            segments.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "length": length})

    ducts = []
    for i, s in enumerate(segments[:20]):
        ducts.append({
            "id": f"duct-{i + 1}", "dimension": "Unknown",
            "pressure_class": "low", "color": PRESSURE_COLORS["low"],
            "duct_type": "supply",
            "coordinates": [{"x": s["x1"], "y": s["y1"]}, {"x": s["x2"], "y": s["y2"]}],
            "description": "Duct (CV fallback)",
        })
    return ducts


# ── Utilities ────────────────────────────────────────────────────────────────

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

    img, w, h, ducts = detect_ducts(f)
    print(f"\nImage: {w}x{h}, Ducts: {len(ducts)}")
    for d in ducts:
        print(f"  {d['id']}: {d['dimension']} ({d['pressure_class']}) pts={len(d['coordinates'])}")

    annotated = img.copy()
    for d in ducts:
        c = tuple(int(d["color"][i:i + 2], 16) for i in (5, 3, 1))
        pts = [(int(p["x"]), int(p["y"])) for p in d["coordinates"]]
        for k in range(len(pts) - 1):
            cv2.line(annotated, pts[k], pts[k + 1], c, 2)

    Path("static/rendered").mkdir(parents=True, exist_ok=True)
    cv2.imwrite("static/rendered/detected_ducts.png", annotated)
    print(f"Saved: static/rendered/detected_ducts.png")
