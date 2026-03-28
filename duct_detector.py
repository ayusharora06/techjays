"""
HVAC Duct Detection — GPT-4o Grid Classification + Targeted CV.

Pipeline:
  1. Render image, crop to floor plan, overlay grid
  2. GPT-4o classifies which grid cells contain ducts + orientation + dimensions
  3. Within duct cells only, run CV detection (dark line parallel pairs)
  4. Connect detected segments across cells into duct paths
  5. Fallback to full-image CV if GPT-4o fails
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
GRID_COLS = 8
GRID_ROWS = 4


def detect_ducts(file_path: str, dpi: int = 200) -> tuple[np.ndarray, int, int, list[dict]]:
    """Main entry: detect ducts in HVAC drawing."""
    is_pdf = file_path.lower().endswith(".pdf")
    if is_pdf:
        img = _render_pdf(file_path, dpi)
    else:
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Cannot read: {file_path}")

    h, w = img.shape[:2]

    # Determine drawing area (exclude title block for PDFs)
    crop_h = int(h * 0.65) if is_pdf else h
    drawing = img[:crop_h]

    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        try:
            ducts = _detect_with_grid_vision(drawing, img, crop_h)
            if ducts:
                print(f"  Grid+CV pipeline: {len(ducts)} ducts")
                return img, w, h, ducts
            print("  Grid vision returned 0 ducts, trying fallback")
        except Exception as e:
            print(f"  Grid vision failed: {e}, trying fallback")

    ducts = _fallback_detect(img, crop_h)
    return img, w, h, ducts


# ── Stage 1: GPT-4o Grid Classification ──────────────────────────────────────

def _detect_with_grid_vision(drawing: np.ndarray, full_img: np.ndarray, crop_h: int) -> list[dict]:
    """Classify grid cells for ducts, then run CV within duct cells."""
    dh, dw = drawing.shape[:2]
    cell_w = dw // GRID_COLS
    cell_h = dh // GRID_ROWS

    # Create grid overlay for GPT-4o (at 100 DPI scale for reasonable API size)
    scale = min(2000 / dw, 2000 / dh, 1.0)
    if scale < 1:
        grid_img = cv2.resize(drawing, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        grid_img = drawing.copy()
    gh, gw = grid_img.shape[:2]
    gcw, gch = gw // GRID_COLS, gh // GRID_ROWS

    # Draw grid lines and labels
    for c in range(1, GRID_COLS):
        cv2.line(grid_img, (c * gcw, 0), (c * gcw, gh), (0, 0, 255), 2)
    for r in range(1, GRID_ROWS):
        cv2.line(grid_img, (0, r * gch), (gw, r * gch), (0, 0, 255), 2)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            label = f"{chr(65 + r)}{c + 1}"
            cv2.putText(grid_img, label, (c * gcw + 5, r * gch + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Send to GPT-4o
    _, buf = cv2.imencode('.jpg', grid_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf).decode()
    print(f"  Grid image: {gw}x{gh}, {len(b64) // 1024}KB")

    client = OpenAI()
    row_labels = ", ".join(f"{chr(65 + r)}1-{chr(65 + r)}{GRID_COLS}" for r in range(GRID_ROWS))

    prompt = (
        f"This HVAC floor plan has a red grid with cells labeled {row_labels}.\n\n"
        "For EACH cell containing visible ductwork (two parallel lines forming a channel), provide:\n"
        "- cell: the cell label\n"
        "- orientation: horizontal, vertical, or diagonal\n"
        "- dimension: duct size if readable (e.g. 14\", 12\"x8\"), or Unknown\n"
        "- enters: which side the duct enters (top/bottom/left/right)\n"
        "- exits: which side the duct exits\n\n"
        "Return ONLY a JSON array. Include ONLY cells with ductwork, not walls or equipment.\n"
        '[{"cell":"B3","orientation":"horizontal","dimension":"14\\"","enters":"left","exits":"right"}]'
    )

    resp = client.chat.completions.create(
        model="gpt-4o", temperature=0, max_tokens=4096,
        messages=[
            {"role": "system", "content": "You are an expert HVAC engineer. Identify cells containing ductwork (NOT walls, NOT equipment outlines). Be thorough."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
            ]},
        ],
    )

    raw = resp.choices[0].message.content or ""
    print(f"  GPT-4o: {len(raw)} chars")
    cells = _parse_grid_response(raw)
    if not cells:
        return []

    print(f"  Duct cells: {[c['cell'] for c in cells]}")

    # Stage 2: Run CV detection within each duct cell
    gray = cv2.cvtColor(full_img[:crop_h], cv2.COLOR_BGR2GRAY)
    all_segments = []
    cell_meta = {}  # store dimension/type per cell

    for cell_info in cells:
        label = cell_info["cell"]
        if len(label) < 2:
            continue
        row = ord(label[0].upper()) - 65
        col = int(label[1:]) - 1
        if row < 0 or row >= GRID_ROWS or col < 0 or col >= GRID_COLS:
            continue

        # Cell bounds in full-resolution image
        x0 = col * cell_w
        y0 = row * cell_h
        x1 = min((col + 1) * cell_w, dw)
        y1 = min((row + 1) * cell_h, dh)

        # Expand slightly to catch ducts at cell borders
        margin = max(cell_w, cell_h) // 10
        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)
        x1 = min(dw, x1 + margin)
        y1 = min(dh, y1 + margin)

        cell_gray = gray[y0:y1, x0:x1]
        orient = cell_info.get("orientation", "horizontal")

        segs = _detect_in_cell(cell_gray, x0, y0, orient)
        for s in segs:
            s["cell"] = label
            s["dimension"] = cell_info.get("dimension", "Unknown")
            s["duct_type"] = cell_info.get("duct_type", "supply")
        all_segments.extend(segs)

        cell_meta[label] = cell_info

    print(f"  Segments from cells: {len(all_segments)}")
    if not all_segments:
        return []

    # Connect segments into paths
    paths = _connect_segments(all_segments)
    print(f"  Connected paths: {len(paths)}")

    # Build output
    ducts = []
    for i, path in enumerate(paths):
        # Get dimension from the first segment's cell
        dim = "Unknown"
        for seg in all_segments:
            for pt in path["coords"]:
                if math.hypot(pt["x"] - seg["x1"], pt["y"] - seg["y1"]) < 50:
                    dim = seg.get("dimension", "Unknown")
                    break
            if dim != "Unknown":
                break

        pressure = _classify_pressure(dim)
        ducts.append({
            "id": f"duct-{i + 1}",
            "dimension": dim,
            "pressure_class": pressure,
            "color": PRESSURE_COLORS[pressure],
            "duct_type": "supply",
            "coordinates": path["coords"],
            "description": f"Duct path, {len(path['coords'])} points",
        })

    return ducts


def _detect_in_cell(cell_gray: np.ndarray, offset_x: int, offset_y: int, orient: str) -> list[dict]:
    """Run CV detection within a single cell region."""
    ch, cw = cell_gray.shape

    # Dark line threshold
    mean_val = np.mean(cell_gray)
    thresh_val = int(mean_val * 0.45)  # more aggressive for cell-level
    thresh_val = max(60, min(thresh_val, 140))
    _, dark = cv2.threshold(cell_gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

    # Clean noise
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))

    edges = cv2.Canny(dark, 50, 150)
    min_line = max(15, cw // 20)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(15, cw // 30),
                            minLineLength=min_line, maxLineGap=max(5, cw // 60))
    if lines is None:
        return []

    # Classify by orientation
    h_lines, v_lines, d_lines = [], [], []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_line:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
        entry = {"x1": x1 + offset_x, "y1": y1 + offset_y,
                 "x2": x2 + offset_x, "y2": y2 + offset_y,
                 "length": length, "angle": angle}
        if angle < 20 or angle > 160:
            h_lines.append(entry)
        elif 70 < angle < 110:
            v_lines.append(entry)
        else:
            d_lines.append(entry)

    # Find parallel pairs based on expected orientation
    segments = []
    scale = max(cw, 1) / 900  # cell-relative scale

    if orient in ("horizontal", "H"):
        segments += _find_pairs(h_lines, "H", scale)
    elif orient in ("vertical", "V"):
        segments += _find_pairs(v_lines, "V", scale)
    elif orient in ("diagonal", "D"):
        segments += _find_pairs(d_lines, "D", scale)

    # Also try other orientations as backup
    if not segments:
        segments += _find_pairs(h_lines, "H", scale)
        segments += _find_pairs(v_lines, "V", scale)

    return segments


def _parse_grid_response(raw: str) -> list[dict]:
    """Parse GPT-4o grid classification response."""
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


# ── CV Utilities ─────────────────────────────────────────────────────────────

def _find_pairs(lines, orient, scale):
    """Find parallel line pairs with duct-appropriate separation."""
    if len(lines) < 2:
        return []
    MIN_SEP = max(3, int(8 * scale))
    MAX_SEP = max(15, int(50 * scale))

    if orient == "H":
        lines.sort(key=lambda l: (l["y1"] + l["y2"]) / 2)
    elif orient == "V":
        lines.sort(key=lambda l: (l["x1"] + l["x2"]) / 2)
    else:
        lines.sort(key=lambda l: (l["x1"] + l["x2"]) / 2 * 0.7 + (l["y1"] + l["y2"]) / 2 * 0.7)

    pairs = []
    used = set()
    for i in range(len(lines)):
        if i in used:
            continue
        l1 = lines[i]
        best_j, best_score = None, 0
        for j in range(i + 1, min(i + 15, len(lines))):
            if j in used:
                continue
            l2 = lines[j]

            if orient == "H":
                sep = abs((l1["y1"] + l1["y2"]) / 2 - (l2["y1"] + l2["y2"]) / 2)
            elif orient == "V":
                sep = abs((l1["x1"] + l1["x2"]) / 2 - (l2["x1"] + l2["x2"]) / 2)
            else:
                mx1 = (l1["x1"] + l1["x2"]) / 2
                my1 = (l1["y1"] + l1["y2"]) / 2
                mx2 = (l2["x1"] + l2["x2"]) / 2
                my2 = (l2["y1"] + l2["y2"]) / 2
                sep = math.hypot(mx1 - mx2, my1 - my2) * abs(math.sin(math.radians(l1["angle"] - 45)))
                sep = max(sep, 1)

            if sep < MIN_SEP or sep > MAX_SEP:
                if sep > MAX_SEP + 10:
                    break
                continue

            a_diff = abs(l1["angle"] - l2["angle"])
            if a_diff > 15 and abs(180 - a_diff) > 15:
                continue

            if orient == "H":
                ov = _overlap(l1["x1"], l1["x2"], l2["x1"], l2["x2"])
                shorter = min(abs(l1["x2"] - l1["x1"]), abs(l2["x2"] - l2["x1"]))
            elif orient == "V":
                ov = _overlap(l1["y1"], l1["y2"], l2["y1"], l2["y2"])
                shorter = min(abs(l1["y2"] - l1["y1"]), abs(l2["y2"] - l2["y1"]))
            else:
                ov = min(l1["length"], l2["length"]) * 0.5
                shorter = min(l1["length"], l2["length"])

            if shorter > 0 and ov / shorter >= 0.3:
                if ov > best_score:
                    best_score = ov
                    best_j = j

        if best_j is not None:
            used.add(i)
            used.add(best_j)
            l2 = lines[best_j]
            pairs.append({
                "x1": (l1["x1"] + l2["x1"]) / 2,
                "y1": (l1["y1"] + l2["y1"]) / 2,
                "x2": (l1["x2"] + l2["x2"]) / 2,
                "y2": (l1["y2"] + l2["y2"]) / 2,
                "length": math.hypot((l1["x2"] + l2["x2"]) / 2 - (l1["x1"] + l2["x1"]) / 2,
                                     (l1["y2"] + l2["y2"]) / 2 - (l1["y1"] + l2["y1"]) / 2),
                "orient": orient,
            })
    return pairs


def _overlap(a1, a2, b1, b2):
    return max(0, min(max(a1, a2), max(b1, b2)) - max(min(a1, a2), min(b1, b2)))


def _connect_segments(segments):
    """Connect nearby segments into continuous paths."""
    if not segments:
        return []
    n = len(segments)
    avg_len = np.mean([s["length"] for s in segments]) if segments else 50
    connect_dist = max(30, int(avg_len * 0.5))

    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = segments[i], segments[j]
            for ex1, ey1 in [(si["x1"], si["y1"]), (si["x2"], si["y2"])]:
                for ex2, ey2 in [(sj["x1"], sj["y1"]), (sj["x2"], sj["y2"])]:
                    if math.hypot(ex1 - ex2, ey1 - ey2) < connect_dist:
                        if j not in adj[i]:
                            adj[i].append(j)
                        if i not in adj[j]:
                            adj[j].append(i)

    visited = [False] * n
    paths = []
    for start in range(n):
        if visited[start]:
            continue
        comp = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            comp.append(node)
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)

        segs = [segments[i] for i in comp]
        total_len = sum(s["length"] for s in segs)
        if total_len < 30:
            continue

        points = []
        for s in segs:
            points.append({"x": round(s["x1"], 1), "y": round(s["y1"], 1)})
            points.append({"x": round(s["x2"], 1), "y": round(s["y2"], 1)})
        ordered = _order_points(points)
        paths.append({"coords": ordered, "total_length": total_len})

    paths.sort(key=lambda p: -p["total_length"])
    return paths


def _order_points(points):
    if len(points) <= 2:
        return points
    unique = [points[0]]
    for p in points[1:]:
        if all(math.hypot(p["x"] - u["x"], p["y"] - u["y"]) > 8 for u in unique):
            unique.append(p)
    if len(unique) <= 1:
        return unique
    ordered = [unique[0]]
    remaining = list(unique[1:])
    while remaining:
        last = ordered[-1]
        dists = [math.hypot(p["x"] - last["x"], p["y"] - last["y"]) for p in remaining]
        ordered.append(remaining.pop(int(np.argmin(dists))))
    return ordered


# ── Fallback ─────────────────────────────────────────────────────────────────

def _fallback_detect(img: np.ndarray, crop_h: int) -> list[dict]:
    """Basic full-image CV fallback."""
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
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=max(20, int(40 * scale)),
                            minLineLength=max(15, int(40 * scale)),
                            maxLineGap=max(5, int(12 * scale)))
    if lines is None:
        return []

    h_lines, v_lines = [], []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < max(12, int(35 * scale)):
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
        entry = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "length": length, "angle": angle}
        if angle < 15 or angle > 165:
            h_lines.append(entry)
        elif 75 < angle < 105:
            v_lines.append(entry)

    segments = _find_pairs(h_lines, "H", scale) + _find_pairs(v_lines, "V", scale)
    if not segments:
        return []

    paths = _connect_segments(segments)
    ducts = []
    for i, path in enumerate(paths):
        ducts.append({
            "id": f"duct-{i + 1}", "dimension": "Unknown",
            "pressure_class": "low", "color": PRESSURE_COLORS["low"],
            "duct_type": "supply", "coordinates": path["coords"],
            "description": f"Duct (CV fallback), {len(path['coords'])} points",
        })
    return ducts


# ── PDF Rendering ────────────────────────────────────────────────────────────

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
            cv2.line(annotated, pts[k], pts[k + 1], c, 4)
        if pts:
            mid = pts[len(pts) // 2]
            lbl = d["dimension"] if d["dimension"] != "Unknown" else d["id"]
            cv2.putText(annotated, lbl, (mid[0] + 5, mid[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

    Path("static/rendered").mkdir(parents=True, exist_ok=True)
    cv2.imwrite("static/rendered/detected_ducts.png", annotated)
    print(f"Saved: static/rendered/detected_ducts.png")
