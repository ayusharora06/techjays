"""
Detect duct areas from input.png using wall-gap-wall pattern scanning.

Ducts = two parallel dark lines (1-3px walls) with a consistent white gap
(8-36px) between them, running in straight horizontal or vertical lines.

Strategy:
  1. Threshold darkest pixels (< 60) + dilate by 1px for anti-aliasing
  2. Scan every column for wall-gap-wall pattern → horizontal duct votes
  3. Scan every row for wall-gap-wall pattern → vertical duct votes
  4. Group votes by y-center (H) or x-center (V) into "bands"
  5. Split bands into continuous runs, filter by length and coverage
  6. Verify each candidate: gap must be white (>200), not gray fill
"""

import cv2
import numpy as np
import math
from pathlib import Path


# --- Parameters ---
DARK_THRESH = 60       # Only the very darkest pixels (duct outlines)
DILATE_SIZE = 3        # Dilate to bridge anti-aliased wall gaps
GAP_MIN = 8            # Min gap between parallel walls (px)
GAP_MAX = 36           # Max gap between parallel walls (px)
WALL_MIN = 1           # Min wall thickness (px, after dilation)
WALL_MAX = 8           # Max wall thickness (px, after dilation)
MIN_VOTES = 15         # Min number of scan lines confirming a band
MIN_RUN_LEN = 80       # Min continuous duct run length (px)
MIN_COVERAGE = 0.25    # Min fraction of columns/rows with votes in a run
RUN_GAP_BREAK = 20     # Gap in votes that splits a run
WHITE_THRESH = 210     # Gap between walls must be brighter than this


def _find_wall_gap_wall(line_data):
    """Find wall-gap-wall patterns in a 1D binary array (255=dark, 0=light).

    Both walls must have similar thickness (ratio < 3:1) to avoid matching
    a thin duct wall against a thick building wall."""
    results = []
    n = len(line_data)
    runs = []
    i = 0
    while i < n:
        if line_data[i] > 0:
            start = i
            while i < n and line_data[i] > 0:
                i += 1
            runs.append((start, i, i - start))
        else:
            i += 1
    for k in range(len(runs) - 1):
        s1, e1, w1 = runs[k]
        s2, e2, w2 = runs[k + 1]
        gap = s2 - e1
        if (WALL_MIN <= w1 <= WALL_MAX and
            WALL_MIN <= w2 <= WALL_MAX and
            GAP_MIN <= gap <= GAP_MAX):
            # Both walls should be similar thickness
            thick = max(w1, w2)
            thin = max(min(w1, w2), 1)
            if thick / thin <= 3:
                results.append((s1, e2, (e1 + s2) / 2, gap))
    return results


def _build_bands(centers_dict):
    """Group nearby center coordinates into bands (within 3px)."""
    keys = sorted(centers_dict.keys())
    bands = []
    current = None
    for k in keys:
        if current is None or k - current['end'] > 3:
            if current is not None:
                bands.append(current)
            current = {'start': k, 'end': k, 'points': list(centers_dict[k])}
        else:
            current['end'] = k
            current['points'].extend(centers_dict[k])
    if current:
        bands.append(current)
    return bands


def _verify_white_gap(gray, x_s, x_e, y_t, y_b, orientation, strict=False):
    """Verify the gap between detected walls is white (not gray fill/equipment).

    Real ducts have a white gap (median > 190) in most cross-sections.
    Elements like equipment, elbows, or text areas have gray fills.

    For short runs (strict=True), require higher pass rate (70%) to
    reduce false positives from equipment symbols and text labels."""
    min_median = 190
    pass_rate = 0.7 if strict else 0.5

    if orientation == 'H':
        n_checks = min(15, max(5, (x_e - x_s) // 10))
        white_count = 0
        total = 0
        for i in range(n_checks):
            x = x_s + (x_e - x_s) * (i + 1) // (n_checks + 1)
            if x < 0 or x >= gray.shape[1]:
                continue
            col = gray[y_t:y_b, x]
            if len(col) < 4:
                continue
            mid_start = len(col) // 4
            mid_end = 3 * len(col) // 4
            mid = col[mid_start:mid_end]
            if len(mid) > 0:
                total += 1
                if np.median(mid) > min_median:
                    white_count += 1
        return total > 0 and white_count >= max(2, total * pass_rate)
    else:
        n_checks = min(15, max(5, (y_b - y_t) // 10))
        white_count = 0
        total = 0
        for i in range(n_checks):
            y = y_t + (y_b - y_t) * (i + 1) // (n_checks + 1)
            if y < 0 or y >= gray.shape[0]:
                continue
            row = gray[y, x_s:x_e]
            if len(row) < 4:
                continue
            mid_start = len(row) // 4
            mid_end = 3 * len(row) // 4
            mid = row[mid_start:mid_end]
            if len(mid) > 0:
                total += 1
                if np.median(mid) > min_median:
                    white_count += 1
        return total > 0 and white_count >= max(2, total * pass_rate)


def _split_inconsistent_run(run_pts, pos_idx, ws_idx, we_idx, gap_idx,
                            gray, orientation):
    """Split a long run with inconsistent wall positions into sub-runs
    where wall positions are stable (std < 10)."""
    if len(run_pts) < MIN_VOTES:
        return []

    # Use a sliding window: group consecutive points with similar wall positions
    sub_runs = []
    window_start = 0

    while window_start < len(run_pts):
        # Grow window while wall positions stay consistent
        best_end = window_start + MIN_VOTES
        if best_end > len(run_pts):
            break

        ref_ws = np.median([p[ws_idx] for p in run_pts[window_start:best_end]])
        ref_we = np.median([p[we_idx] for p in run_pts[window_start:best_end]])

        for j in range(best_end, len(run_pts)):
            ws = run_pts[j][ws_idx]
            we = run_pts[j][we_idx]
            if abs(ws - ref_ws) > 10 or abs(we - ref_we) > 10:
                break
            best_end = j + 1

        if best_end - window_start >= MIN_VOTES:
            seg = run_pts[window_start:best_end]
            r_start = seg[0][pos_idx]
            r_end = seg[-1][pos_idx]
            run_len = r_end - r_start

            if run_len >= MIN_RUN_LEN:
                seg_ws = [p[ws_idx] for p in seg]
                seg_we = [p[we_idx] for p in seg]
                coord_start = int(np.median(seg_ws))
                coord_end = int(np.median(seg_we))
                duct_width = coord_end - coord_start
                avg_gap = np.median([p[gap_idx] for p in seg])
                cov = len(seg) / max(run_len, 1)

                if cov >= MIN_COVERAGE:
                    strict = run_len < 100
                    if orientation == 'H':
                        ok = _verify_white_gap(gray, r_start, r_end,
                                               coord_start, coord_end, 'H', strict)
                    else:
                        ok = _verify_white_gap(gray, coord_start, coord_end,
                                               r_start, r_end, 'V', strict)
                    if ok:
                        sub_runs.append((r_start, r_end, coord_start, coord_end,
                                        duct_width, avg_gap, cov))

        window_start = best_end

    return sub_runs


def _extract_runs(band_points, positions_idx, wall_start_idx, wall_end_idx, gap_idx,
                  gray, img_w, img_h, orientation):
    """Extract continuous duct runs from a band, with verification."""
    positions = [p[positions_idx] for p in band_points]
    if len(positions) < MIN_VOTES:
        return []

    gaps = [p[gap_idx] for p in band_points]
    w1s = [p[wall_start_idx] for p in band_points]
    w2s = [p[wall_end_idx] for p in band_points]

    avg_gap = np.median(gaps)
    coord_start = int(np.median(w1s))
    coord_end = int(np.median(w2s))
    duct_width = coord_end - coord_start

    # Split into continuous runs
    pos_sorted = sorted(set(positions))
    runs = []
    run_start = pos_sorted[0]
    prev = pos_sorted[0]
    for p in pos_sorted[1:]:
        if p - prev > RUN_GAP_BREAK:
            runs.append((run_start, prev))
            run_start = p
        prev = p
    runs.append((run_start, prev))

    results = []
    for r_start, r_end in runs:
        run_len = r_end - r_start
        if run_len < MIN_RUN_LEN:
            continue
        run_votes = sum(1 for p in positions if r_start <= p <= r_end)
        run_coverage = run_votes / max(run_len, 1)
        if run_coverage < MIN_COVERAGE:
            continue

        # Verify white gap (stricter for short runs which are more likely FP)
        strict = run_len < 100
        if orientation == 'H':
            if not _verify_white_gap(gray, r_start, r_end, coord_start, coord_end, 'H', strict):
                continue
        else:
            if not _verify_white_gap(gray, coord_start, coord_end, r_start, r_end, 'V', strict):
                continue

        # For long runs, check consistency of wall positions.
        # If positions vary too much, split into consistent sub-runs.
        if run_len > 200:
            run_pts = sorted(
                [p for p in band_points if r_start <= p[positions_idx] <= r_end],
                key=lambda p: p[positions_idx])
            if run_pts:
                starts_arr = np.array([p[wall_start_idx] for p in run_pts])
                ends_arr = np.array([p[wall_end_idx] for p in run_pts])
                if np.std(starts_arr) > 40 or np.std(ends_arr) > 40:
                    # Split into windows and find consistent sub-runs
                    sub_runs = _split_inconsistent_run(
                        run_pts, positions_idx, wall_start_idx, wall_end_idx,
                        gap_idx, gray, orientation)
                    results.extend(sub_runs)
                    continue

        results.append((r_start, r_end, coord_start, coord_end,
                        duct_width, avg_gap, run_coverage))
    return results


def detect_duct_rects(img_path="input.png"):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Cannot read {img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"Image: {w}x{h}")

    # Step 1: Dark mask + dilate
    _, dark = cv2.threshold(gray, DARK_THRESH, 255, cv2.THRESH_BINARY_INV)
    k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (DILATE_SIZE, DILATE_SIZE))
    dark = cv2.dilate(dark, k_dil, iterations=1)

    # Step 2: Scan columns for horizontal duct walls
    h_centers = {}
    for x in range(w):
        col = dark[:, x]
        for y1, y2, yc, gap in _find_wall_gap_wall(col):
            yc_r = round(yc)
            h_centers.setdefault(yc_r, []).append((x, y1, y2, gap))

    # Step 3: Scan rows for vertical duct walls
    v_centers = {}
    for y in range(h):
        row = dark[y, :]
        for x1, x2, xc, gap in _find_wall_gap_wall(row):
            xc_r = round(xc)
            v_centers.setdefault(xc_r, []).append((y, x1, x2, gap))

    # Step 4: Group into bands and extract runs
    h_bands = _build_bands(h_centers)
    v_bands = _build_bands(v_centers)

    result = img.copy()
    ducts = []

    # Horizontal ducts
    for band in h_bands:
        runs = _extract_runs(band['points'], 0, 1, 2, 3, gray, w, h, 'H')
        for x_start, x_end, y_top, y_bot, duct_w, gap, cov in runs:
            y_center = (y_top + y_bot) / 2
            length = x_end - x_start
            ducts.append({
                "orientation": "H",
                "x_start": x_start, "x_end": x_end,
                "y_top": y_top, "y_bot": y_bot,
                "cx": (x_start + x_end) / 2, "cy": y_center,
                "length": length, "width": duct_w,
                "gap": gap, "coverage": cov,
            })
            cv2.rectangle(result, (x_start, y_top), (x_end, y_bot), (0, 255, 0), 2)
            cv2.line(result, (x_start, int(y_center)), (x_end, int(y_center)),
                     (0, 0, 255), 1)
            lbl = f"{length}x{duct_w}"
            cv2.putText(result, lbl, (x_start, y_top - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)

    # Vertical ducts
    for band in v_bands:
        runs = _extract_runs(band['points'], 0, 1, 2, 3, gray, w, h, 'V')
        for y_start, y_end, x_left, x_right, duct_w, gap, cov in runs:
            x_center = (x_left + x_right) / 2
            length = y_end - y_start
            ducts.append({
                "orientation": "V",
                "x_left": x_left, "x_right": x_right,
                "y_start": y_start, "y_end": y_end,
                "cx": x_center, "cy": (y_start + y_end) / 2,
                "length": length, "width": duct_w,
                "gap": gap, "coverage": cov,
            })
            cv2.rectangle(result, (x_left, y_start), (x_right, y_end),
                          (255, 255, 0), 2)
            cv2.line(result, (int(x_center), y_start), (int(x_center), y_end),
                     (0, 0, 255), 1)
            lbl = f"{duct_w}x{length}"
            cv2.putText(result, lbl, (x_right + 5, y_start + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)

    print(f"Ducts found: {len(ducts)}  (H={sum(1 for d in ducts if d['orientation']=='H')}, "
          f"V={sum(1 for d in ducts if d['orientation']=='V')})")

    # Save outputs
    Path("static/rendered").mkdir(parents=True, exist_ok=True)
    cv2.imwrite("static/rendered/dark_mask.png", dark)
    cv2.imwrite("static/rendered/detected_rects.png", result)
    print("Saved: dark_mask.png, detected_rects.png")

    return ducts


if __name__ == "__main__":
    ducts = detect_duct_rects()
    if ducts:
        print("\nDetected ducts:")
        for d in sorted(ducts, key=lambda d: d['length'], reverse=True):
            o = d['orientation']
            print(f"  {o}  len={d['length']:4d}  w={d['width']:2d}  "
                  f"gap={d['gap']:.0f}  cov={d['coverage']:.0%}  "
                  f"at ({d['cx']:.0f}, {d['cy']:.0f})")
