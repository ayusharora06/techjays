"""Tests for duct_detector.py — unit tests for pure functions + integration test on testset2.pdf."""

import os
import sys
import math
import json
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from duct_detector import (
    DuctDetectorConfig,
    DEFAULT_CONFIG,
    _classify_pressure,
    _estimate_dimension,
    _parse_json_response,
    _deduplicate,
    _find_duct_rectangles,
    _match_pressure,
    _get_nearest_metadata,
    _render_pdf,
    detect_ducts,
    PRESSURE_BGR,
    PRESSURE_HEX,
)

PDF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "testset2.pdf")
HAS_PDF = os.path.exists(PDF_PATH)


# ─── _classify_pressure ──────────────────────────────────────────────────────

class TestClassifyPressure:
    def test_high_large_dimension(self):
        assert _classify_pressure('22"x14"') == "high"

    def test_high_18_inch(self):
        assert _classify_pressure('18"ø') == "high"

    def test_medium_14_inch(self):
        assert _classify_pressure('14"ø') == "medium"

    def test_medium_12_inch(self):
        assert _classify_pressure('12"') == "medium"

    def test_low_8_inch(self):
        assert _classify_pressure('8"ø') == "low"

    def test_low_unknown(self):
        assert _classify_pressure("Unknown") == "low"

    def test_low_empty(self):
        assert _classify_pressure("") == "low"

    def test_low_none(self):
        assert _classify_pressure(None) == "low"

    def test_high_boundary_18(self):
        assert _classify_pressure("18") == "high"

    def test_medium_boundary_12(self):
        assert _classify_pressure("12") == "medium"

    def test_low_boundary_11(self):
        assert _classify_pressure("11") == "low"


# ─── _estimate_dimension ─────────────────────────────────────────────────────

class TestEstimateDimension:
    def test_large_duct(self):
        duct = {"width": 150.0}
        assert _estimate_dimension(duct) == '22"x14"'

    def test_18_inch(self):
        duct = {"width": 80.0}
        assert _estimate_dimension(duct) == '18"ø'

    def test_14_inch(self):
        duct = {"width": 50.0}
        assert _estimate_dimension(duct) == '14"ø'

    def test_12_inch(self):
        duct = {"width": 30.0}
        assert _estimate_dimension(duct) == '12"'

    def test_8_inch(self):
        duct = {"width": 20.0}
        assert _estimate_dimension(duct) == '8"ø'

    def test_zero_width(self):
        duct = {"width": 0}
        assert _estimate_dimension(duct) == '8"ø'

    def test_missing_width(self):
        duct = {}
        assert _estimate_dimension(duct) == '8"ø'

    def test_boundary_120(self):
        assert _estimate_dimension({"width": 120.0}) == '22"x14"'

    def test_boundary_65(self):
        assert _estimate_dimension({"width": 65.0}) == '18"ø'

    def test_boundary_40(self):
        assert _estimate_dimension({"width": 40.0}) == '14"ø'

    def test_boundary_25(self):
        assert _estimate_dimension({"width": 25.0}) == '12"'


# ─── _parse_json_response ────────────────────────────────────────────────────

class TestParseJsonResponse:
    def test_clean_json_array(self):
        raw = '[{"cell":"B3","dimension":"14\\"","duct_type":"supply"}]'
        result = _parse_json_response(raw)
        assert len(result) == 1
        assert result[0]["cell"] == "B3"

    def test_json_in_code_block(self):
        raw = '```json\n[{"cell":"A1","dimension":"8\\"","duct_type":"return"}]\n```'
        result = _parse_json_response(raw)
        assert len(result) == 1
        assert result[0]["cell"] == "A1"

    def test_json_with_surrounding_text(self):
        raw = 'Here are the ducts:\n[{"cell":"C2","dimension":"12\\"","duct_type":"supply"}]\nThat is all.'
        result = _parse_json_response(raw)
        assert len(result) == 1

    def test_empty_array(self):
        assert _parse_json_response("[]") == []

    def test_invalid_json(self):
        assert _parse_json_response("not json at all") == []

    def test_empty_string(self):
        assert _parse_json_response("") == []

    def test_multiple_entries(self):
        raw = '[{"cell":"A1"},{"cell":"B2"},{"cell":"C3"}]'
        result = _parse_json_response(raw)
        assert len(result) == 3


# ─── _deduplicate ────────────────────────────────────────────────────────────

class TestDeduplicate:
    def test_no_duplicates(self):
        ducts = [
            {"orientation": "H", "x1": 100, "y1": 200, "x2": 500, "y2": 200, "length": 400, "cx": 300, "cy": 200},
            {"orientation": "V", "x1": 600, "y1": 100, "x2": 600, "y2": 500, "length": 400, "cx": 600, "cy": 300},
        ]
        result = _deduplicate(ducts)
        assert len(result) == 2

    def test_overlapping_horizontal(self):
        ducts = [
            {"orientation": "H", "x1": 100, "y1": 200, "x2": 500, "y2": 200, "length": 400, "cx": 300, "cy": 200},
            {"orientation": "H", "x1": 120, "y1": 205, "x2": 480, "y2": 205, "length": 360, "cx": 300, "cy": 205},
        ]
        result = _deduplicate(ducts)
        assert len(result) == 1

    def test_overlapping_vertical(self):
        ducts = [
            {"orientation": "V", "x1": 300, "y1": 100, "x2": 300, "y2": 500, "length": 400, "cx": 300, "cy": 300},
            {"orientation": "V", "x1": 310, "y1": 110, "x2": 310, "y2": 490, "length": 380, "cx": 310, "cy": 300},
        ]
        result = _deduplicate(ducts)
        assert len(result) == 1

    def test_different_orientations_not_deduped(self):
        ducts = [
            {"orientation": "H", "x1": 100, "y1": 200, "x2": 500, "y2": 200, "length": 400, "cx": 300, "cy": 200},
            {"orientation": "V", "x1": 300, "y1": 100, "x2": 300, "y2": 500, "length": 400, "cx": 300, "cy": 300},
        ]
        result = _deduplicate(ducts)
        assert len(result) == 2

    def test_far_apart_horizontal_kept(self):
        ducts = [
            {"orientation": "H", "x1": 100, "y1": 200, "x2": 500, "y2": 200, "length": 400, "cx": 300, "cy": 200},
            {"orientation": "H", "x1": 100, "y1": 500, "x2": 500, "y2": 500, "length": 400, "cx": 300, "cy": 500},
        ]
        result = _deduplicate(ducts)
        assert len(result) == 2

    def test_empty_list(self):
        assert _deduplicate([]) == []

    def test_single_duct(self):
        ducts = [{"orientation": "H", "x1": 0, "y1": 0, "x2": 100, "y2": 0, "length": 100, "cx": 50, "cy": 0}]
        assert len(_deduplicate(ducts)) == 1


# ─── _match_pressure (without metadata — fallback) ──────────────────────────

class TestMatchPressureFallback:
    def test_large_duct_high(self):
        duct = {"cx": 100.0, "cy": 100.0, "width": 150.0}
        assert _match_pressure(duct, [], 3000, 7000) == "high"

    def test_medium_duct(self):
        duct = {"cx": 100.0, "cy": 100.0, "width": 50.0}
        assert _match_pressure(duct, [], 3000, 7000) == "medium"

    def test_small_duct_low(self):
        duct = {"cx": 100.0, "cy": 100.0, "width": 20.0}
        assert _match_pressure(duct, [], 3000, 7000) == "low"


# ─── _match_pressure (with metadata) ────────────────────────────────────────

class TestMatchPressureWithMetadata:
    def test_matches_nearest_cell(self):
        duct = {"cx": 500.0, "cy": 500.0, "width": 20.0}
        metadata = [
            {"cx": 510.0, "cy": 510.0, "dimension": '22"x14"', "duct_type": "supply"},
            {"cx": 2000.0, "cy": 2000.0, "dimension": '8"', "duct_type": "return"},
        ]
        assert _match_pressure(duct, metadata, 3000, 7000) == "high"

    def test_too_far_falls_back(self):
        duct = {"cx": 100.0, "cy": 100.0, "width": 20.0}
        metadata = [
            {"cx": 6000.0, "cy": 2500.0, "dimension": '22"x14"', "duct_type": "supply"},
        ]
        # Distance > max(3000,7000)*0.3 = 2100, so falls back to pixel width
        assert _match_pressure(duct, metadata, 3000, 7000) == "low"


# ─── _get_nearest_metadata ───────────────────────────────────────────────────

class TestGetNearestMetadata:
    def test_no_metadata_fallback(self):
        duct = {"cx": 100.0, "cy": 100.0, "width": 80.0}
        dim, dtype = _get_nearest_metadata(duct, [], 3000, 7000)
        assert dim == '18"ø'
        assert dtype == "supply"

    def test_matches_nearest(self):
        duct = {"cx": 500.0, "cy": 500.0, "width": 20.0}
        metadata = [
            {"cx": 520.0, "cy": 480.0, "dimension": '14"ø', "duct_type": "return"},
        ]
        dim, dtype = _get_nearest_metadata(duct, metadata, 3000, 7000)
        assert dim == '14"ø'
        assert dtype == "return"


# ─── _find_duct_rectangles (synthetic image) ─────────────────────────────────

class TestFindDuctRectanglesSynthetic:
    def _make_duct_image(self, width=2000, height=1000):
        """Create a white image with two synthetic duct rectangles drawn on it."""
        img = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Horizontal duct: 2 parallel black lines, 500px long, 40px apart
        cv2.line(img, (200, 300), (800, 300), (0, 0, 0), 2)
        cv2.line(img, (200, 340), (800, 340), (0, 0, 0), 2)

        # Vertical duct: 2 parallel black lines, 400px long, 30px apart
        cv2.line(img, (1200, 200), (1200, 700), (0, 0, 0), 2)
        cv2.line(img, (1230, 200), (1230, 700), (0, 0, 0), 2)

        return img

    def test_detects_horizontal_duct(self):
        img = self._make_duct_image()
        ducts = _find_duct_rectangles(img)
        h_ducts = [d for d in ducts if d["orientation"] == "H"]
        assert len(h_ducts) >= 1
        d = h_ducts[0]
        assert d["x1"] < d["x2"]
        assert abs(d["y1"] - d["y2"]) < 5  # horizontal line

    def test_detects_vertical_duct(self):
        img = self._make_duct_image()
        ducts = _find_duct_rectangles(img)
        v_ducts = [d for d in ducts if d["orientation"] == "V"]
        assert len(v_ducts) >= 1
        d = v_ducts[0]
        assert d["y1"] < d["y2"]
        assert abs(d["x1"] - d["x2"]) < 5  # vertical line

    def test_blank_image_no_ducts(self):
        img = np.ones((1000, 2000, 3), dtype=np.uint8) * 255
        ducts = _find_duct_rectangles(img)
        assert len(ducts) == 0

    def test_duct_keys_present(self):
        img = self._make_duct_image()
        ducts = _find_duct_rectangles(img)
        if ducts:
            d = ducts[0]
            for key in ["orientation", "x1", "y1", "x2", "y2", "cx", "cy", "length", "width"]:
                assert key in d


# ─── Color mappings ──────────────────────────────────────────────────────────

class TestColorMappings:
    def test_pressure_bgr_has_all_keys(self):
        for key in ["high", "medium", "low"]:
            assert key in PRESSURE_BGR
            assert len(PRESSURE_BGR[key]) == 3

    def test_pressure_hex_has_all_keys(self):
        for key in ["high", "medium", "low"]:
            assert key in PRESSURE_HEX
            assert PRESSURE_HEX[key].startswith("#")
            assert len(PRESSURE_HEX[key]) == 7

    def test_bgr_hex_consistency(self):
        # high BGR (0,0,255) should correspond to hex #FF0000
        assert PRESSURE_HEX["high"] == "#FF0000"
        assert PRESSURE_HEX["medium"] == "#FF8C00"
        assert PRESSURE_HEX["low"] == "#4488FF"


# ─── Integration: testset2.pdf ────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_PDF, reason="testset2.pdf not found")
class TestIntegrationPDF:
    @pytest.fixture(scope="class")
    def detection_result(self):
        """Run detection once, share across tests in this class."""
        # Temporarily unset OPENAI_API_KEY to test without GPT
        key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            img, w, h, ducts = detect_ducts(PDF_PATH, dpi=200)
        finally:
            if key is not None:
                os.environ["OPENAI_API_KEY"] = key
        return img, w, h, ducts

    def test_image_dimensions(self, detection_result):
        img, w, h, ducts = detection_result
        assert w == 7200
        assert h == 4800

    def test_returns_image(self, detection_result):
        img, w, h, ducts = detection_result
        assert img is not None
        assert img.shape[0] == h
        assert img.shape[1] == w
        assert img.shape[2] == 3

    def test_detects_ducts(self, detection_result):
        img, w, h, ducts = detection_result
        assert len(ducts) >= 15  # should find at least 15 ducts

    def test_duct_data_structure(self, detection_result):
        img, w, h, ducts = detection_result
        for d in ducts:
            assert "id" in d
            assert "coordinates" in d
            assert len(d["coordinates"]) == 2
            assert "x" in d["coordinates"][0]
            assert "y" in d["coordinates"][0]
            assert "dimension" in d
            assert "duct_type" in d
            assert "pressure_class" in d
            assert d["pressure_class"] in ("high", "medium", "low")
            assert "color" in d
            assert d["color"].startswith("#")
            assert "description" in d

    def test_duct_data_json_serializable(self, detection_result):
        img, w, h, ducts = detection_result
        serialized = json.dumps(ducts)
        assert isinstance(serialized, str)
        parsed = json.loads(serialized)
        assert len(parsed) == len(ducts)

    def test_coordinates_within_image(self, detection_result):
        img, w, h, ducts = detection_result
        for d in ducts:
            for pt in d["coordinates"]:
                assert 0 <= pt["x"] <= w
                assert 0 <= pt["y"] <= h

    def test_has_multiple_pressure_classes(self, detection_result):
        """Fallback should produce varied pressure classes based on duct width."""
        img, w, h, ducts = detection_result
        classes = set(d["pressure_class"] for d in ducts)
        assert len(classes) >= 2  # at least 2 different pressure classes

    def test_has_horizontal_and_vertical(self, detection_result):
        """Should detect both horizontal and vertical ducts."""
        img, w, h, ducts = detection_result
        # Check via coordinates: H ducts have same y, V ducts have same x
        h_count = sum(1 for d in ducts if d["coordinates"][0]["y"] == d["coordinates"][1]["y"])
        v_count = sum(1 for d in ducts if d["coordinates"][0]["x"] == d["coordinates"][1]["x"])
        assert h_count >= 5
        assert v_count >= 3


@pytest.mark.skipif(not HAS_PDF, reason="testset2.pdf not found")
class TestRenderPDF:
    def test_render_returns_image(self):
        img = _render_pdf(PDF_PATH, 200)
        assert img is not None
        assert len(img.shape) == 3
        assert img.shape[2] == 3

    def test_render_dimensions_at_200dpi(self):
        img = _render_pdf(PDF_PATH, 200)
        h, w = img.shape[:2]
        assert w > 5000  # should be large at 200dpi
        assert h > 3000
