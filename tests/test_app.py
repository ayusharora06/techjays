"""Tests for the FastAPI app endpoints."""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
from app import app

PDF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "testset2.pdf")
HAS_PDF = os.path.exists(PDF_PATH)

client = TestClient(app)


class TestIndexPage:
    def test_returns_html(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_contains_title(self):
        resp = client.get("/")
        assert "HVAC" in resp.text
        assert "Duct Annotation" in resp.text

    def test_contains_upload_zone(self):
        resp = client.get("/")
        assert "uploadZone" in resp.text

    def test_contains_canvas(self):
        resp = client.get("/")
        assert "<canvas" in resp.text


class TestUploadEndpoint:
    def test_rejects_unsupported_format(self):
        resp = client.post(
            "/api/upload",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 400

    @pytest.mark.skipif(not HAS_PDF, reason="testset2.pdf not found")
    def test_accepts_pdf(self):
        with open(PDF_PATH, "rb") as f:
            resp = client.post(
                "/api/upload",
                files={"file": ("test.pdf", f, "application/pdf")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "image_url" in data
        assert "ducts" in data
        assert "image_width" in data
        assert "image_height" in data

    def test_accepts_png(self):
        # Create a minimal valid PNG (white 100x100 image)
        import cv2
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        _, buf = cv2.imencode(".png", img)
        resp = client.post(
            "/api/upload",
            files={"file": ("test.png", buf.tobytes(), "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "ducts" in data
        assert isinstance(data["ducts"], list)


@pytest.mark.skipif(not HAS_PDF, reason="testset2.pdf not found")
class TestDemoEndpoint:
    def test_demo_returns_result(self):
        resp = client.get("/api/demo")
        assert resp.status_code == 200
        data = resp.json()
        assert "image_url" in data
        assert "ducts" in data
        assert len(data["ducts"]) >= 15

    def test_demo_image_url_accessible(self):
        resp = client.get("/api/demo")
        data = resp.json()
        img_resp = client.get(data["image_url"])
        assert img_resp.status_code == 200

    def test_demo_duct_structure(self):
        resp = client.get("/api/demo")
        data = resp.json()
        for d in data["ducts"]:
            assert "id" in d
            assert "coordinates" in d
            assert "dimension" in d
            assert "pressure_class" in d
            assert "color" in d
