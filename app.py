"""HVAC Duct Annotation System — FastAPI Server"""

import os
import uuid
import shutil
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

import cv2
from duct_detector import detect_ducts

logger = logging.getLogger(__name__)

app = FastAPI(title="HVAC Duct Annotation System")

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RENDERED_DIR = BASE_DIR / "static" / "rendered"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RENDERED_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
DETECTION_TIMEOUT = 120  # seconds

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in (".pdf", ".png", ".jpg", ".jpeg"):
        raise HTTPException(400, "Supported: PDF, PNG, JPG")

    # Check file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large. Max {MAX_FILE_SIZE // (1024*1024)}MB")

    job_id = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{job_id}{ext}"
    save_path.write_bytes(content)

    try:
        return await _process(str(save_path), job_id)
    finally:
        save_path.unlink(missing_ok=True)


@app.get("/api/demo")
async def demo():
    pdf_path = BASE_DIR / "testset2.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, "testset2.pdf not found")
    return await _process(str(pdf_path), "demo")


async def _process(file_path: str, job_id: str) -> JSONResponse:
    try:
        img, w, h, ducts = await asyncio.wait_for(
            asyncio.to_thread(detect_ducts, file_path, 200),
            timeout=DETECTION_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, "Detection timed out. Try a smaller file.")
    except ValueError as e:
        raise HTTPException(400, f"Detection failed: {e}")
    except Exception as e:
        logger.error(f"Detection error: {e}", exc_info=True)
        raise HTTPException(500, "Detection failed unexpectedly")

    img_name = f"{job_id}.png"
    img_path = RENDERED_DIR / img_name
    cv2.imwrite(str(img_path), img)

    return JSONResponse({
        "image_url": f"/static/rendered/{img_name}",
        "image_width": w,
        "image_height": h,
        "ducts": ducts,
    })


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
