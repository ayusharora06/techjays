"""HVAC Duct Annotation System — FastAPI Server"""

import os
import uuid
import shutil
import asyncio
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

import cv2
from duct_detector import detect_ducts

app = FastAPI(title="HVAC Duct Annotation System")

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RENDERED_DIR = BASE_DIR / "static" / "rendered"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RENDERED_DIR.mkdir(parents=True, exist_ok=True)

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

    job_id = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{job_id}{ext}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return await _process(str(save_path), job_id)


@app.get("/api/demo")
async def demo():
    pdf_path = BASE_DIR / "testset2.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, "testset2.pdf not found")
    return await _process(str(pdf_path), "demo")


async def _process(file_path: str, job_id: str) -> JSONResponse:
    img, w, h, ducts = await asyncio.to_thread(detect_ducts, file_path, 200)

    # Save rendered image
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
