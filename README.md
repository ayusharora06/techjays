# HVAC Duct Annotation System

Detects ducts in HVAC mechanical floor plans, annotates them with colored centerlines based on pressure class, and shows duct dimensions on click.

## How It Works

1. **PDF/Image Upload** — Renders PDF at 200dpi or reads PNG/JPG directly
2. **OpenCV Detection** — Morphological close+open finds elongated rectangular duct shapes
3. **GPT-4o Vision** — Reads duct dimensions and types from the drawing via grid overlay
4. **Pressure Classification** — Classifies each duct by dimension:
   - **Red** (High) — ≥18" ducts
   - **Orange** (Medium) — 12"–17" ducts
   - **Blue** (Low) — <12" ducts
5. **Interactive UI** — Click any duct line to see dimension, type, and pressure class

## Setup

```bash
# Clone
git clone https://github.com/ayusharora06/techjays.git
cd techjays

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Run

```bash
python app.py
```

Open http://localhost:8000 — upload a PDF/PNG or click "Load Sample Drawing".

## Test

```bash
python -m pytest tests/ -v
```

## Project Structure

```
├── app.py                  # FastAPI server
├── duct_detector.py        # OpenCV detection + GPT-4o metadata + config
├── templates/
│   └── index.html          # Frontend SPA (canvas + tooltips)
├── tests/
│   ├── test_duct_detector.py   # 60 unit + integration tests
│   └── test_app.py             # 10 API endpoint tests
├── testset2.pdf            # Sample HVAC floor plan
├── video/                  # Demo recording
├── requirements.txt
└── .env.example
```

## Configuration

All detection parameters are in `DuctDetectorConfig` dataclass (`duct_detector.py`). Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dpi` | 200 | PDF render resolution |
| `dark_threshold` | 50 | Pixel intensity cutoff for duct outlines |
| `morph_close_size` | 70 | Kernel size to fill gaps between parallel walls |
| `morph_open_size` | 80 | Kernel size to keep only elongated strips |
| `min_aspect_ratio` | 2.5 | Minimum length/width ratio for a duct |
| `pressure_high_min` | 18.0 | Dimension threshold for high pressure (inches) |
| `pressure_medium_min` | 12.0 | Dimension threshold for medium pressure (inches) |

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/upload` | POST | Upload PDF/PNG/JPG, returns annotated image + duct data |

Response format:
```json
{
  "image_url": "/static/rendered/demo.png",
  "image_width": 7200,
  "image_height": 4800,
  "ducts": [
    {
      "id": 0,
      "coordinates": [{"x": 3592, "y": 1462}, {"x": 4624, "y": 1462}],
      "dimension": "22\"x14\"",
      "duct_type": "supply",
      "pressure_class": "high",
      "color": "#FF0000",
      "description": "Supply Air duct — high pressure"
    }
  ]
}
```

## Tech Stack

- **Backend**: FastAPI, OpenCV, PyMuPDF, OpenAI GPT-4o
- **Frontend**: Vanilla JS, Canvas API
- **Testing**: pytest
