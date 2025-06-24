from fastapi.responses import HTMLResponse
from pathlib import Path

def get_ui():
    html = Path("static/index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)
