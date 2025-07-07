# utilities.py

from __future__ import annotations
import io, sqlite3, threading, time, queue, sys, os, math, ctypes
from datetime import datetime, timezone
from pathlib import Path

import mss #ignore
from PIL import Image, ImageOps         # image resize/JPEG
import numpy as np
import onnxruntime as ort              # ONNX Runtime
from pynput import mouse, keyboard      # global-hook
import platform



# ------------------------------ utility helpers ----------------------------#
DB_FILE         = "screen_log.db"
MODEL_PATH      = "tiny_clip/model.onnx"
JPEG_QUALITY    = 70
SHOT_INTERVAL_S = 600                  # force a screenshot every N seconds
THUMB_SIZE      = (224, 224)           # CLIP image input
GRID_SIZE       = 10                   # 10×10 patch around click


def timestamp() -> int: # int() and miliseconds
    return int(time.time() * 1000)

def screen_dimensions() -> tuple[int, int]:
    with mss.mss() as sct:
        mon = sct.monitors[0]          # primary / virtual bounding box
        return mon["width"], mon["height"]

# -------- active window title (best-effort, platform-specific) ------------ #



def active_title() -> str:
    system = platform.system()

    if system == "Darwin":  # macOS
        from AppKit import NSWorkspace
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        name = app.localizedName()
        return name

    elif system == "Windows":
        import win32gui, win32process, psutil
        hwnd = win32gui.GetForegroundWindow()
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        try:
            proc = psutil.Process(pid)
            name = proc.name()
            title = win32gui.GetWindowText(hwnd)
            return f"{name}: {title}"
        except Exception:
            return "UnknownApp"

    elif system == "Linux":
        import subprocess
        try:
            wmctrl = subprocess.check_output(['xdotool', 'getwindowfocus', 'getwindowname'])
            return wmctrl.decode('utf-8').strip()
        except Exception:
            return "UnknownWindow"

    return "UnknownPlatform"



# ---------------------------------------------------------------------------#


# ----------------------------- ONNX embedding ------------------------------#
def get_clip_embedding_from_blob(blob: bytes, session) -> np.ndarray:
    """Takes a full screen image blob (JPEG or raw RGB), returns CLIP vector."""
    img = Image.open(io.BytesIO(blob)).convert("RGB")  # load from blob
    resized = img.resize((224, 224), Image.Resampling.BICUBIC)
    
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=85)  # or keep as raw RGB
    jpeg_bytes = buf.getvalue()

    return embed(session, jpeg_bytes)

def build_session(model_path: str=MODEL_PATH) -> ort.InferenceSession:
    providers = ['CPUExecutionProvider']
    return ort.InferenceSession(model_path, providers=providers)


def embed(session: ort.InferenceSession, jpeg_bytes: bytes) -> np.ndarray:
    img = (Image.open(io.BytesIO(jpeg_bytes))
           .convert("RGB")
           .resize((224, 224), Image.Resampling.BICUBIC))

    arr = (np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 127.5) - 1.0
    arr = arr[np.newaxis, ...]  # (1, 3, 224, 224)

    ids  = np.zeros((1, 77), dtype=np.int64)
    mask = np.ones((1, 77), dtype=np.int64)

    inputs = {}
    for inp in session.get_inputs():
        n = inp.name
        if "pixel"  in n: inputs[n] = arr
        elif "mask" in n: inputs[n] = mask
        elif "input" in n: inputs[n] = ids

    # Run and fetch *only* image embeddings
    output = session.run(["image_embeds"], inputs)[0]  # shape: (1, 512)

    return output[0]  # shape: (512,)


# ----------------------------- screenshot utils ----------------------------#
def capture_full() -> Image.Image:
    """Return a Pillow RGB image of the full virtual screen."""
    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[0])
        return Image.frombytes("RGB", shot.size, shot.rgb)

def capture_jpeg() -> bytes:
    img = capture_full().resize(THUMB_SIZE, Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()

def compress_image(img: Image.Image, quality: int = 85) -> bytes:
    """Compress a Pillow image to JPEG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def crop_grid(img_full: Image.Image, x:int, y:int) -> bytes:
    left   = max(x - GRID_SIZE//2, 0)
    top    = max(y - GRID_SIZE//2, 0)
    right  = left + GRID_SIZE
    bottom = top  + GRID_SIZE
    grid   = img_full.crop((left, top, right, bottom))
    return grid.tobytes()               # raw RGB 10×10×3 bytes


# ----------------------------- SQLite helper -------------------------------#
def init_db(path: str=DB_FILE) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    cur  = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS click_sessions (
        ts_first_click INTEGER PRIMARY KEY,
        screenshot_jpeg BLOB,
        clip_vector     BLOB,
        app             TEXT,
        action          TEXT,
        key_summary     TEXT
    );

    """)
    conn.commit()
    return conn


# ----------------------------- event handling ------------------------------#
Event = tuple[str, tuple]  # kind, payload

def run_mouse_listener(ev_q: queue.Queue):
    last_pos = (0,0)

    def on_move(x, y):
        nonlocal last_pos
        last_pos = (int(x), int(y))

    def on_click(x, y, button, pressed):
        if pressed and button == mouse.Button.left:
            ev_q.put(("click", (int(x), int(y), *last_pos)))

    with mouse.Listener(on_move=on_move, on_click=on_click) as listener:
        listener.join()  # blocks

def run_keyboard_listener(ev_q: queue.Queue):
    def on_press(key):
        ev_q.put(("key", (str(key),)))
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

