"""
dev_screen_logger.py — July 2025
---------------------------------------------------------------------------
• Cross-platform (macOS / Windows / Linux X11/Wayland)
• Pure-Python equivalents of the Rust crates used:

    screenshots   → mss               (pip install mss pillow)
    rdev          → pynput            (pip install pynput)
    ort-1.16.0    → onnxruntime-1.18  (pip install onnxruntime)
    rusqlite      → sqlite3 (stdlib)
    packet sniffing → scapy           (pip install scapy)

---------------------------------------------------------------------------
Tables written (SQLite):  shots · mouse_clicks · keystrokes · network_requests
---------------------------------------------------------------------------
"""

from __future__ import annotations
import io, threading, queue, sys, os
from PIL import Image
from scapy.all import sniff, Raw, TCP
from datetime import datetime
import sqlite3

from utilities import init_db, build_session, run_mouse_listener, run_keyboard_listener, screen_dimensions, active_title, capture_full, crop_grid, embed, timestamp

# ---------------------------------------------------------------------------#
# Configuration — tweak to taste                                             #
# ---------------------------------------------------------------------------#
DB_FILE         = "screen_log.db"
MODEL_PATH      = "tiny_clip/model.onnx"
JPEG_QUALITY    = 70
SHOT_INTERVAL_S = 600                  # force a screenshot every N seconds
THUMB_SIZE      = (224, 224)           # CLIP image input
GRID_SIZE       = 10                   # 10×10 patch around click
# ---------------------------------------------------------------------------#

def compress_image(img, quality=85):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def get_clip_embedding_from_blob(blob: bytes, session) -> np.ndarray:
    img = Image.open(io.BytesIO(blob)).convert("RGB")
    resized = img.resize((224, 224), Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=85)
    jpeg_bytes = buf.getvalue()
    return embed(session, jpeg_bytes)


# ----------------------------- main routine --------------------------------#

def main():
    conn    = init_db()                        # now has click_sessions
    print("📁 DB path:", os.path.abspath(DB_FILE))
    session = build_session()

    ev_q: "queue.Queue[Event]" = queue.Queue()
    threading.Thread(target=run_mouse_listener,   args=(ev_q,), daemon=True).start()
    threading.Thread(target=run_keyboard_listener,args=(ev_q,), daemon=True).start()

    # ---- session-level state ------------------------------------
    recording      = False        # are we between 1st-click and 2nd-click?
    first_click    = None         # tuple with metadata from click #1
    keys_buffer: list[str] = []   # list of key names pressed so far
    # -------------------------------------------------------------

    print("🖥️  Screen logger running — Ctrl+C to stop.")
    try:
        while True:
            try:
                kind, data = ev_q.get(timeout=2)
            except queue.Empty:
                continue

            # ────────────────────────────────────────────────────
            if kind == "click":
                click_x, click_y, *_ = data
                ts        = timestamp()
                title_now = active_title()

                # capture full screen NOW (before anything changes)
                full_img   = capture_full()
                blob       = compress_image(full_img, JPEG_QUALITY)
                vec        = get_clip_embedding_from_blob(blob, session)

                if not recording:
                    # ---- FIRST click: start a new buffer -------
                    recording   = True
                    keys_buffer = []
                    first_click = (ts, click_x, click_y, title_now, blob, vec.tobytes())
                else:
                    # ---- SECOND click: commit previous buffer --
                    (ts0, x0, y0, title0, blob0, vec0) = first_click
                    key_seq_txt = " ".join(keys_buffer)  # simple space-sep string

                    conn.execute("""INSERT OR IGNORE INTO click_sessions
                                    VALUES (?,?,?,?,?,?,?)""",
                                 (ts0, x0, y0, title0, blob0, vec0, key_seq_txt))
                    conn.commit()

                    # ---- start NEW buffer with this click -----
                    keys_buffer = []
                    first_click = (ts, click_x, click_y, title_now, blob, vec.tobytes())
                    # recording stays True

            # ────────────────────────────────────────────────────
            elif kind == "key":
                (keyname,) = data
                if recording:
                    keys_buffer.append(keyname)

    except KeyboardInterrupt:
        print("\n⏹️  Exiting.")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
