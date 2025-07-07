# screen_logger.py  (July-2025 refactor)
# ---------------------------------------------------------------
import io, queue, threading, sys, os, sqlite3
from typing import List, Tuple, Any
from datetime import datetime
from PIL import Image
from pynput.keyboard import Listener as KeyListener, Key
from pynput.mouse    import Listener as MouseListener

from clip_helpers import (compress_image, embed_image, euclidean_distance,
                          classify_image, summarize_keystrokes)

# -------- config -----------------------------------------------
DB_FILE        = "screen_log.db"
JPEG_QUALITY   = 90
DIST_THRESHOLD = 0.01          # ignore if distance < 0.01
# ---------------------------------------------------------------


# ------------ tiny helpers -------------------------------------
def utc_ts_ms() -> int:
    """Epoch-milliseconds."""
    return int(datetime.utcnow().timestamp() * 1000)


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""CREATE TABLE IF NOT EXISTS click_sessions (
                      ts_first_click INTEGER PRIMARY KEY,
                      screenshot_jpeg BLOB,
                      clip_vector     BLOB,
                      app             TEXT,
                      action          TEXT,
                      key_summary     TEXT)""")
    return conn


def capture_full() -> Image.Image:
    """Grab full virtual screen with mss."""
    import mss
    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[0])
        return Image.frombytes("RGB", shot.size, shot.rgb)


# ----------------------------------------------------------------
# event queue items: ("click", (x, y)) or ("key", str_keyname)
Event = Tuple[str, Tuple[Any, ...]]
ev_q: "queue.Queue[Event]" = queue.Queue()

def run_mouse_listener(q):
    def on_click(x, y, button, pressed):
        if pressed: q.put(("click", (x, y)))
    MouseListener(on_click=on_click).start()

def run_keyboard_listener(q):
    def on_press(k):
        q.put(("key", (str(k),)))
    KeyListener(on_press=on_press).start()

# ----------------------------------------------------------------
def main() -> None:
    conn = init_db()
    run_mouse_listener(ev_q)
    run_keyboard_listener(ev_q)
    print("🖥️  Logger started.  DB:", os.path.abspath(DB_FILE))

    # ---- session state -----------------------------------------
    recording   = False
    prev_vec    = None       # np.ndarray | None
    prev_blob   = None       # JPEG bytes
    keys_buf: List[str] = []
    # ------------------------------------------------------------

    try:
        while True:
            kind, data = ev_q.get()   # blocking

            # -------------- mouse click -------------------------
            if kind == "click":
                # screenshot right now
                img    = capture_full()
                blob   = compress_image(img, JPEG_QUALITY)
                cur_vec = embed_image(img)

                if not recording:
                    # first click → start session
                    recording = True
                    prev_vec, prev_blob = cur_vec, blob
                    keys_buf.clear()
                    continue

                # second+ click → compute distance
                dist = euclidean_distance(prev_vec, cur_vec)
                if dist < DIST_THRESHOLD:
                    # too similar → ignore; keep recording
                    continue

                # enough change → classify & save
                ts      = utc_ts_ms()
                vis_res = classify_image(Image.open(io.BytesIO(prev_blob)))
                key_sum = summarize_keystrokes(keys_buf, top_k=5, return_json=True)

                conn.execute("""INSERT INTO click_sessions
                                VALUES (?,?,?,?,?,?)""",
                             (ts, prev_blob, prev_vec.tobytes(),
                              vis_res["app"], vis_res["action"], key_sum))
                conn.commit()
                print("✔ saved session @", ts, "|", vis_res["app"], "→", vis_res["action"])

                # reset for next session
                prev_vec, prev_blob = cur_vec, blob
                keys_buf.clear()      # start buffering anew

            # -------------- key press ---------------------------
            elif kind == "key" and recording:
                (keyname,) = data
                keys_buf.append(keyname)

    except KeyboardInterrupt:
        print("\n⏹️  Exiting.")
    except Exception as e:
        print("[ERROR]", e, file=sys.stderr)


if __name__ == "__main__":
    main()

