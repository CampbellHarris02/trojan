from pynput import keyboard
import time, json
from PIL import ImageGrab
import embedder  # your module above

LOG_FILE = 'key_context_log.json'

def get_embedding():
    img = ImageGrab.grab()  # full screen or window region
    vec = embedder.embed_screen(img)
    return vec.tolist()

def log_key_event(key):
    ts = time.time()
    embedding = get_embedding()
    entry = {
        'timestamp': ts,
        'key': str(key),
        'screen2vec': embedding,
    }
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(entry) + '\n')

with keyboard.Listener(on_press=log_key_event) as listener:
    listener.join()
