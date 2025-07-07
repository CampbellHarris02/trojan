# prototype/mouse.py

import time
import json
from pynput import mouse
from PIL import ImageGrab
import numpy as np


# Import embed_screen from embedder.py (must be in the same directory)
try:
    from embedder import embed_screen
except ImportError:
    print("Error: Could not import 'embed_screen' from 'embedder.py'.")
    print("Please ensure 'embedder.py' is in the same directory as 'mouse.py'.")
    print("Also ensure the Screen2Vec library and its models are correctly set up.")
    exit(1)

LOG_FILE = 'mouse_click_embedding_log.json'

# Get screen size once at start (assumes static resolution)
screen_width, screen_height = ImageGrab.grab().size

def on_click(x, y, button, pressed):
    if pressed and button == mouse.Button.left:
        timestamp = time.time()
        print(f"Left click at ({x}, {y}). Capturing screen and generating embedding...")

        try:
            # Capture the entire screen
            full_screen_img = ImageGrab.grab()

            # Convert image to RGB if needed
            if full_screen_img.mode != 'RGB':
                full_screen_img = full_screen_img.convert('RGB')

            # Generate the screen embedding
            screen_embedding = embed_screen(full_screen_img)

            # Convert NumPy array to list for JSON serialization
            embedding_list = screen_embedding.tolist()

            # Calculate percentage location
            x_perc = x / screen_width
            y_perc = y / screen_height

            log_entry = {
                'timestamp': timestamp,
                'x': x,
                'y': y,
                'x_perc': x_perc,
                'y_perc': y_perc,
                'screen_embedding': embedding_list
            }

            # Save log entry
            with open(LOG_FILE, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

            print(f"Logged click at ({x},{y}) [{x_perc:.3f}, {y_perc:.3f}] with embedding.")

        except Exception as e:
            print(f"Error during screen capture or embedding: {e}")

print("Listening for mouse clicks (capturing screen and embedding). Press Ctrl+C to stop.")

with mouse.Listener(on_click=on_click) as listener:
    listener.join()

print("Mouse listener stopped.")

