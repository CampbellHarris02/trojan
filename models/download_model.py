# download_model.py

import os
import requests
from pathlib import Path
from tqdm import tqdm

# === Configuration ===
DOWNLOAD_URL = "https://your-link.com/model.onnx"  # 🔁 Replace with actual URL
TARGET_PATH = Path("tiny_clip/model.onnx")

def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            desc=f"⬇️ Downloading {dest.name}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

if __name__ == "__main__":
    if TARGET_PATH.exists():
        print(f"✅ Model already exists at {TARGET_PATH}")
    else:
        print(f"🚀 Starting download from {DOWNLOAD_URL}")
        download_file(DOWNLOAD_URL, TARGET_PATH)
        print("✅ Done!")
