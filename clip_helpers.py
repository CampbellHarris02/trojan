# clip_helpers.py
# ---------------------------------------------------------------------
# Reusable helpers for CLIP-based image & keystroke classification
# ---------------------------------------------------------------------
import io, json, pathlib
from typing import List, Dict

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import CLIPTokenizerFast
from pynput.keyboard import Key  # only for type hints

# ---------------------------------------------------------------------
# 0) Paths (edit to taste)
ROOT          = pathlib.Path(__file__).parent
MODEL_PATH    = ROOT / "tiny_clip/model.onnx"
LABELS_PATH   = ROOT / "labels.json"
JPEG_QUALITY  = 100
# ---------------------------------------------------------------------

# === 1) One-time global objects ======================================
img_session = ort.InferenceSession(str(MODEL_PATH))

tok          = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
with open(LABELS_PATH) as f:
    label_data = json.load(f)

APP_LABELS       = label_data["apps"]
ACT_LABELS       = label_data["actions"]
ACTION_PRIORS    = label_data["action_priors"]
TEXT_LABELS_HIER = label_data["text_labels"]  # hierarchical

# -- text-label flatten + embed ---------------------------------------
_flat_text_labels, _label2cat = [], {}
for cat, items in TEXT_LABELS_HIER.items():
    for lbl in items:
        _flat_text_labels.append(lbl)
        _label2cat[lbl] = cat

def _text_embed(sentences: List[str]) -> np.ndarray:
    toks = tok(sentences, padding=True, return_tensors="np")
    feeds = {
        "input_ids":      toks["input_ids"],
        "attention_mask": toks["attention_mask"],
        "pixel_values":   np.zeros((len(sentences), 3, 224, 224), dtype=np.float32)
    }
    vecs = img_session.run(["text_embeds"], feeds)[0]
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs

_app_vecs  = _text_embed(APP_LABELS)
_act_vecs  = _text_embed(ACT_LABELS)
_text_vecs = _text_embed(_flat_text_labels)
# =====================================================================


# === 2) Image helpers =================================================
def compress_image(img: Image.Image, quality: int = JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def embed_image(pil_img: Image.Image) -> np.ndarray:
    """PIL.Image → 512-D CLIP embedding"""
    arr = (np.asarray(pil_img.resize((224, 224), Image.Resampling.BICUBIC),
                      dtype=np.float32).transpose(2, 0, 1) / 127.5) - 1.0
    arr = arr[np.newaxis, ...]
    feeds = {k.name: arr if "pixel" in k.name else np.zeros((1, 77), np.int64)
             for k in img_session.get_inputs()}
    vec = img_session.run(["image_embeds"], feeds)[0][0]
    return vec / np.linalg.norm(vec)

# ---------------------------------------------------------------------
def _apply_action_priors(app_name: str, act_scores: np.ndarray) -> np.ndarray:
    priors  = ACTION_PRIORS.get(app_name, {})
    boosted = act_scores.copy()
    for act, w in priors.items():
        try:
            j = ACT_LABELS.index(act)
            boosted[j] *= w
        except ValueError:
            pass
    return boosted

def classify_image(pil_img: Image.Image, top_k_apps: int = 3) -> Dict:
    vec        = embed_image(pil_img)
    app_scores = _app_vecs @ vec
    act_scores = _act_vecs @ vec

    top_app_idx = app_scores.argsort()[-top_k_apps:][::-1]
    top_probs   = app_scores[top_app_idx]

    agg_act = np.zeros_like(act_scores)
    for idx, s in zip(top_app_idx, top_probs):
        boosted = _apply_action_priors(APP_LABELS[idx], act_scores)
        agg_act += s * boosted

    best_app = APP_LABELS[top_app_idx[0]]
    best_act = ACT_LABELS[agg_act.argmax()]

    return {
        "app": best_app,
        "action": best_act,
        "confidence": {
            "app": float(app_scores[top_app_idx[0]]),
            "action_raw":  float(act_scores[agg_act.argmax()]),
            "action_boost": float(agg_act.max())
        }
    }
    
def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.linalg.norm(v1 - v2))

# === 3) Keystroke helpers ============================================
def reconstruct(keys: List[str]) -> str:
    """Undo backspaces / enters → plain string."""
    buf = []
    for k in keys:
        if k in ("Key.backspace", "Key.delete"):
            if buf: buf.pop()
        elif k == "Key.enter":
            buf.append("\n")
        elif k.startswith("Key."):
            continue
        else:
            buf.append(k)
    return "".join(buf)

def summarize_keystrokes(keys: List[str], top_k: int = 5,
                         return_json: bool = True) -> Dict | str:
    plain = reconstruct(keys)
    vec   = _text_embed([plain])[0]
    sims  = _text_vecs @ vec

    idx   = sims.argsort()[-top_k:][::-1]
    best  = idx[0]
    summary = {
        "plain_text": plain,
        "best_label": {
            "label":      _flat_text_labels[best],
            "category":   _label2cat[_flat_text_labels[best]],
            "similarity": float(sims[best])
        },
        "top_k": [
            {
                "rank": r + 1,
                "label":      _flat_text_labels[i],
                "category":   _label2cat[_flat_text_labels[i]],
                "similarity": float(sims[i])
            } for r, i in enumerate(idx)
        ]
    }
    return json.dumps(summary, ensure_ascii=False, indent=2) if return_json else summary



