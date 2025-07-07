# trojan 🐴 

> *Can automation help you?*

I'm on a mission to eliminate repetitive digital tasks — the kind that drain human time, focus, and creativity.

---

### Why?

Over the past four summers as a student intern, I experienced first-hand how tedious, manual processes can be. In each role, I managed to automate huge parts of my job. In one case, what started as a **full day of work** became just **10 minutes** by the end of the summer — thanks to Python pipelines I built from scratch.

With the rise of AI, the opportunity to automate even more of these tasks is massive.

But here’s the strange thing:

> **If the technology exists to automate almost all back-office work...  
> why are millions of people still doing it manually?**

---

### My Theory

The friction isn’t technical — it’s cognitive.

You can’t automate what you don’t understand.  
In my case, I had to do the tasks over and over again to spot the patterns and figure out how to automate them.

---

### This Project

**Trojan** is an experimental Python tool that acts like a friendly little ghost living inside your computer.

It will:

- Monitor mouse clicks  
- Track keystrokes  
- Observe screen activity  

All of this runs **quietly in the background**, learning how I work — so it can detect **repetitive behavior** and help design automations.

Think of it as a **Trojan Horse for productivity**: infiltrating the system not to cause harm, but to discover where our time is slipping away — and give it back to us.

---

### Inspired By

One magical library that blew my mind was [`self-operating-computer`](https://github.com/OthersideAI/self-operating-computer) — it’s like watching a ghost operate your machine right before your eyes. That’s the level of magic I want this project to reach.

---

### Goals

- Build a passive logging engine
- Analyze user behavior over time
- Automatically identify tasks that could be automated
- Prototype automation plans using AI/LLMs

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/CampbellHarris02/trojan.git
cd trojan
```

### 2. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## ⚠️ ONNX Model Not Included

This project requires a separate download of the ONNX model used for embedding.

> GitHub doesn’t allow files larger than 100MB, so `python/tiny_clip/model.onnx` has been excluded.

### ⬇️ Download <code>tiny_clip</code> Model:

<url>"https://huggingface.co/valhalla/clip-vit-base-patch32-onnx/resolve/main/model.onnx"</url>

place the downloaded model in the models folder:

```bash
models/tiny_clip/model.onnx
```
