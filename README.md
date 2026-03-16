---
title: Celebrity Face Matching
emoji: 🎭
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Celebrity Face Matching

A computer vision app that finds which CelebA dataset identity you most resemble, using face embeddings and nearest-neighbor search. Built with facenet-pytorch and FAISS, deployed as a Gradio web app.

---

## How It Works

### Phase 1 — Offline Index Building (`embed.py`)
Runs once locally to build the searchable database:

1. Load the **CelebA dataset** (202,599 celebrity images)
2. Standardize each image to 160×160 PIL format
3. Detect and align faces using **MTCNN**
4. Generate a 512-dim embedding per face using **InceptionResnetV1** (VGGFace2 pretrained) — chosen for its strong performance on face verification benchmarks
5. **L2-normalize** all embeddings (enables cosine similarity)
6. Build a **FAISS `IndexFlatIP`** index and save to disk

### Phase 2 — Online Inference (`app.py`)
Runs for each uploaded photo:

1. Detect and align face using MTCNN
2. Generate 512-dim embedding via InceptionResnetV1
3. L2-normalize and search the FAISS index for top-k nearest neighbors
4. Deduplicate results by identity and return ranked matches with similarity scores

---

## File Breakdown

| File | Role |
|---|---|
| `app.py` | Gradio web app — handles inference and UI |
| `embed.py` | Offline pipeline — builds FAISS index from CelebA |
| `celeba_face_index.faiss` | Pre-built FAISS search index |
| `celeba_identities.npy` | Maps FAISS positions → CelebA identity IDs |
| `identity.txt` | Maps image filenames → identity IDs (ships with CelebA as `identity_CelebA.txt`) |
| `requirements.txt` | Dependencies for HF Spaces deployment |

---

## Tech Stack

- **[facenet-pytorch](https://github.com/timesler/facenet-pytorch)** — MTCNN face detection + InceptionResnetV1 embeddings
- **[FAISS](https://github.com/facebookresearch/faiss)** — exact nearest-neighbor search
- **[Gradio](https://gradio.app)** — web UI
- **[CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)** — 202k celebrity face images

---

## Notes

- CelebA uses numeric identity IDs — real celebrity names are not publicly released with the dataset
- Embeddings are 512-dimensional, L2-normalized; similarity scores range from 0 (no match) to 1 (identical), though in practice scores rarely exceed ~0.85 even for the same person
- Index building takes ~2 hours on a consumer GPU and is impractical on CPU; inference runs fine on CPU (just slower)

---

## Local Setup

### Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended (required for index building, optional for inference)
- ~2 GB disk space for the CelebA dataset and generated index files

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Build the index (requires CelebA dataset)
```bash
python embed.py
```

CelebA will auto-download via torchvision on the first run. If the download fails (common due to Google Drive quotas), download manually from [the official page](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and place the files in `./data/celeba/`.

You will also need `identity_CelebA.txt` from the dataset — rename or symlink it to `./identity.txt` so the app can map identities to representative images.

### 3. Run the app
```bash
python app.py
```
