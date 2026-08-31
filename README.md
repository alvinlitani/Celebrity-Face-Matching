# Celebrity Face Matching

A computer vision app that takes in photo as input and find which celebrity in [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) that you most resemble using face embeddings and nearest-neighbor search. It is built with facenet-pytorch and FAISS. User interface is a Gradio web app.

![Python](https://img.shields.io/badge/Python-v3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.5.1-green?logo=pytorch)
![FAISS](https://img.shields.io/badge/FAISS-v1.13.2-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Demo

![Demo Screenshot](demo.png)

For a live demo, you can have a look at this [HuggingFace page](https://huggingface.co/spaces/aliauw/celebrity-face-matching).

## How It Works

### Phase 1 — Offline Index Building (`embed.py`)
Runs once locally to build the searchable database:

1. Load the **CelebA dataset** (202,599 celebrity images)
2. Standardize each image to 160×160 PIL format
3. Detect and align faces using **MTCNN**
4. Generate a 512-dim embedding per face using **InceptionResnetV1** (VGGFace2 pretrained) 
5. **L2-normalize** all embeddings (enables cosine similarity)
6. Build a **FAISS** index and save to disk

### Phase 2 — Online Inference (`app.py`)
Runs for each uploaded photo:

1. Detect and align face using MTCNN
2. Generate 512-dim embedding via InceptionResnetV1
3. L2-normalize and search the FAISS index for top-k nearest neighbors
4. Filter duplicate results by identity and return ranked matches with similarity scores

## File Breakdown

| File | Role |
|---|---|
| `app.py` | Gradio web app handling inference and UI |
| `embed.py` | Initial pipeline building FAISS index from CelebA |
| `celeba_face_index.faiss` | Pre-built FAISS search index |
| `celeba_identities.npy` | Maps FAISS positions to CelebA identity IDs |
| `identity.txt` | Maps image filenames to identity IDs (same with CelebA `identity_CelebA.txt`) |
| `requirements.txt` | Dependencies |

## Tech Stack

- **[facenet-pytorch](https://github.com/timesler/facenet-pytorch)** - MTCNN face detection + InceptionResnetV1 embeddings
- **[FAISS](https://github.com/facebookresearch/faiss)** - exact nearest-neighbor search
- **[Gradio](https://gradio.app)** - web UI
- **[CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)** - 202k celebrity face images

## Notes

- CelebA uses numeric identity IDs as real celebrity names are not publicly released with the dataset
- Embeddings are 512-dimensional, L2-normalized with similarity scores range from 0 (no match) to 1 (identical)
- Index building (embed.py) takes 2-3 hours on a consumer GPU in my own experience. Index building will be unfeasible/very slow for CPU. 
- Inference runs fine on CPU

## Local Setup

### Prerequisites

- CUDA-capable GPU recommended (required for index building, optional for inference)
- ~3+ GB disk space for the CelebA dataset and generated index files

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Build the index (requires CelebA dataset)
```bash
python embed.py
```

CelebA will auto-download via torchvision on the first run. If the download fails (common due to Google Drive quotas), download manually from [the official page](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and place the files in `./data/celeba/`.

### 3. Run the app
```bash
python app.py
```

## Design Decisions
 
- **Two-model pipeline (MTCNN + InceptionResnetV1)**: MTCNN handles the detection and alignment of faces then converting to a standardized 160×160 crop. InceptionResnetV1 extracts an embedding from the aligned face that represents the identity. Having raw and uncropped images as input into the embedding model will produce noisy vectors polluted by background and clothing.

- **VGGFace2 pretrained weights**: facenet-pytorch offers two pretrained options (CASIA-WebFace and VGGFace2). VGGFace2 was chosen because it has larger training data (3.31 million images versus 494,000 images) and it having great diversity in pose, age, lighting, and ethnicity. This helps the system to work better with the different kinds of uploaded photos.
 
- **Offline/online split**: Building the index can take a long time (2–3 hours on laptop GPU in personal experience) but it only needs to be done once. Putting it in a separate script (embed.py) keeps the main app (app.py) lightweight as it loads the pre-built index on startup and does not need to build the index everytime.
 
- **L2 normalization + IndexFlatIP**: Normalizing all embeddings to the same scale means that a simple dot product gives us the same results as cosine similarity. This lets us use FAISS's IndexFlatIP for fast similarity search without needing any special cosine index. Cosine similarity compares the direction of two embeddings instead of their length which tends to produce more reliable similarity scores.
 
- **Removing identity duplicates**: CelebA has around 20 images per person. Without removing duplicates, a top-4 search could return four different photos of the same person. Instead, the app fetches extra results from FAISS and keeps only the best match per person.

## Potential Improvements
 
- **Named celebrity mapping**: CelebA only provides numeric identity IDs without the person's real name. If the dataset provider updates the dataset with inclusion of names, the app can show the celebrity names instead of just identity numbers.
 
- **Approximate nearest-neighbor search**: The current index uses exact search for every embedding which works fine at 200k images. For larger datasets (millions+), switching to an approximate index like IndexIVFFlat or IndexHNSW will lower search times with a small accuracy tradeoff.
