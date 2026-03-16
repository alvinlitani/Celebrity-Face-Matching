import numpy as np
import faiss
import torch
import gradio
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# Config
INDEX_FILE     = "./celeba_face_index.faiss"
IDENTITY_FILE  = "./celeba_identities.npy"
IDENTITY_TXT   = "./identity.txt"
IMG_DIR        = "./rep_images"         # pre-extracted representative images for HuggingFace Spaces

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load FAISS index and the parallel identity array
index = faiss.read_index(INDEX_FILE)
identities_array = np.load(IDENTITY_FILE)
print("Index loaded.")

# Build identity_id -> representative filename lookup dictionary.
id_to_repfile = {}

# Use the first image encountered per identity as its gallery thumbnail.
with open(IDENTITY_TXT) as f:
    for line in f:

        # format is "000001.jpg 2880"
        fname, iid = line.strip().split()
        iid = int(iid)

        # Only adds to dictionary if entry does not exist yet
        if iid not in id_to_repfile:
            id_to_repfile[iid] = fname

# Load models
# Keep all faces as user may input group photo and largest face will be chosen later for detection
mtcnn = MTCNN(image_size=160, margin=0, device=device, keep_all=True)           
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("Models loaded.")

# Take user image converted to numpy as input. 
# Return top_n most similar CelebA identities (gallery_items, table_rows) as output.
def match(img_np, top_n: int):
    
    if img_np is None:
        raise gradio.Error("Please upload an image first.")

    img_pil = Image.fromarray(img_np)

    # Detect all bounding boxes 
    boxes, _ = mtcnn.detect(img_pil)
    if boxes is None:
        raise gradio.Error("No face detected. Try a clearer, well-lit, front-facing photo.")

    # Extract the aligned face tensors using the boxes
    faces = mtcnn.extract(img_pil, boxes, save_path=None)
    if faces is None:           # Error handling purpose but this should never happen 
        raise gradio.Error("No face detected. Try a clearer, well-lit, front-facing photo.")
 
    # If only one face detected, use it directly. Otherwise pick the largest face by bounding-box area.
    if len(boxes) == 1:
        face = faces[0].unsqueeze(0).to(device)         # return [1, C, H, W] for the model
    else:
        # Box is in [x1, y1, x2, y2] format. Area is calculated by (x2 - x1) * (y2 - y1).
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        largest_idx = int(np.argmax(areas))
        face = faces[largest_idx].unsqueeze(0).to(device)

    # Skip gradient calculation since only inference required (no learning/weight updates).
    # Convert tensor into embedding for the selected face 
    with torch.no_grad():
        embedding = resnet(face).cpu().numpy().astype('float32')

    # Reshape to 2D [1, 512] for error prevention as FAISS takes 2D array as input
    vector = embedding.reshape(1, -1)

    # L2-normalize all embeddings to unit length so vector lengths do not affect inner/dot product. 
    # Without this, vector length could affect search results with longer vectors outscoring shorter ones.
    faiss.normalize_L2(vector)

    # Search FAISS for matching faces. Search in excess as duplicate identities are in the results.
    distances, indices = index.search(vector, 100)             # get the 100 closest images

    # Filtering to keep only the best match per identity
    seen_ids = set()
    results = []

    # Get the celebrity identity from the search results
    # Only one face is queried so take from the first and only row
    for dist, idx in zip(distances[0], indices[0]):
        
        if idx == -1:  # FAISS basically returning "no result"
            continue
        
        # Get the identity from the array
        identity_id = int(identities_array[idx])

        # Only adds if identity is not added yet
        if identity_id not in seen_ids:
            seen_ids.add(identity_id)
            results.append((identity_id, float(dist)))
        
        if len(results) >= top_n:
            break

    # Build gallery images and score table
    gallery_items = []
    table_rows = []
    
    # Format the search results into expected Gradio format
    for rank, (identity_id, score) in enumerate(results, start=1):

        label = f"Identity #{identity_id}"

        rep_file = id_to_repfile.get(identity_id)
        
        img_path = f"{IMG_DIR}/{rep_file}"

        gallery_items.append((img_path, f"{label}  |  Score: {score:.4f}"))
        table_rows.append([rank, label, f"{score:.4f}"])

    return gallery_items, table_rows


# Gradio UI
with gradio.Blocks(title="Celebrity Face Matching") as demo:

    gradio.Markdown(
        """
        # Celebrity Face Matching
        Upload a photo to find which [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
        dataset identities you most resemble, ranked by **cosine similarity**.

        > **Note:** CelebA uses numeric identity IDs — real celebrity names are not publicly
        > released with the dataset.
        """
    )

    with gradio.Row():

        with gradio.Column(scale=1):

            input_image = gradio.Image(type="numpy", label="Your photo", sources=["upload", "webcam"], height=400)

            top_n_slider = gradio.Slider(minimum=1, maximum=10, value=4, step=1, label="Number of matches")

            submit_btn = gradio.Button("Find my celebrity match", variant="primary")

        with gradio.Column(scale=2):

            gallery_output = gradio.Gallery(label="Top matches", columns=4, height=400, object_fit="cover", show_label=True)

    table_output = gradio.Dataframe(headers=["Rank", "Identity", "Cosine similarity"], label="Match Details", interactive=False)

    with gradio.Accordion("Score Guideline", open=True):

        gradio.Markdown(
            """
            | Score | Meaning |
            |---|---|
            | > 0.70 | Very strong facial resemblance |
            | 0.50 – 0.70 | Moderate resemblance |
            | 0.30 – 0.50 | Partial resemblance |
            | < 0.30 | Weak match (closest available in database) |
            """
        )

    submit_btn.click(fn=match, inputs=[input_image, top_n_slider], outputs=[gallery_output, table_output])

if __name__ == "__main__":
    demo.launch()
