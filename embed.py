import numpy as np
import faiss
import torch
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from facenet_pytorch import MTCNN, InceptionResnetV1
from tqdm import tqdm

# Settings
DATA_DIR = './data'
INDEX_FILE = "./celeba_face_index.faiss"
IDENTITY_FILE = "./celeba_identities.npy"

# Use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using {device}")

# MTCNN is a face detection and alignment model. The input is PIL images. The output is 160x160 tensors from cropped/aligned faces.
mtcnn = MTCNN(image_size=160, margin=0, device=device, keep_all=False)          # keep only one face as CelebA is single person per photo

# InceptionResnetV1 (VGGFace2 pretrained) as a face recognition model will produce embeddings for the faces.
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Custom collation keeping images as PIL objects since MTCNN takes them as input.
# Default DataLoader collation takes all samples and stack them into a single tensor
def pil_collate(batch):
    
    images, identities = zip(*batch)            # separate the tuples
    return list(images), list(identities)       # convert them into lists   


if __name__ == '__main__':

    # CelebA dataset is supposed to be uniformly 218 x 178 pixels.
    # However batch shape error occured when put in directly, so we resize to uniform size to prevent it.
    # ToTensor/Normalize are not used since MTCNN takes in PIL images and not tensors.
    transform = transforms.Compose([transforms.Resize(215, 175)])   # 215 height, 175 width
  
    celeba_dataset = datasets.CelebA(root=DATA_DIR, transform=transform, 
                                     download=True,                 # Download the dataset if it is not there (first time only).
                                     split='all',                   # 'all' images will be used.
                                     target_type='identity')        # 'identity' labels are also needed.

    # Wrapper to handle batch and parallel processing.  
    celeba_loader = DataLoader(celeba_dataset, 
                               batch_size=32,                       # 32 images in one batch.
                               shuffle=False,                       # FAISS index will be similar on every run for debugging and reproducibility purposes.
                               num_workers=4,                       # number of background processes
                               collate_fn=pil_collate)              # custom collation function
    
    print("Data loaded")

    all_embeddings = []
    all_identities = []

    for pil_images, identities in tqdm(celeba_loader, desc="Processing images"):

        # Fast batch MTCNN detection with fall back to single image detection if numpy error occurs.
        try:
            face_tensors = mtcnn(pil_images)
        except ValueError:
            face_tensors = [mtcnn(img) for img in pil_images]
        
        valid_faces = []
        valid_ids = []

        # Only successfully detected faces and their corresponding IDs are kept for embedding.
        for face, identity in zip(face_tensors, identities):
            if face is not None:
                valid_faces.append(face)
                valid_ids.append(identity)

        if not valid_faces:
            continue

        # Stack individual face tensors [C, H, W] into a single [32/N, C, H, W] batch tensor.
        # InceptionResnetV1 takes a single 4D batch tensor as input.
        face_batch = torch.stack(valid_faces).to(device)

        # Skip gradient calculation since only inference required (no learning/weight updates).
        # Convert tensor into embeddings in a batch which is faster than doing it per image
        with torch.no_grad():                                       
            embeddings = resnet(face_batch).cpu().numpy()           # FAISS only takes in numpy arrays

        # all_embeddings stores embeddings in numpy arrays.
        # all_identities stores individual IDs in matching order.
        all_embeddings.append(embeddings)
        all_identities.extend(valid_ids)                # flatten the IDs before storing it

    # Stores the identities into numpy array
    identities_array = np.array(all_identities)

    # Convert the list of arrays into one single array so FAISS can use a single call to add
    embeddings_array = np.vstack(all_embeddings).astype('float32')          # FAISS only takes in float32

    # L2-normalize all embeddings to unit length so vector lengths do not affect inner/dot product. 
    # Without this, vector length could affect search results with longer vectors outscoring shorter ones.
    faiss.normalize_L2(embeddings_array)

    # Create empty index with the proper dimensions and using Inner Product for similarity comparison 
    index = faiss.IndexFlatIP(embeddings_array.shape[1])
    index.add(embeddings_array)            

    # Save the FAISS index and identities for reference 
    faiss.write_index(index, INDEX_FILE)
    np.save(IDENTITY_FILE, identities_array)

    print(f"Saved FAISS index with {index.ntotal} faces detected to {INDEX_FILE}")
    print(f"Saved identity mapping to {IDENTITY_FILE}")
        
    # Copy identity mapping from CelebA dataset for use by app.py
    shutil.copy('./data/celeba/identity_CelebA.txt', './identity.txt')
    print(f"Copied identity mapping to ./identity.txt")
