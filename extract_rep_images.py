# Extract one representative image per CelebA identity into ./rep_images/.
# Run after embed.py. The output folder is what gets uploaded to HuggingFace Spaces.

import os
import shutil

IDENTITY_TXT = "./identity.txt"
IMG_DIR = "./data/celeba/img_align_celeba"
OUTPUT_DIR = "./rep_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pick the first image encountered per identity (same as app.py's id_to_repfile)
seen_ids = set()

with open(IDENTITY_TXT) as f:
    for line in f:
        fname, iid = line.strip().split()
        iid = int(iid)

        if iid not in seen_ids:
            seen_ids.add(iid)
            src = os.path.join(IMG_DIR, fname)
            dst = os.path.join(OUTPUT_DIR, fname)

            if os.path.exists(src):
                shutil.copy(src, dst)

print(f"Copied representative images ({len(seen_ids)} identities) to {OUTPUT_DIR}/")
