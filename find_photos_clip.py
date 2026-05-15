"""
Photo Similarity Finder (Face + CLIP)

1. Place 2-5 reference photos of the person or subject you want to find in './reference_photos'.
2. Set the folder path to your photo collection below.
3. Run this script. Matching photos will be copied to './similar_photos'.
"""

import os
import sys
import face_recognition
from PIL import Image
import shutil
from pathlib import Path
import torch
import clip
import numpy as np

# --- CONFIGURATION ---
REFERENCE_FOLDER = './reference_photos'  # Folder with reference images
PHOTO_FOLDER = '/Volumes/_Backups/4TDrive/EaseUS 03-24 0934'  # Folder with photos to scan
OUTPUT_FOLDER = './similar_photos'  # Where to copy matching photos
SIMILARITY_THRESHOLD = 0.5  # Face recognition threshold
CLIP_SIMILARITY_THRESHOLD = 0.28  # Lower is stricter (0.25-0.3 is typical for CLIP)

# --- LOAD CLIP MODEL ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model, preprocess = clip.load('ViT-B/32', device=device)

# --- LOAD REFERENCE ENCODINGS ---
print('Loading reference images...')
reference_encodings = []
reference_clip_features = []
for ref_file in Path(REFERENCE_FOLDER).glob('*'):
    try:
        img = face_recognition.load_image_file(ref_file)
        encs = face_recognition.face_encodings(img)
        if encs:
            reference_encodings.append(encs[0])
            print(f'Loaded reference (face): {ref_file}')
        else:
            print(f'No face found in reference: {ref_file}')
        # CLIP features
        pil_img = preprocess(Image.open(ref_file)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = clip_model.encode_image(pil_img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            reference_clip_features.append(feat.cpu().numpy()[0])
    except Exception as e:
        print(f'Error loading {ref_file}: {e}')

if not reference_encodings or not reference_clip_features:
    print('No valid reference faces/features found. Exiting.')
    sys.exit(1)

# --- SCAN PHOTOS ---
print('Scanning photos...')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

photo_files = list(Path(PHOTO_FOLDER).rglob('*'))
photo_files = [f for f in photo_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

for idx, photo_path in enumerate(photo_files):
    try:
        img = face_recognition.load_image_file(photo_path)
        encs = face_recognition.face_encodings(img)
        found = False
        # Step 1: Face recognition
        for enc in encs:
            matches = face_recognition.compare_faces(reference_encodings, enc, tolerance=SIMILARITY_THRESHOLD)
            if any(matches):
                found = True
                break
        # Step 2: CLIP similarity (if not found by face, or always)
        pil_img = preprocess(Image.open(photo_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = clip_model.encode_image(pil_img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            feat = feat.cpu().numpy()[0]
        similarities = [np.dot(feat, ref_feat) for ref_feat in reference_clip_features]
        max_sim = max(similarities)
        if found or max_sim > CLIP_SIMILARITY_THRESHOLD:
            dest = Path(OUTPUT_FOLDER) / photo_path.name
            shutil.copy2(photo_path, dest)
            print(f'MATCH: {photo_path} (CLIP={max_sim:.3f}) -> {dest}')
        else:
            print(f'No match: {photo_path} (CLIP={max_sim:.3f})')
    except Exception as e:
        print(f'Error processing {photo_path}: {e}')

print('Done! Matching photos are in', OUTPUT_FOLDER)