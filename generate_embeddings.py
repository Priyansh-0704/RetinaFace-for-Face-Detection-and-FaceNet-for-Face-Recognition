import os
import cv2
import pickle
import numpy as np
from retinaface import RetinaFace
from keras_facenet import FaceNet

DATASET_DIR = "dataset"
OUT_PATH = "embeddings/embeddings.pkl"

embedder = FaceNet()

def get_embedding(img):
    dets = RetinaFace.detect_faces(img)
    if not isinstance(dets, dict):
        return None
    x1, y1, x2, y2 = list(dets.values())[0]["facial_area"]
    face = img[y1:y2, x1:x2]
    if face.size == 0:
        return None
    face = cv2.resize(face, (160,160))
    return embedder.embeddings([face])[0]

db = {}

for person in os.listdir(DATASET_DIR):
    pdir = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(pdir):
        continue

    embs = []
    for imgname in os.listdir(pdir):
        img = cv2.imread(os.path.join(pdir, imgname))
        if img is None:
            continue
        emb = get_embedding(img)
        if emb is not None:
            embs.append(emb)

    if embs:
        db[person] = np.mean(embs, axis=0)
        print(f"{person}: {len(embs)} images")

os.makedirs("embeddings", exist_ok=True)
with open(OUT_PATH, "wb") as f:
    pickle.dump(db, f)

print("Embeddings saved")