import os, cv2, time, pickle, numpy as np
from retinaface import RetinaFace
from keras_facenet import FaceNet

TEST_DIR = "test_images"
SIM_THRESHOLD = 0.6
DET_SCALE = 0.5

embedder = FaceNet()
with open("embeddings/embeddings.pkl","rb") as f:
    db = pickle.load(f)

def cosine(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

correct = total = 0
det_t, emb_t, match_t = [], [], []
latencies = []

for person in os.listdir(TEST_DIR):
    pdir = os.path.join(TEST_DIR, person)
    if not os.path.isdir(pdir):
        continue

    for imgname in os.listdir(pdir):
        img = cv2.imread(os.path.join(pdir,imgname))
        if img is None:
            continue

        t0 = time.time()
        small = cv2.resize(img,None,fx=DET_SCALE,fy=DET_SCALE)
        dets = RetinaFace.detect_faces(small)
        det_t.append(time.time()-t0)

        if not isinstance(dets,dict):
            continue

        x1,y1,x2,y2 = list(dets.values())[0]["facial_area"]
        x1,y1,x2,y2 = [int(v/DET_SCALE) for v in (x1,y1,x2,y2)]
        face = img[y1:y2,x1:x2]
        face = cv2.resize(face,(160,160))

        t1 = time.time()
        emb = embedder.embeddings([face])[0]
        emb_t.append(time.time()-t1)

        t2 = time.time()
        scores = {n:cosine(emb,r) for n,r in db.items()}
        name = max(scores,key=scores.get)
        match_t.append(time.time()-t2)

        pred = name if scores[name]>SIM_THRESHOLD else "Unknown"
        if pred == person:
            correct += 1
        total += 1

        latencies.append(det_t[-1]+emb_t[-1]+match_t[-1])

print("\n=== PERFORMANCE REPORT ===")
print("Accuracy:", round(correct/total*100,2),"%")
print("Avg Detection Time (ms):", round(np.mean(det_t)*1000,2))
print("Avg Embedding Time (ms):", round(np.mean(emb_t)*1000,2))
print("Avg Matching Time (ms):", round(np.mean(match_t)*1000,2))
print("Total Pipeline (ms):", round(np.mean(latencies)*1000,2))
print("Avg Latency (s):", round(np.mean(latencies),3))
print("Throughput (fps):", round(1/np.mean(latencies),2))