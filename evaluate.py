import os, cv2, time, pickle, numpy as np
from retinaface import RetinaFace
from keras_facenet import FaceNet

TEST_DIR = "test_images"
EMB_PATH = "embeddings/embeddings.pkl"
SIM_THRESHOLD = 0.6
DET_SCALE = 0.5

embedder = FaceNet()
with open(EMB_PATH, "rb") as f:
    db = pickle.load(f)

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

total = correct = 0

known_total = known_correct = 0
unknown_total = unknown_correct = 0

per_person = {}  # {name: {"correct": x, "total": y}}

det_t, emb_t, match_t = [], [], []
latencies = []

for person in os.listdir(TEST_DIR):
    person_dir = os.path.join(TEST_DIR, person)
    if not os.path.isdir(person_dir):
        continue

    per_person.setdefault(person, {"correct": 0, "total": 0})

    for img_name in os.listdir(person_dir):
        img = cv2.imread(os.path.join(person_dir, img_name))
        if img is None:
            continue

        t0 = time.time()
        small = cv2.resize(img, None, fx=DET_SCALE, fy=DET_SCALE)
        dets = RetinaFace.detect_faces(small)
        det_time = time.time() - t0
        det_t.append(det_time)

        predicted = "Unknown"

        if isinstance(dets, dict):
            x1, y1, x2, y2 = list(dets.values())[0]["facial_area"]
            x1, y1, x2, y2 = [int(v / DET_SCALE) for v in (x1, y1, x2, y2)]
            face = img[y1:y2, x1:x2]

            if face.size != 0:
                face = cv2.resize(face, (160, 160))

                t1 = time.time()
                emb = embedder.embeddings([face])[0]
                emb_time = time.time() - t1
                emb_t.append(emb_time)

                t2 = time.time()
                scores = {name: cosine(emb, ref) for name, ref in db.items()}
                best_name = max(scores, key=scores.get)
                best_score = scores[best_name]
                match_t.append(time.time() - t2)

                if best_score > SIM_THRESHOLD:
                    predicted = best_name

        total += 1
        per_person[person]["total"] += 1

        if person.lower() == "unknown":
            unknown_total += 1
            if predicted == "Unknown":
                correct += 1
                unknown_correct += 1
                per_person[person]["correct"] += 1
        else:
            known_total += 1
            if predicted == person:
                correct += 1
                known_correct += 1
                per_person[person]["correct"] += 1

        latencies.append(det_t[-1] + (emb_t[-1] if emb_t else 0) + (match_t[-1] if match_t else 0))

print("\n========== PERFORMANCE REPORT ==========")

overall_acc = (correct / total) * 100 if total else 0
known_acc = (known_correct / known_total) * 100 if known_total else 0
unknown_acc = (unknown_correct / unknown_total) * 100 if unknown_total else 0

print(f"Overall Accuracy              : {overall_acc:.2f}%")
print(f"Known Faces Accuracy          : {known_acc:.2f}%")
print(f"Unknown Detection Accuracy    : {unknown_acc:.2f}%")

print("\n--- Per-Person Performance ---")
for person, stats in per_person.items():
    if stats["total"] > 0:
        acc = (stats["correct"] / stats["total"]) * 100
        print(f"{person.title():<20}: {acc:.2f}%")

print("\n--- Timing Metrics ---")
print("Avg Detection Time (ms) :", round(np.mean(det_t) * 1000, 2))
print("Avg Embedding Time (ms) :", round(np.mean(emb_t) * 1000, 2))
print("Avg Matching Time (ms)  :", round(np.mean(match_t) * 1000, 2))
print("Total Pipeline (ms)     :", round(np.mean(latencies) * 1000, 2))
print("Avg Latency (s)         :", round(np.mean(latencies), 3))
print("Throughput (fps)        :", round(1 / np.mean(latencies), 2))