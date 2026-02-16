import cv2
import time
import pickle
import numpy as np
from retinaface import RetinaFace
from keras_facenet import FaceNet

EMB_PATH = "embeddings/embeddings.pkl"
SIM_THRESHOLD = 0.6
DETECT_EVERY_N = 15
DET_SCALE = 0.5
DISPLAY_W = 480

embedder = FaceNet()
with open(EMB_PATH, "rb") as f:
    db = pickle.load(f)

def cosine(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def recognize(emb):
    best, score = "Unknown", -1
    for name, ref in db.items():
        s = cosine(emb, ref)
        if s > score:
            best, score = name, s
    return (best, score) if score > SIM_THRESHOLD else ("Unknown", score)

def resize_keep(img, w):
    h, ow = img.shape[:2]
    return cv2.resize(img, (w, int(h*w/ow)))

cap = cv2.VideoCapture(0)
frame_id = 0
cached = []
fps_list = []

RetinaFace.detect_faces(np.zeros((240,320,3),dtype=np.uint8))

prev = time.perf_counter()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = resize_keep(frame, DISPLAY_W)

    if frame_id % DETECT_EVERY_N == 0:
        cached.clear()
        small = cv2.resize(frame, None, fx=DET_SCALE, fy=DET_SCALE)
        sx, sy = frame.shape[1]/small.shape[1], frame.shape[0]/small.shape[0]

        dets = RetinaFace.detect_faces(small)
        if isinstance(dets, dict):
            for d in dets.values():
                x1,y1,x2,y2 = d["facial_area"]
                x1,x2 = int(x1*sx), int(x2*sx)
                y1,y2 = int(y1*sy), int(y2*sy)

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                face = cv2.resize(face,(160,160))
                emb = embedder.embeddings([face])[0]
                name, score = recognize(emb)
                cached.append((x1,y1,x2,y2,name,score))

    for x1,y1,x2,y2,n,s in cached:
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,f"{n} {s:.2f}",(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    now = time.perf_counter()
    fps = 1/(now-prev)
    prev = now
    fps_list.append(fps)

    cv2.putText(frame,f"FPS: {fps:.2f}",(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

    cv2.imshow("RetinaFace + FaceNet", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()

arr = np.array(fps_list)
print("\n=== FPS REPORT ===")
print("Avg FPS:", round(arr.mean(),2))
print("Max FPS:", round(arr.max(),2))
print("Min FPS:", round(arr.min(),2))
print("Throughput:", round(len(arr)/(1/arr).sum(),2),"fps")