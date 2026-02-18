import cv2
import time
import pickle
import numpy as np
from retinaface import RetinaFace
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

IMAGE_PATH = "test_images/test1.jpg"
EMBEDDING_PATH = "embeddings/embeddings.pkl"

SIMILARITY_THRESHOLD = 0.6
DISPLAY_WIDTH = 400

DET_SCALE = 0.5
MIN_FACE_SIZE = 40

model = FaceNet()

_ = model.embeddings([np.zeros((160, 160, 3), dtype=np.uint8)])

with open(EMBEDDING_PATH, "rb") as f:
    embeddings_db = pickle.load(f)


def resize_for_display(image, width):
    h, w = image.shape[:2]
    scale = width / w
    return cv2.resize(image, (width, int(h * scale)))


def recognize_face(embedding):
    best_name = "Unknown"
    best_score = -1.0

    for person, ref_emb in embeddings_db.items():
        score = cosine_similarity([embedding], [ref_emb])[0][0]
        if score > best_score:
            best_score = score
            best_name = person

    if best_score < SIMILARITY_THRESHOLD:
        return "Unknown", best_score

    return best_name, best_score


def draw_label(image, text, x1, y1, x2, y2):
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    (tw, th), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )

    tx = max(10, x1)
    ty = max(th + 10, y1 - 10)

    cv2.rectangle(
        image,
        (tx, ty - th - 5),
        (tx + tw, ty + 5),
        (0, 255, 0),
        -1
    )

    cv2.putText(
        image,
        text,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )


def main():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("Image not found.")
        return

    image = resize_for_display(image, DISPLAY_WIDTH)

    total_start = time.time()

    det_start = time.time()
    small = cv2.resize(image, None, fx=DET_SCALE, fy=DET_SCALE)
    detections = RetinaFace.detect_faces(small)
    det_end = time.time()

    if not isinstance(detections, dict):
        print("No face detected.")
        return

    img_h, img_w = image.shape[:2]

    det = list(detections.values())[0]
    x1, y1, x2, y2 = det["facial_area"]

    x1, y1, x2, y2 = [int(v / DET_SCALE) for v in (x1, y1, x2, y2)]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)

    face = image[y1:y2, x1:x2]

    if face.shape[0] < MIN_FACE_SIZE or face.shape[1] < MIN_FACE_SIZE:
        print("Face too small.")
        return

    face = cv2.resize(face, (160, 160))

    emb_start = time.time()
    embedding = model.embeddings([face])[0]
    emb_end = time.time()

    match_start = time.time()
    name, score = recognize_face(embedding)
    match_end = time.time()

    label = f"{name.replace('_',' ').title()} ({score:.2f})"
    draw_label(image, label, x1, y1, x2, y2)

    total_end = time.time()

    det_time = det_end - det_start
    emb_time = emb_end - emb_start
    match_time = match_end - match_start
    pipeline_time = det_time + emb_time + match_time

    print(f"Detection Time        : {det_time:.4f} sec")
    print(f"Embedding Time        : {emb_time:.4f} sec")
    print(f"Matching Time         : {match_time:.6f} sec")
    print(f"Total Pipeline Time   : {pipeline_time:.4f} sec")
    print(f"Throughput (fps)      : {1 / pipeline_time:.2f}")

    cv2.imshow("RetinaFace + FaceNet (Recognize)", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()