import os
import cv2
import pickle
from retinaface import RetinaFace
from keras_facenet import FaceNet

DATASET_DIR = "dataset"
OUTPUT_FILE = "embeddings/embeddings.pkl"

model = FaceNet()


def get_face_embedding(image):
    detections = RetinaFace.detect_faces(image)

    if not isinstance(detections, dict):
        return None

    # take first detected face
    for key in detections:
        x1, y1, x2, y2 = detections[key]["facial_area"]

        face = image[y1:y2, x1:x2]

        if face.size == 0:
            return None

        face = cv2.resize(face, (160, 160))
        embedding = model.embeddings([face])[0]
        return embedding

    return None


def generate_embeddings():
    embeddings_data = {}

    for person in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person)

        if not os.path.isdir(person_path):
            continue

        embeddings_data[person] = []

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            emb = get_face_embedding(img)

            if emb is not None:
                embeddings_data[person].append(emb)

        print(f"{person}: {len(embeddings_data[person])} embeddings created")

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(embeddings_data, f)

    print("Embeddings saved successfully.")


if __name__ == "__main__":
    generate_embeddings()