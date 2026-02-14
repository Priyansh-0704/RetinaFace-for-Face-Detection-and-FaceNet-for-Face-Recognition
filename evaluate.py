import os
import cv2
import pickle
import numpy as np
from retinaface import RetinaFace
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

SIMILARITY_THRESHOLD = 0.6
TEST_PATH = "test_images"

embedder = FaceNet()

with open("embeddings/embeddings.pkl", "rb") as f:
    embeddings_dict = pickle.load(f)


def recognize_face(face_embedding):
    best_match = None
    highest_similarity = -1

    for person_name, embeddings in embeddings_dict.items():
        for emb in embeddings:
            similarity = cosine_similarity(
                [face_embedding], [emb]
            )[0][0]

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = person_name

    if highest_similarity > SIMILARITY_THRESHOLD:
        return best_match
    else:
        return "Unknown"


total = 0
correct = 0

known_total = 0
known_correct = 0

unknown_total = 0
unknown_correct = 0

per_person_results = {}

for person in os.listdir(TEST_PATH):

    person_folder = os.path.join(TEST_PATH, person)

    if not os.path.isdir(person_folder):
        continue

    person_correct = 0
    person_total = 0

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)
        img = cv2.imread(image_path)

        if img is None:
            continue

        detections = RetinaFace.detect_faces(img)

        if not isinstance(detections, dict):
            continue

        for key in detections.keys():

            x1, y1, x2, y2 = detections[key]['facial_area']
            face = img[y1:y2, x1:x2]
            face = cv2.resize(face, (160, 160))

            embedding = embedder.embeddings([face])[0]
            predicted = recognize_face(embedding)

            total += 1
            person_total += 1

            if person.lower() == "unknown":
                unknown_total += 1

                if predicted == "Unknown":
                    correct += 1
                    unknown_correct += 1
                    person_correct += 1

            else:
                known_total += 1

                if predicted == person:
                    correct += 1
                    known_correct += 1
                    person_correct += 1

    if person_total > 0:
        per_person_results[person] = round(
            (person_correct / person_total) * 100, 2
        )

overall_accuracy = (correct / total) * 100 if total > 0 else 0
known_accuracy = (known_correct / known_total) * 100 if known_total > 0 else 0
unknown_accuracy = (unknown_correct / unknown_total) * 100 if unknown_total > 0 else 0

print("\n--- Evaluation Results ---")
print("Total Samples Evaluated:", total)
print("Overall Accuracy:", round(overall_accuracy, 2), "%")

print("\nKnown Faces Accuracy:", round(known_accuracy, 2), "%")
print("Unknown Detection Accuracy:", round(unknown_accuracy, 2), "%")

print("\nPer Person Accuracy:")
for person, acc in per_person_results.items():
    print(person, ":", acc, "%")