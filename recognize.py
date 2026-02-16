import cv2
import time
import pickle
from retinaface import RetinaFace
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity


IMAGE_PATH = "test_images/test1.jpg"
SIMILARITY_THRESHOLD = 0.6
DISPLAY_WIDTH = 400
EMBEDDING_PATH = "embeddings/embeddings.pkl"


model = FaceNet()

with open(EMBEDDING_PATH, "rb") as f:
    embeddings_db = pickle.load(f)


def resize_for_display(image, width):
    """Resize image while keeping aspect ratio."""
    h, w = image.shape[:2]
    scale = width / w
    new_height = int(h * scale)
    return cv2.resize(image, (width, new_height))


def recognize_face(embedding):
    """Find best match using cosine similarity."""
    best_name = "Unknown"
    best_score = -1

    for person, stored_embeddings in embeddings_db.items():
        for stored_emb in stored_embeddings:
            score = cosine_similarity([embedding], [stored_emb])[0][0]

            if score > best_score:
                best_score = score
                best_name = person

    if best_score < SIMILARITY_THRESHOLD:
        return "Unknown", best_score

    return best_name, best_score


def draw_label(image, text, x1, y1, x2, y2):
    """Draw bounding box and label safely."""
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label = text
    (text_w, text_h), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )

    text_x = max(10, x1)
    text_y = max(text_h + 10, y1 - 10)

    cv2.rectangle(
        image,
        (text_x, text_y - text_h - 5),
        (text_x + text_w, text_y + 5),
        (0, 255, 0),
        -1
    )

    cv2.putText(
        image,
        label,
        (text_x, text_y),
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

    detect_start = time.time()
    detections = RetinaFace.detect_faces(image)
    detect_end = time.time()

    if not isinstance(detections, dict):
        print("No face detected.")
        return

    img_h, img_w = image.shape[:2]

    for key in detections:
        x1, y1, x2, y2 = detections[key]["facial_area"]

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        face = image[y1:y2, x1:x2]

        if face.shape[0] < 40 or face.shape[1] < 40:
            continue

        face = cv2.resize(face, (160, 160))

        embed_start = time.time()
        embedding = model.embeddings([face])[0]
        embed_end = time.time()

        match_start = time.time()
        name, score = recognize_face(embedding)
        match_end = time.time()

        display_name = name.replace("_", " ").title()
        label_text = f"{display_name} ({score:.2f})"

        draw_label(image, label_text, x1, y1, x2, y2)

    total_end = time.time()

    print("\n--- Performance Metrics ---")
    print("Detection Time :", round(detect_end - detect_start, 4), "sec")
    print("Embedding Time :", round(embed_end - embed_start, 4), "sec")
    print("Matching Time  :", round(match_end - match_start, 6), "sec")
    print("Total Time     :", round(total_end - total_start, 4), "sec")

    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()