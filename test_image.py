import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis

print("========================================")
print("       IMAGE FACE TEST STARTED")
print("========================================")

# ------------------------------------------
# 1. LOAD INSIGHTFACE
# ------------------------------------------
print("Loading InsightFace model...")

app = FaceAnalysis(name="buffalo_s")
app.prepare(
    ctx_id=-1,
    det_size=(320, 320)
)

print("InsightFace loaded successfully!")


# ------------------------------------------
# 2. LOAD FACE DATABASE
# ------------------------------------------
KNOWN_FACES_DIR = "known_faces"

known_embeddings = []
known_names = []

print()
print("Loading registered faces...")

for person_name in os.listdir(KNOWN_FACES_DIR):

    person_folder = os.path.join(
        KNOWN_FACES_DIR,
        person_name
    )

    if not os.path.isdir(person_folder):
        continue

    for filename in os.listdir(person_folder):

        file_path = os.path.join(
            person_folder,
            filename
        )

        image = cv2.imread(file_path)

        if image is None:
            continue

        faces = app.get(image)

        if len(faces) == 0:
            print(f"No face found: {filename}")
            continue

        embedding = faces[0].embedding

        embedding = embedding / np.linalg.norm(embedding)

        known_embeddings.append(embedding)
        known_names.append(person_name)

        print(f"Loaded: {person_name} - {filename}")


print()
print("Database loaded successfully!")


# ------------------------------------------
# 3. ASK FOR TEST IMAGE
# ------------------------------------------
image_path = input(
    "\nEnter the path of the image to test: "
)

image = cv2.imread(image_path)

if image is None:

    print()
    print("ERROR: Could not open image.")
    print("Check the image path.")
    exit()


# ------------------------------------------
# 4. DETECT FACE
# ------------------------------------------
faces = app.get(image)

if len(faces) == 0:

    print()
    print("ERROR: No face detected in this image.")
    exit()


print()
print(f"Faces detected: {len(faces)}")


# ------------------------------------------
# 5. COMPARE FACE
# ------------------------------------------
for face in faces:

    current_embedding = face.embedding

    current_embedding = (
        current_embedding /
        np.linalg.norm(current_embedding)
    )

    best_name = "Unknown"
    best_similarity = -1

    for known_embedding, known_name in zip(
        known_embeddings,
        known_names
    ):

        similarity = np.dot(
            current_embedding,
            known_embedding
        )

        if similarity > best_similarity:

            best_similarity = similarity
            best_name = known_name


    # --------------------------------------
    # 6. RESULT
    # --------------------------------------
    if best_similarity < 0.40:

        best_name = "Unknown"

    print()
    print("----------------------------------------")
    print("RESULT")
    print("----------------------------------------")
    print("Name:", best_name)
    print(f"Similarity: {best_similarity:.2f}")
    print("----------------------------------------")


    # --------------------------------------
    # 7. DRAW RESULT
    # --------------------------------------
    box = face.bbox.astype(int)

    x1, y1, x2, y2 = box

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    label = f"{best_name} ({best_similarity:.2f})"

    cv2.putText(
        image,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


# ------------------------------------------
# 8. SHOW RESULT
# ------------------------------------------
cv2.imshow(
    "Image Face Test",
    image
)

print()
print("Press any key on the image window to close.")

cv2.waitKey(0)
cv2.destroyAllWindows()