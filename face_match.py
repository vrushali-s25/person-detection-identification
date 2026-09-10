import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis

print("========================================")
print("   FACE IDENTIFICATION STARTED")
print("========================================")

# Load InsightFace
print("Loading InsightFace model...")

app = FaceAnalysis(name="buffalo_s")
app.prepare(ctx_id=-1, det_size=(320, 320))

print("InsightFace loaded successfully!")

# Load known faces
KNOWN_FACES_DIR = "known_faces"

known_embeddings = []
known_names = []

print()
print("Loading registered people...")

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
            print(f"WARNING: No face found: {filename}")
            continue

        embedding = faces[0].embedding
        embedding = embedding / np.linalg.norm(embedding)

        known_embeddings.append(embedding)
        known_names.append(person_name)

        print(f"Loaded: {person_name} - {filename}")


# Check database
if len(known_embeddings) == 0:

    print()
    print("ERROR: No face embeddings found.")
    print("Please check your known_faces folder.")
    exit()

print()
print("Database loaded successfully!")

print()
print("Registered people:")

for name in sorted(set(known_names)):
    print(" -", name)


# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Could not open webcam.")
    exit()

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print()
print("Camera started!")
print("Press Q to close.")


# Face identification
frame_count = 0
last_faces = []

while True:

    ret, frame = camera.read()

    if not ret:

        print("ERROR: Could not read camera.")
        break

    frame_count += 1

    # Detect face every 5th frame
    if frame_count % 5 == 0:

        small_frame = cv2.resize(
            frame,
            (320, 240)
        )

        last_faces = app.get(small_frame)


    # Process detected faces
    for face in last_faces:

        box = face.bbox.astype(int)

        x1, y1, x2, y2 = box

        # Convert coordinates back
        x1 = int(x1 * 2)
        y1 = int(y1 * 2)
        x2 = int(x2 * 2)
        y2 = int(y2 * 2)


        # Current face embedding
        current_embedding = face.embedding

        current_embedding = (
            current_embedding /
            np.linalg.norm(current_embedding)
        )


        # Find best match
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


        # Recognition threshold
        if best_similarity < 0.40:

            best_name = "Unknown"


        # Draw box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # Display name
        label = f"{best_name} ({best_similarity:.2f})"

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # Show camera
    cv2.imshow(
        "Face Identification",
        frame
    )


    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


camera.release()
cv2.destroyAllWindows()

print()
print("Face identification stopped.")