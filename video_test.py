import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis

print("========================================")
print("       FAST VIDEO FACE TEST")
print("========================================")

# Load InsightFace
print("Loading InsightFace...")
app = FaceAnalysis(name="buffalo_s")
app.prepare(ctx_id=-1, det_size=(320, 320))
print("InsightFace loaded!")

# -------------------------------
# Load known faces
# -------------------------------
KNOWN_FACES_DIR = "known_faces"

known_embeddings = []
known_names = []

print("\nLoading registered faces...")

for person_name in os.listdir(KNOWN_FACES_DIR):

    person_folder = os.path.join(KNOWN_FACES_DIR, person_name)

    if not os.path.isdir(person_folder):
        continue

    for filename in os.listdir(person_folder):

        file_path = os.path.join(person_folder, filename)

        image = cv2.imread(file_path)

        if image is None:
            continue

        faces = app.get(image)

        if len(faces) == 0:
            print("No face:", filename)
            continue

        embedding = faces[0].embedding
        embedding = embedding / np.linalg.norm(embedding)

        known_embeddings.append(embedding)
        known_names.append(person_name)

        print("Loaded:", person_name, filename)

print("\nDatabase loaded successfully!")

# -------------------------------
# Open video
# -------------------------------

video_path = input("\nEnter video path: ")

video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("ERROR: Could not open video.")
    exit()

print("\nVideo started!")
print("Press Q to close.")

frame_count = 0
last_faces = []

while True:

    ret, frame = video.read()

    if not ret:
        print("Video finished.")
        break

    frame_count += 1

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (320, 180))

    # Process only every 10th frame
    if frame_count % 10 == 0:

        last_faces = app.get(small_frame)

        for face in last_faces:

            box = face.bbox.astype(int)

            # Scale coordinates back to original frame
            x1 = box[0] * 2
            y1 = box[1] * 2
            x2 = box[2] * 2
            y2 = box[3] * 2

            current_embedding = face.embedding
            current_embedding = current_embedding / np.linalg.norm(current_embedding)

            best_name = "Unknown"
            best_similarity = -1

            # Compare with registered people
            for known_embedding, known_name in zip(
                known_embeddings, known_names
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

            label = f"{best_name} ({best_similarity:.2f})"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Show video
    cv2.imshow("Fast Video Face Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

print("\nVideo test stopped.")