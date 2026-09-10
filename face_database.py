import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Folder containing known people
KNOWN_FACES_DIR = "known_faces"

print("Loading face recognition model...")

# Load InsightFace
app = FaceAnalysis(name="buffalo_s")

# Use CPU
app.prepare(ctx_id=-1, det_size=(320, 320))

# Lists for face embeddings and names
known_embeddings = []
known_names = []

print("Loading known faces...")

# Check known_faces folder
if not os.path.exists(KNOWN_FACES_DIR):
    print("ERROR: known_faces folder not found.")
    exit()

# Go through Person1, Person2, etc.
for person_name in os.listdir(KNOWN_FACES_DIR):

    person_folder = os.path.join(
        KNOWN_FACES_DIR,
        person_name
    )

    # Skip files
    if not os.path.isdir(person_folder):
        continue

    print(f"Checking folder: {person_name}")

    # Read images
    for filename in os.listdir(person_folder):

        file_path = os.path.join(
            person_folder,
            filename
        )

        # Read image
        image = cv2.imread(file_path)

        # Skip invalid files
        if image is None:
            print(f"  Could not read: {filename}")
            continue

        # Detect faces
        faces = app.get(image)

        # No face found
        if len(faces) == 0:
            print(f"  No face found: {filename}")
            continue

        # Take first detected face
        embedding = faces[0].embedding

        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)

        # Store embedding and person's name
        known_embeddings.append(embedding)
        known_names.append(person_name)

        print(f"  Loaded: {filename}")


# Display result
print()
print("=" * 40)
print(f"Total known faces loaded: {len(known_embeddings)}")
print("=" * 40)

# No photos available
if len(known_embeddings) == 0:

    print()
    print("No photos found yet.")
    print("This is OK!")
    print()
    print("Later, add authorized photos here:")
    print("known_faces\\Person1")
    print("known_faces\\Person2")

else:

    print()
    print("Face database loaded successfully!")
    print("People:", set(known_names))