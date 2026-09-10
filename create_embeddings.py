import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

print("========================================")
print("   FACE EMBEDDING CREATION STARTED")
print("========================================")

# ==========================================
# 1. LOAD INSIGHTFACE
# ==========================================

print("\nLoading InsightFace model...")

face_app = FaceAnalysis(name="buffalo_s")

face_app.prepare(
    ctx_id=-1,
    det_size=(320, 320)
)

print("InsightFace loaded successfully!")


# ==========================================
# 2. FOLDER SETTINGS
# ==========================================

known_faces_folder = "known_faces"

output_file = "face_database.pkl"


# ==========================================
# 3. CHECK KNOWN FACES FOLDER
# ==========================================

if not os.path.exists(known_faces_folder):

    print("\nERROR: known_faces folder not found!")

    print("Please create:")
    print("known_faces/Vrushali/")

    exit()


# ==========================================
# 4. CREATE DATABASE
# ==========================================

face_database = {}


# ==========================================
# 5. READ PERSON FOLDERS
# ==========================================

for person_name in os.listdir(known_faces_folder):

    person_folder = os.path.join(
        known_faces_folder,
        person_name
    )

    if not os.path.isdir(person_folder):
        continue

    print("\n----------------------------------------")
    print("Person:", person_name)
    print("----------------------------------------")

    embeddings = []


    # ======================================
    # READ PHOTOS
    # ======================================

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(
            person_folder,
            image_name
        )

        # Only process image files
        if not image_name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        print("Processing:", image_name)

        # Read image
        image = cv2.imread(image_path)

        if image is None:

            print("Could not read:", image_name)

            continue


        # ==================================
        # DETECT FACE
        # ==================================

        faces = face_app.get(image)


        if len(faces) == 0:

            print("NO FACE FOUND:", image_name)

            continue


        # ==================================
        # SELECT LARGEST FACE
        # ==================================

        face = max(
            faces,
            key=lambda x:
            (x.bbox[2] - x.bbox[0]) *
            (x.bbox[3] - x.bbox[1])
        )


        # ==================================
        # GET FACE EMBEDDING
        # ==================================

        embedding = face.embedding

        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)

        embeddings.append(embedding)

        print("Face embedding created successfully.")


    # ======================================
    # SAVE PERSON EMBEDDINGS
    # ======================================

    if len(embeddings) > 0:

        face_database[person_name] = embeddings

        print(
            f"Saved {len(embeddings)} embeddings for {person_name}"
        )

    else:

        print(
            f"WARNING: No valid face found for {person_name}"
        )


# ==========================================
# 6. CHECK DATABASE
# ==========================================

if len(face_database) == 0:

    print("\nERROR: No faces were successfully processed.")

    print("Check your photos.")

    exit()


# ==========================================
# 7. SAVE DATABASE
# ==========================================

with open(output_file, "wb") as file:

    pickle.dump(
        face_database,
        file
    )


# ==========================================
# 8. FINAL MESSAGE
# ==========================================

print("\n========================================")
print("FACE DATABASE CREATED SUCCESSFULLY!")
print("========================================")

print("\nDatabase file:")

print(output_file)

print("\nPeople registered:")

for person_name in face_database:

    count = len(face_database[person_name])

    print(
        f"  {person_name}: {count} embeddings"
    )

print("\nYou can now use this database")
print("for face identification.")