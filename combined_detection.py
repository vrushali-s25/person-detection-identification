from ultralytics import YOLO
import cv2
import os
import numpy as np
import csv

from insightface.app import FaceAnalysis

print("========================================")
print("   PERSON DETECTION + IDENTIFICATION")
print("========================================")

# ===============================
# 1. LOAD YOLO
# ===============================

print("Loading YOLO...")

yolo = YOLO("yolo11n.pt")

print("YOLO loaded successfully!")

# ===============================
# 2. LOAD INSIGHTFACE
# ===============================

print("Loading InsightFace...")

face_app = FaceAnalysis(name="buffalo_s")

face_app.prepare(
    ctx_id=-1,
    det_size=(320, 320)
)

print("InsightFace loaded successfully!")

# ===============================
# 3. LOAD KNOWN FACES
# ===============================

KNOWN_FACES_DIR = "known_faces"

known_embeddings = []
known_names = []

print("\nLoading registered faces...")

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

        faces = face_app.get(image)

        if len(faces) == 0:
            continue

        embedding = faces[0].embedding

        embedding = (
            embedding /
            np.linalg.norm(embedding)
        )

        known_embeddings.append(embedding)
        known_names.append(person_name)

        print(
            "Loaded:",
            person_name,
            filename
        )

print("\nDatabase loaded successfully!")

print("Registered people:")

for name in sorted(set(known_names)):
    print(" -", name)

# ===============================
# 4. CREATE CSV FILE
# ===============================

csv_file = open(
    "detection_results.csv",
    "w",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "Frame",
    "Time",
    "Name",
    "Similarity"
])

print("\nCSV file created:")
print("detection_results.csv")

# ===============================
# 5. OPEN VIDEO
# ===============================

video = cv2.VideoCapture("video.mp4")

if not video.isOpened():

    print("ERROR: Could not open video.mp4")

    csv_file.close()

    exit()

fps = video.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25

print("\nVideo started!")
print("Press Q to close.")

# ===============================
# 6. VARIABLES
# ===============================

frame_count = 0

last_person_boxes = []
last_face_results = []

# ===============================
# 7. VIDEO LOOP
# ===============================

while True:

    ret, frame = video.read()

    if not ret:

        print("Video finished.")

        break

    frame_count += 1

    # Resize video
    frame = cv2.resize(
        frame,
        (640, 360)
    )

    # ===========================
    # YOLO
    # ===========================

    if frame_count % 3 == 0:

        results = yolo(
            frame,
            classes=[0],
            conf=0.35,
            verbose=False
        )

        last_person_boxes = (
            results[0]
            .boxes
            .xyxy
            .cpu()
            .numpy()
        )

    # ===========================
    # FACE DETECTION
    # ===========================

    if frame_count % 10 == 0:

        last_face_results = []

        faces = face_app.get(frame)

        for face in faces:

            box = face.bbox.astype(int)

            fx1, fy1, fx2, fy2 = box

            current_embedding = face.embedding

            current_embedding = (
                current_embedding /
                np.linalg.norm(current_embedding)
            )

            best_name = "Unknown"
            best_similarity = -1

            # Compare with database
            for (
                known_embedding,
                known_name
            ) in zip(
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

            center_x = (
                fx1 + fx2
            ) // 2

            center_y = (
                fy1 + fy2
            ) // 2

            last_face_results.append(
                (
                    fx1,
                    fy1,
                    fx2,
                    fy2,
                    center_x,
                    center_y,
                    best_name,
                    best_similarity
                )
            )

    # ===========================
    # COUNTERS
    # ===========================

    people_count = len(
        last_person_boxes
    )

    known_count = 0
    unknown_count = 0

    # ===========================
    # DRAW PEOPLE
    # ===========================

    for person_box in last_person_boxes:

        x1, y1, x2, y2 = map(
            int,
            person_box
        )

        person_name = "Person"
        person_similarity = 0.0

        for face_data in last_face_results:

            (
                fx1,
                fy1,
                fx2,
                fy2,
                center_x,
                center_y,
                face_name,
                similarity
            ) = face_data

            # Check if face belongs
            # to this person
            if (
                x1 <= center_x <= x2
                and
                y1 <= center_y <= y2
            ):

                person_name = face_name
                person_similarity = similarity

                if face_name == "Unknown":

                    unknown_count += 1

                else:

                    known_count += 1

                # Face box
                cv2.rectangle(
                    frame,
                    (fx1, fy1),
                    (fx2, fy2),
                    (255, 0, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{face_name} {similarity:.2f}",
                    (
                        fx1,
                        max(20, fy1 - 8)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2
                )

                # ===================
                # SAVE CSV RESULT
                # ===================

                video_time = frame_count / fps

                csv_writer.writerow([
                    frame_count,
                    f"{video_time:.2f}",
                    face_name,
                    f"{similarity:.2f}"
                ])

                csv_file.flush()

                break

        # Person box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            person_name,
            (
                x1,
                max(20, y1 - 10)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # ===========================
    # DISPLAY COUNTERS
    # ===========================

    cv2.putText(
        frame,
        f"People: {people_count}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Known: {known_count}",
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Unknown: {unknown_count}",
        (10, 79),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    # ===========================
    # SHOW VIDEO
    # ===========================

    cv2.imshow(
        "Person Detection + Identification",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break

# ===============================
# CLOSE
# ===============================

video.release()

csv_file.close()

cv2.destroyAllWindows()

print("\n========================================")
print("Program stopped.")
print("========================================")
print("Detection results saved to:")
print("detection_results.csv")