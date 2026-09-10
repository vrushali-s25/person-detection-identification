import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import csv
from insightface.app import FaceAnalysis
from ultralytics import YOLO


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Person Detection & Identification",
    page_icon="👤",
    layout="wide"
)

st.title("👤 Person Detection & Identification System")

st.write(
    "Computer Vision application using YOLO and InsightFace "
    "for person detection and authorized face identification."
)


# =========================================================
# LOAD AI MODELS
# =========================================================

@st.cache_resource
def load_models():

    yolo = YOLO("yolo11n.pt")

    face_app = FaceAnalysis(
        name="buffalo_s"
    )

    face_app.prepare(
        ctx_id=-1,
        det_size=(320, 320)
    )

    return yolo, face_app


# =========================================================
# LOAD MODELS
# =========================================================

with st.spinner("Loading AI models..."):

    try:

        yolo, face_app = load_models()

        st.success("AI models loaded successfully!")

    except Exception as e:

        st.error("Could not load AI models.")
        st.exception(e)
        st.stop()


# =========================================================
# SESSION DATABASE
# =========================================================

if "known_embeddings" not in st.session_state:

    st.session_state.known_embeddings = []


if "known_names" not in st.session_state:

    st.session_state.known_names = []


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Face Registration")

st.sidebar.write(
    "Register authorized reference photos "
    "for face identification."
)


# =========================================================
# REGISTER PERSON
# =========================================================

person_name = st.sidebar.text_input(
    "Person name",
    placeholder="Example: Ananya"
)


reference_files = st.sidebar.file_uploader(
    "Upload reference photos",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="reference_upload"
)


if st.sidebar.button("Register Person"):

    if not person_name:

        st.sidebar.error(
            "Please enter a person name."
        )

    elif not reference_files:

        st.sidebar.error(
            "Please upload at least one photo."
        )

    else:

        added = 0

        for uploaded_file in reference_files:

            file_bytes = np.asarray(
                bytearray(
                    uploaded_file.read()
                ),
                dtype=np.uint8
            )

            image = cv2.imdecode(
                file_bytes,
                cv2.IMREAD_COLOR
            )

            if image is None:

                st.sidebar.warning(
                    f"Could not read {uploaded_file.name}"
                )

                continue

            faces = face_app.get(image)

            if len(faces) == 0:

                st.sidebar.warning(
                    f"No face found in "
                    f"{uploaded_file.name}"
                )

                continue

            # Use the largest face
            face = max(
                faces,
                key=lambda x:
                (x.bbox[2] - x.bbox[0]) *
                (x.bbox[3] - x.bbox[1])
            )

            embedding = face.embedding

            embedding = (
                embedding /
                np.linalg.norm(embedding)
            )

            st.session_state.known_embeddings.append(
                embedding
            )

            st.session_state.known_names.append(
                person_name
            )

            added += 1

        if added > 0:

            st.sidebar.success(
                f"{added} photo(s) registered "
                f"for {person_name}."
            )

        else:

            st.sidebar.error(
                "No usable faces were found."
            )


# =========================================================
# CLEAR DATABASE
# =========================================================

if st.sidebar.button("Clear Registered People"):

    st.session_state.known_embeddings = []

    st.session_state.known_names = []

    st.sidebar.success(
        "Registered people cleared."
    )

    st.rerun()


# =========================================================
# SHOW REGISTERED PEOPLE
# =========================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "Registered People"
)


registered_people = sorted(
    set(
        st.session_state.known_names
    )
)


if registered_people:

    for name in registered_people:

        st.sidebar.write(
            "✅",
            name
        )

else:

    st.sidebar.info(
        "No people registered yet."
    )


# =========================================================
# RECOGNITION FUNCTION
# =========================================================

def recognize_face(face):

    current_embedding = face.embedding

    current_embedding = (
        current_embedding /
        np.linalg.norm(current_embedding)
    )

    best_name = "Unknown"

    best_similarity = -1

    for (
        known_embedding,
        known_name
    ) in zip(
        st.session_state.known_embeddings,
        st.session_state.known_names
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

    return best_name, best_similarity


# =========================================================
# IMAGE SECTION
# =========================================================

st.header("📷 Image Detection")

image_file = st.file_uploader(
    "Upload an image to analyze",
    type=["jpg", "jpeg", "png"],
    key="image_test"
)


if image_file is not None:

    image_bytes = np.asarray(
        bytearray(
            image_file.read()
        ),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:

        st.error(
            "Could not read the uploaded image."
        )

    else:

        # ---------------------------------------------
        # YOLO
        # ---------------------------------------------

        yolo_results = yolo(
            image,
            classes=[0],
            conf=0.35,
            verbose=False
        )

        person_boxes = (
            yolo_results[0]
            .boxes
            .xyxy
            .cpu()
            .numpy()
        )

        # ---------------------------------------------
        # INSIGHTFACE
        # ---------------------------------------------

        faces = face_app.get(image)

        output = image.copy()

        known_count = 0

        unknown_count = 0

        # ---------------------------------------------
        # DRAW PERSON BOXES
        # ---------------------------------------------

        for person_box in person_boxes:

            x1, y1, x2, y2 = map(
                int,
                person_box
            )

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                output,
                "Person",
                (
                    x1,
                    max(20, y1 - 10)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # ---------------------------------------------
        # FACE IDENTIFICATION
        # ---------------------------------------------

        for face in faces:

            box = face.bbox.astype(int)

            fx1, fy1, fx2, fy2 = box

            name, similarity = recognize_face(
                face
            )

            if name == "Unknown":

                unknown_count += 1

            else:

                known_count += 1

            # Face box
            cv2.rectangle(
                output,
                (fx1, fy1),
                (fx2, fy2),
                (255, 0, 0),
                2
            )

            label = (
                f"{name} "
                f"({similarity:.2f})"
            )

            cv2.putText(
                output,
                label,
                (
                    fx1,
                    max(25, fy1 - 10)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        # ---------------------------------------------
        # RESULTS
        # ---------------------------------------------

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "👥 People detected",
            len(person_boxes)
        )

        col2.metric(
            "✅ Known",
            known_count
        )

        col3.metric(
            "❓ Unknown",
            unknown_count
        )

        st.subheader(
            "Detection Result"
        )

        output_rgb = cv2.cvtColor(
            output,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            output_rgb,
            use_container_width=True
        )

        if len(faces) == 0:

            st.warning(
                "No face detected."
            )


# =========================================================
# VIDEO SECTION
# =========================================================

st.markdown("---")

st.header("🎥 Video Detection")

video_file = st.file_uploader(
    "Upload a video to analyze",
    type=["mp4", "avi", "mov"],
    key="video_test"
)


if video_file is not None:

    st.video(video_file)

    if len(
        st.session_state.known_embeddings
    ) == 0:

        st.warning(
            "Please register at least one "
            "authorized person first."
        )

    else:

        process_video = st.button(
            "▶️ Process Video"
        )

        if process_video:

            # =========================================
            # SAVE INPUT VIDEO
            # =========================================

            input_temp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            input_temp.write(
                video_file.read()
            )

            input_temp.close()

            input_path = input_temp.name

            # =========================================
            # OPEN VIDEO
            # =========================================

            video = cv2.VideoCapture(
                input_path
            )

            if not video.isOpened():

                st.error(
                    "Could not open uploaded video."
                )

                os.remove(input_path)

                st.stop()

            fps = video.get(
                cv2.CAP_PROP_FPS
            )

            if fps <= 0:

                fps = 25

            total_frames = int(
                video.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            # =========================================
            # OUTPUT VIDEO
            # =========================================

            width = 640

            height = 360

            output_temp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            output_temp.close()

            output_path = output_temp.name

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            # =========================================
            # PROGRESS
            # =========================================

            progress = st.progress(0)

            status = st.empty()

            frame_count = 0

            last_person_boxes = []

            last_faces = []

            # =========================================
            # CSV RESULTS
            # =========================================

            csv_rows = []

            # =========================================
            # PROCESS VIDEO
            # =========================================

            while True:

                ret, frame = video.read()

                if not ret:

                    break

                frame_count += 1

                # Resize for faster CPU processing
                frame = cv2.resize(
                    frame,
                    (width, height)
                )

                # =====================================
                # YOLO EVERY 3 FRAMES
                # =====================================

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

                # =====================================
                # INSIGHTFACE EVERY 10 FRAMES
                # =====================================

                if frame_count % 10 == 0:

                    last_faces = []

                    faces = face_app.get(
                        frame
                    )

                    for face in faces:

                        box = face.bbox.astype(int)

                        fx1, fy1, fx2, fy2 = box

                        name, similarity = (
                            recognize_face(face)
                        )

                        center_x = (
                            fx1 + fx2
                        ) // 2

                        center_y = (
                            fy1 + fy2
                        ) // 2

                        last_faces.append(
                            (
                                fx1,
                                fy1,
                                fx2,
                                fy2,
                                center_x,
                                center_y,
                                name,
                                similarity
                            )
                        )

                        # Save CSV result
                        video_time = (
                            frame_count / fps
                        )

                        csv_rows.append(
                            [
                                frame_count,
                                f"{video_time:.2f}",
                                name,
                                f"{similarity:.2f}"
                            ]
                        )

                # =====================================
                # DRAW PEOPLE
                # =====================================

                for person_box in (
                    last_person_boxes
                ):

                    x1, y1, x2, y2 = map(
                        int,
                        person_box
                    )

                    person_name = "Person"

                    # Find matching face
                    for face_data in last_faces:

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

                        if (
                            x1 <= center_x <= x2
                            and
                            y1 <= center_y <= y2
                        ):

                            person_name = (
                                face_name
                            )

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
                                (
                                    f"{face_name} "
                                    f"{similarity:.2f}"
                                ),
                                (
                                    fx1,
                                    max(
                                        20,
                                        fy1 - 8
                                    )
                                ),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 0, 0),
                                2
                            )

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
                            max(
                                20,
                                y1 - 10
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                # =====================================
                # WRITE OUTPUT FRAME
                # =====================================

                writer.write(frame)

                # =====================================
                # UPDATE PROGRESS
                # =====================================

                if total_frames > 0:

                    percentage = (
                        frame_count /
                        total_frames
                    )

                    progress.progress(
                        min(
                            percentage,
                            1.0
                        )
                    )

                    status.write(
                        f"Processing: "
                        f"{frame_count} / "
                        f"{total_frames} frames"
                    )

            # =========================================
            # CLOSE VIDEO
            # =========================================

            video.release()

            writer.release()

            progress.progress(1.0)

            status.success(
                "✅ Video processing completed!"
            )

            # =========================================
            # CREATE CSV
            # =========================================

            csv_path = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".csv"
            ).name

            with open(
                csv_path,
                "w",
                newline="",
                encoding="utf-8"
            ) as csv_file:

                csv_writer = csv.writer(
                    csv_file
                )

                csv_writer.writerow(
                    [
                        "Frame",
                        "Time",
                        "Name",
                        "Similarity"
                    ]
                )

                csv_writer.writerows(
                    csv_rows
                )

            # =========================================
            # SHOW PROCESSED VIDEO
            # =========================================

            st.subheader(
                "🎬 Processed Video"
            )

            with open(
                output_path,
                "rb"
            ) as video_output:

                output_bytes = (
                    video_output.read()
                )

            st.video(
                output_bytes
            )

            # =========================================
            # DOWNLOAD VIDEO
            # =========================================

            st.download_button(
                label="⬇️ Download Processed Video",
                data=output_bytes,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )

            # =========================================
            # DOWNLOAD CSV
            # =========================================

            with open(
                csv_path,
                "rb"
            ) as csv_output:

                csv_bytes = (
                    csv_output.read()
                )

            st.download_button(
                label="📊 Download Detection CSV",
                data=csv_bytes,
                file_name="detection_results.csv",
                mime="text/csv"
            )

            # =========================================
            # CLEAN TEMP FILES
            # =========================================

            try:

                os.remove(input_path)

                os.remove(output_path)

                os.remove(csv_path)

            except:

                pass