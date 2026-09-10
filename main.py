import cv2
from ultralytics import YOLO
from insightface.app import FaceAnalysis

print("Starting Person Detection System...")

# ==========================================
# 1. LOAD YOLO MODEL
# ==========================================
print("Loading YOLO model...")

yolo_model = YOLO("yolo11n.pt")


# ==========================================
# 2. LOAD INSIGHTFACE MODEL
# ==========================================
print("Loading Face Detection model...")

face_app = FaceAnalysis(name="buffalo_s")

face_app.prepare(
    ctx_id=-1,
    det_size=(320, 320)
)


# ==========================================
# 3. OPEN WEBCAM
# ==========================================
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam")
    exit()

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("System started successfully!")
print("Press Q or ESC to close.")


# ==========================================
# 4. VARIABLES
# ==========================================
frame_count = 0
last_faces = []


# ==========================================
# 5. MAIN LOOP
# ==========================================
try:

    while True:

        # ==================================
        # READ CAMERA
        # ==================================
        ret, frame = camera.read()

        if not ret:
            print("ERROR: Could not read camera")
            break

        frame_count += 1


        # ==================================
        # PERSON DETECTION USING YOLO
        # ==================================
        results = yolo_model(
            frame,
            classes=[0],
            conf=0.5,
            verbose=False
        )

        output = results[0].plot()


        # ==================================
        # FACE DETECTION
        # ==================================
        if frame_count % 5 == 0:

            small_frame = cv2.resize(
                frame,
                (320, 240)
            )

            last_faces = face_app.get(small_frame)


        # ==================================
        # DRAW FACE BOXES
        # ==================================
        for face in last_faces:

            box = face.bbox.astype(int)

            x1, y1, x2, y2 = box

            # Resize coordinates back to 640x480
            x1 = int(x1 * 2)
            y1 = int(y1 * 2)
            x2 = int(x2 * 2)
            y2 = int(y2 * 2)

            # Draw face rectangle
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Face label
            cv2.putText(
                output,
                "Face Detected",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


        # ==================================
        # DISPLAY CAMERA
        # ==================================
        cv2.imshow(
            "Person Detection and Identification",
            output
        )


        # ==================================
        # KEYBOARD CONTROL
        # ==================================
        key = cv2.waitKey(1)

        # Q or q
        if key == ord("q") or key == ord("Q"):
            print("Q pressed. Closing system...")
            break

        # ESC key
        if key == 27:
            print("ESC pressed. Closing system...")
            break


# ==========================================
# HANDLE CTRL+C
# ==========================================
except KeyboardInterrupt:

    print("\nCtrl+C detected.")
    print("Stopping system...")


# ==========================================
# 6. CLEANUP
# ==========================================
finally:

    camera.release()
    cv2.destroyAllWindows()

    # Give Windows time to close OpenCV window
    cv2.waitKey(1)

    print("System stopped.")