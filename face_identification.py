import cv2
from insightface.app import FaceAnalysis

print("Starting face detection...")

# Load InsightFace
app = FaceAnalysis(name="buffalo_s")

# CPU mode + smaller detection size
app.prepare(ctx_id=-1, det_size=(320, 320))

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam")
    exit()

# Reduce camera resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera started!")
print("Press Q to close.")

frame_count = 0
last_faces = []

while True:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read camera frame")
        break

    frame_count += 1

    # Run face detection every 5th frame
    if frame_count % 5 == 0:
        small_frame = cv2.resize(frame, (320, 240))
        last_faces = app.get(small_frame)

    # Draw previous detection results
    for face in last_faces:

        box = face.bbox.astype(int)

        # Convert coordinates back to original frame size
        x1 = int(box[0] * 2)
        y1 = int(box[1] * 2)
        x2 = int(box[2] * 2)
        y2 = int(box[3] * 2)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face Detected",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Detection Test", frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Face detection stopped.")