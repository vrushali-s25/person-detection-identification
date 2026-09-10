from ultralytics import YOLO
import cv2

print("========================================")
print("       CROWD PERSON DETECTION")
print("========================================")

# Load YOLO
print("Loading YOLO...")
model = YOLO("yolo11n.pt")
print("YOLO loaded successfully!")

# Open video
video = cv2.VideoCapture("video.mp4")

if not video.isOpened():
    print("ERROR: Could not open video.mp4")
    exit()

print("Video started!")
print("Press Q to close.")

while True:

    ret, frame = video.read()

    if not ret:
        print("Video finished.")
        break

    # Resize for faster processing
    frame = cv2.resize(frame, (640, 360))

    # Detect ONLY people
    results = model(
        frame,
        classes=[0],
        conf=0.35,
        verbose=False
    )

    # Draw detections
    output = results[0].plot()

    # Show
    cv2.imshow("Crowd Person Detection", output)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

print("Detection stopped.")