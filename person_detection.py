from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam")
    exit()

print("Person detection started!")
print("Press Q to close.")

while True:

    # Read frame
    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read camera frame")
        break

    # Detect only people
    results = model(frame, classes=[0], conf=0.5)

    # Draw bounding boxes
    output = results[0].plot()

    # Display result
    cv2.imshow("Person Detection", output)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera
camera.release()
cv2.destroyAllWindows()