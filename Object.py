import cv2
import pyttsx3
import time
from ultralytics import YOLO

# Load a smaller YOLOv8 model for better performance (use 'yolov8n.pt' - nano version for smaller size)
model = YOLO('yolov8n.pt')

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# Function to convert detected object name to speech
def announce_name(object_name):
    engine.say(f"{object_name} identified")
    engine.runAndWait()

# Set up video capture from webcam
cap = cv2.VideoCapture(0)

# Set the resolution to lower for faster processing (e.g., 640x480 or lower)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize variables to store previously detected objects
last_detected_object = None
already_announced = False

# Set confidence threshold
CONFIDENCE_THRESHOLD = 0.5  # Only announce objects with confidence > 50%

# Add a detection delay to avoid repeating announcements quickly (in seconds)
DETECTION_DELAY = 5  # Delay between detections to avoid multiple announcements
last_detection_time = 0

# Reduce frame rate to improve performance (e.g., 10 frames per second)
FRAME_RATE = 10
frame_delay = 1.0 / FRAME_RATE

while cap.isOpened():
    start_time = time.time()

    ret, frame = cap.read()

    if not ret:
        break

    # Skip frames to reduce load on Raspberry Pi (process every nth frame)
    current_time = time.time()
    if current_time - last_detection_time > DETECTION_DELAY:
        # Perform object detection with YOLOv8
        results = model(frame)  # Detect objects in the current frame

        # Parse the detection results
        for result in results:
            if len(result.boxes) > 0:
                first_box = result.boxes[0]  # Get the first detected object (highest confidence)
                object_name = result.names[int(first_box.cls)]  # Get the object name
                confidence = first_box.conf.item()  # Convert tensor to float

                # Check if the confidence is above the threshold
                if confidence > CONFIDENCE_THRESHOLD:
                    # Display the object name and confidence on the video feed
                    cv2.putText(frame, f"Detected: {object_name} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    # Check if a new object is detected and not already announced
                    if object_name != last_detected_object:
                        last_detected_object = object_name
                        already_announced = False

                    # Announce the object if not already announced
                    if not already_announced:
                        announce_name(object_name)
                        already_announced = True
                        last_detection_time = current_time  # Update last detection time

    # Show video feed
    cv2.imshow("YOLOv8 Object Detection", frame)

    # Sleep for the remaining time to maintain frame rate
    time.sleep(max(0, frame_delay - (time.time() - start_time)))

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()