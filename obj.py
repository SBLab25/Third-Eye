import cv2
import pyttsx3
import time
from ultralytics import YOLO
import threading

# Load YOLO model
model = YOLO('yolov8n.pt')

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# Function to convert detected object name to speech
def announce_name(object_name):
    engine.say(f"{object_name} identified")
    engine.runAndWait()

# Function to calculate distance based on the object's bounding box height
def calculate_distance(bbox_height):
    # Assume an average object height in cm (e.g., a human is about 170 cm)
    known_height = 170  # Known height of the object in cm
    # Focal length in pixels (this value needs to be calibrated for your camera)
    focal_length = 55  # Focal length in pixels
    # Calculate the distance using the formula: distance = (known_height * focal_length) / bbox_height
    distance = (known_height * focal_length) / bbox_height
    return distance

# Function to capture video frames
def video_capture():
    global frame, is_frame_ready
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Lower resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while True:
        ret, current_frame = cap.read()
        if not ret:
            break
        frame = current_frame
        is_frame_ready = True

    cap.release()

# Start the video capture in a separate thread
frame = None
is_frame_ready = False
threading.Thread(target=video_capture, daemon=True).start()

# Initialize variables to store previously detected objects
last_detected_object = None
already_announced = False

# Set confidence threshold
CONFIDENCE_THRESHOLD = 0.5  # Only announce objects with confidence > 50%

# Add a detection delay to avoid repeating announcements quickly (in seconds)
DETECTION_DELAY = 5  # Delay between detections to avoid multiple announcements
last_detection_time = 0

while True:
    if is_frame_ready:
        current_time = time.time()

        # Perform object detection with YOLOv8
        results = model(frame)  # Detect objects in the current frame

        # Parse the detection results
        for result in results:
            if len(result.boxes) > 0:
                first_box = result.boxes[0]  # Get the first detected object (highest confidence)
                object_name = result.names[int(first_box.cls)]  # Get the object name
                confidence = first_box.conf.item()  # Convert tensor to float
                
                # Get bounding box dimensions
                bbox_height = int(first_box.xyxy[0][3] - first_box.xyxy[0][1])  # Height of bounding box in pixels
                
                # Calculate distance using bounding box height
                distance = calculate_distance(bbox_height)
                
                # Check if the confidence is above the threshold
                if confidence > CONFIDENCE_THRESHOLD:
                    # Display the object name, confidence, and distance on the video feed
                    cv2.putText(frame, f"Detected: {object_name} ({confidence:.2f})", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Distance: {distance:.2f} cm", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

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

        # Reset the flag
        is_frame_ready = False

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()