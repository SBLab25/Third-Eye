import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import pyttsx3
from math import sqrt

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize webcam feed
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set frame height
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Utility function to calculate Euclidean distance
def distance(point1, point2):
    return sqrt((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2)

# Function to handle speaking gestures
def speak_gesture(gesture):
    text_to_speak = ""
    if gesture == "Hello":
        text_to_speak = "Hello"
    elif gesture == "fine":
        text_to_speak = "Fine"
    elif gesture == "Not good":
        text_to_speak = "Not good"
    elif gesture == "cool":
        text_to_speak = "Cool"
    elif gesture == "Help":
        text_to_speak = "Please help"
    elif gesture == "Thank you":
        text_to_speak = "Thank You"
    # Add more two-hand gestures here
    # ...
    
    if text_to_speak:
        engine.say(text_to_speak)
        engine.runAndWait()

# Function to detect single-hand gestures
def detect_single_hand_gesture(hand):
    landmarks = hand['landmarks']
    
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    index_finger_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_finger_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_finger_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]

    # Calculate thumb-index distance for gesture checks
    thumb_index_distance = distance(thumb_tip, index_finger_tip)

    # Confidence threshold
    confidence_threshold = 0.7  # Adjust this value based on testing

    # Gesture recognition with confidence scoring
    confidence_scores = {}

    # Check for open hand
    open_hand_score = (
        (index_finger_tip.y < index_finger_mcp.y) + 
        (middle_finger_tip.y < middle_finger_mcp.y) + 
        (ring_finger_tip.y < ring_finger_mcp.y) + 
        (pinky_tip.y < pinky_mcp.y) + 
        (thumb_tip.y < thumb_mcp.y) +
        (thumb_index_distance > 0.1)
    ) / 6.0  # Normalizing to a score between 0 and 1
    confidence_scores["Hello"] = open_hand_score

    # Check for thumbs up
    thumbs_up_score = (
        (thumb_tip.y < index_finger_tip.y) + 
        (thumb_tip.y < middle_finger_tip.y) + 
        (thumb_tip.y < ring_finger_tip.y) + 
        (thumb_tip.y < pinky_tip.y) + 
        (thumb_index_distance > 0.2)
    ) / 5.0
    confidence_scores["fine"] = thumbs_up_score

    # Check for thumbs down
    thumbs_down_score = (
        (thumb_tip.y > index_finger_tip.y) + 
        (thumb_tip.y > middle_finger_tip.y) + 
        (thumb_tip.y > ring_finger_tip.y) + 
        (thumb_tip.y > pinky_tip.y) +
        (index_finger_tip.y > index_finger_mcp.y) + 
        (middle_finger_tip.y > middle_finger_mcp.y)
    ) / 6.0
    confidence_scores["Not good"] = thumbs_down_score

    # Check for peace sign
    peace_sign_score = (
        (index_finger_tip.y < thumb_tip.y) + 
        (middle_finger_tip.y < thumb_tip.y) + 
        (ring_finger_tip.y > thumb_tip.y) + 
        (pinky_tip.y > thumb_tip.y) +
        (thumb_index_distance > 0.15)
    ) / 5.0
    confidence_scores["cool"] = peace_sign_score

    # Check for Fist
    fist_score = (
        (index_finger_tip.y > index_finger_mcp.y) + 
        (middle_finger_tip.y > middle_finger_mcp.y) + 
        (ring_finger_tip.y > ring_finger_mcp.y) + 
        (pinky_tip.y > pinky_mcp.y) + 
        (thumb_tip.x < index_finger_tip.x) + 
        (thumb_tip.y < index_finger_tip.y)
    ) / 6.0
    confidence_scores["Help"] = fist_score

    # Determine the gesture with the highest confidence
    best_gesture = max(confidence_scores, key=confidence_scores.get)
    best_confidence = confidence_scores[best_gesture]

    # Only return the gesture if its confidence score is above the threshold
    if best_confidence >= confidence_threshold:
        return best_gesture, best_confidence

    return None, 0.0  # Return None if no gesture is recognized with enough confidence

# Function to detect two-hand gestures
def detect_two_hand_gesture(hand1, hand2):
    landmarks1 = hand1['landmarks']
    landmarks2 = hand2['landmarks']
    
    # Example: Both thumbs up
    thumb1 = landmarks1[mp_hands.HandLandmark.THUMB_TIP]
    thumb2 = landmarks2[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip1 = landmarks1[mp_hands.HandLandmark.THUMB_IP]
    thumb_ip2 = landmarks2[mp_hands.HandLandmark.THUMB_IP]
    index1 = landmarks1[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index2 = landmarks2[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip1 = landmarks1[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_pip2 = landmarks2[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    
    # Simple condition for thumbs up on both hands
    thumbs_up = (
        thumb1.y < thumb_ip1.y and
        thumb2.y < thumb_ip2.y and
        index1.y > index_pip1.y and
        index2.y > index_pip2.y
    )
    
    if thumbs_up:
        return "Thank you", 1.0
    
    # Example: Both fists
    index_mcp1 = landmarks1[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    index_mcp2 = landmarks2[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp1 = landmarks1[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    middle_mcp2 = landmarks2[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp1 = landmarks1[mp_hands.HandLandmark.RING_FINGER_MCP]
    ring_mcp2 = landmarks2[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp1 = landmarks1[mp_hands.HandLandmark.PINKY_MCP]
    pinky_mcp2 = landmarks2[mp_hands.HandLandmark.PINKY_MCP]
    
    fists = (
        landmarks1[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > index_mcp1.y and
        landmarks1[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > middle_mcp1.y and
        landmarks1[mp_hands.HandLandmark.RING_FINGER_TIP].y > ring_mcp1.y and
        landmarks1[mp_hands.HandLandmark.PINKY_TIP].y > pinky_mcp1.y and
        landmarks2[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > index_mcp2.y and
        landmarks2[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > middle_mcp2.y and
        landmarks2[mp_hands.HandLandmark.RING_FINGER_TIP].y > ring_mcp2.y and
        landmarks2[mp_hands.HandLandmark.PINKY_TIP].y > pinky_mcp2.y
    )
    
    if fists:
        return "Both Fists", 1.0
    
    # Add more two-hand gesture detections here
    # ...
    
    # If no two-hand gesture is detected
    return None, 0.0

# Gesture recognition function with confidence scoring
def detect_gesture(hands_data):
    if len(hands_data) == 1:
        # Single-hand gesture
        return detect_single_hand_gesture(hands_data[0])
    elif len(hands_data) == 2:
        # Two-hand gesture
        return detect_two_hand_gesture(hands_data[0], hands_data[1])
    return None, 0.0

# Start MediaPipe Hands
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Allow up to two hands
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
) as hands:
    last_gesture = None
    last_time = time.time()
    delay = 1.0  # Delay in seconds before allowing another gesture detection
    gesture_history = []  # To keep track of the gesture detections
    gesture_confidence_threshold = 5  # The number of consecutive detections needed to show a gesture

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: No frame captured.")
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        hands_data = []
        # Process detected hands
        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                hand_info = {
                    'landmarks': hand_landmarks.landmark,
                    'handedness': handedness.classification[0].label  # 'Left' or 'Right'
                }
                hands_data.append(hand_info)
                
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Detect gesture based on hands data
        gesture, confidence = detect_gesture(hands_data)

        # Check if sufficient time has passed before changing the gesture
        current_time = time.time()
        if gesture:
            gesture_history.append(gesture)
            
            # Keep only the last 10 gestures in history
            if len(gesture_history) > 10:
                gesture_history.pop(0)

            # Check if the same gesture has been consistently detected
            if gesture_history.count(gesture) >= gesture_confidence_threshold:
                if gesture != last_gesture or (current_time - last_time) > delay:
                    last_gesture = gesture
                    last_time = current_time
                    # Speak the meaning of the gesture in a separate thread
                    threading.Thread(target=speak_gesture, args=(last_gesture,)).start()

        # Only display the gesture if it's been consistently detected
        if last_gesture:
            cv2.putText(frame, f"{last_gesture} (Confidence: {confidence:.2f})", 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame with landmarks and detected gesture
        cv2.imshow('Hand Gesture Recognition', frame)

        # Break loop with 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()