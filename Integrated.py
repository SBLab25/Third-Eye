# Existing imports remain unchanged
from groq import Groq
from PIL import ImageGrab, Image
from openai import OpenAI
from os import system
from faster_whisper import WhisperModel
import speech_recognition as sr
import google.generativeai as genai
import pyperclip
import cv2
import pyaudio
import os
import time
import re
import sys
# Amazon Polly Neural Engine
import numpy as np
import sounddevice as sd
import boto3
from pydub import playback
import pydub 

#Hand Gesture
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import pyttsx3
from math import sqrt

#Object Detection
import cv2
import pyttsx3
import time
from ultralytics import YOLO

from supervision import BoxAnnotator

# Initialize voice assistant
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# Load YOLOv8 model
yolo_model = YOLO('yolov8n.pt')

# Set up MediaPipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

wake_word = 'jarvis'
groq_client = Groq(api_key='Enter your Secret API Key here')
genai.configure(api_key='Enter your Secret API Key here')
# openai_client = OpenAI(api_key='Enter your Secret API Key here')
web_cam = cv2.VideoCapture(0)

r = sr.Recognizer()
source = sr.Microphone()

sys_msg = (
    'You are a multi-modal AI voice assistant. Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response before ! '
    'adding new tokens to the response. Do not expect or request images, just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your responses clear and concise, avoiding any verbosity.'
)

convo = [{'role': 'system', 'content': sys_msg},]

generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 512
}

safety_settings = [
    {
        'category':'HARM_CATEGORY_HARASSMENT',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category':'HARM_CATEGORY_HATE_SPEECH',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category':'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category':'HARM_CATEGORY_DANGEROUS_CONTENT',
        'threshold': 'BLOCK_NONE'
    },
]

model = genai.GenerativeModel('gemini-1.5-flash-latest',
                              generation_config=generation_config,
                              safety_settings=safety_settings)

num_cores = os.cpu_count()
whisper_size = 'base'
whisper_model = WhisperModel(
    whisper_size,
    device='cpu',
    compute_type='int8',
    cpu_threads=num_cores //2,
    num_workers=num_cores //2
)

# Function to synthesize speech using Amazon Polly
def synthesize_speech(text, output_filename):
    polly = boto3.client('polly', region_name='us-east-1')
    response = polly.synthesize_speech(
        VoiceId='Ruth',
        OutputFormat='mp3', 
        Text=text, 
        Engine='neural'
    )

    with open(output_filename, 'wb') as f:
        f.write(response['AudioStream'].read())

# Function to play audio file
def play_audio(file):
    sound = pydub.AudioSegment.from_file(file, format="mp3")
    playback.play(sound)

def groq_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\n  IMAGE CONTEXT: {img_context}'
    convo.append({'role': 'user', 'content': prompt})
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    convo.append(response)

    return response.content

def function_call(prompt):
    sys_msg = (
        'You are an AI function calling model. You will determine whether extracting the users clipboard content, '
        'taking a screenshot, capturing the webcam or calling no functions is best for a voice assistant to respond '
        'to the users prompt. The webcam can be assumed to be a normal laptop webcam facing the user. You will '
        'respond with only one selection from this list: ["extract clipboard", "take screenshot", "capture webcam", "None"] \n'
        'Do not respond with anything but the most logical selection from that list with no explanations. Format the '
        'function call name exactly as I listed.'
    )

    function_convo = [{'role': 'system', 'content': sys_msg},
                      {'role': 'user', 'content': prompt}]
    
    chat_completion = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message

    return response.content

def take_screenshot():
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality=15)

def web_cam_capture():
    if not web_cam.isOpened():
        print('Webcam is not open')
        exit()

    path = 'webcam_capture.jpg'
    ret, frame = web_cam.read()
    cv2.imwrite(path, frame)

def get_clipboard_text():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print('No text found in clipboard')
        return None

def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    prompt = (
        'You are the vision analysis AI that provides semantic meaning from images to provide context '
        'to send to another AI that will create a response to the user. Do not respond as the AI assistant '
        'to the user. Instead take the user prompt input and try to extract all meaning from the photo '
        'relevant to the user prompt. Then generate as much objective data about the image for the AI '
        f'assistant who will respond to the user. \nUSER PROMPT: {prompt}'
    )
    response = model.generate_content([prompt, img])
    return response.text

def speak(text):
    output_filename = 'speech.mp3'
    synthesize_speech(text, output_filename)
    play_audio(output_filename)


def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text

def callback(recognizer, audio):
    prompt_audio_path = 'prompt.wav'
    with open(prompt_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())

    prompt_text = wav_to_text(prompt_audio_path)
    clean_prompt = extract_prompt(prompt_text, wake_word)

    if clean_prompt:
        print(f'USER: {clean_prompt}')
        call = function_call(clean_prompt)
        if 'take screenshot' in call:
            print('Taking Screenshot')
            take_screenshot()
            visual_context = vision_prompt(prompt=clean_prompt, photo_path='screenshot.jpg')
        elif 'capture webcam' in call:
            print('Capturing webcam')
            web_cam_capture()
            visual_context = vision_prompt(prompt=clean_prompt, photo_path='webcam_capture.jpg')
        elif 'extract clipboard' in call:
            print('Extracting clipboard text')
            paste = get_clipboard_text()
            clean_prompt = f'{clean_prompt}\n\n CLIPBOARD CONTENT: {paste}'
            visual_context = None
        else:
            visual_context = None

        response = groq_prompt(prompt=clean_prompt, img_context=visual_context)
        print(response)
        speak(response)

def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
    print('\nSay', wake_word, 'followed with your prompt. \n')
    r.listen_in_background(source, callback)

    while True:
        time.sleep(0.5)

def extract_prompt(transcribed_text, wake_word):
    pattern = rf'\b{re.escape(wake_word)}[\s,.?!]*([A-Za-z0-9].*)'
    match = re.search(pattern, transcribed_text, re.IGNORECASE)

    if match:
        prompt = match.group(1).strip()
        return prompt
    else:
        return None

def recognize_hand_gesture(frame):
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Add gesture-based commands
            # Example: thumb up gesture
            # Process hand landmarks and trigger actions based on gestures
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Custom gesture handling logic can be implemented here.
    return frame

def detect_objects(frame):
    # Run YOLO model on the frame
    results = yolo_model.predict(source=frame)
    
    # Annotate frame
    box_annotator = BoxAnnotator()
    annotated_frame = box_annotator.annotate(frame, results)
    return annotated_frame


#Hand Gesture
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
                mp_drawing = mp.solutions.drawing_utils
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


#Object Detection
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



def main():
    cap = cv2.VideoCapture(0)  # Open webcam
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Listen for voice commands or detect gestures
        # Command for hand recognition
        if "recognize hand" in user_command:
            frame = recognize_hand_gesture(frame)
            speak("Hand gesture recognized")

        # Command for object detection
        elif "detect object" in user_command:
            frame = detect_objects(frame)
            speak("Object detection completed")

        # Display frame
        cv2.imshow("AI Assistant", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

start_listening()

