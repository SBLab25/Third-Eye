import speech_recognition as sr
import os
import webbrowser
import boto3
import pydub
from pydub import playback
def synthesize_speech(text,output_filename):
    polly = boto3.client('polly', region_name='us-east-1')
    response = polly.synthesize_speech(
        VoiceId='Ruth',
        OutputFormat='mp3', 
        Text=text, 
        Engine='neural'
    )

    with open(output_filename, 'wb') as f:
        f.write(response['AudioStream'].read())

def play_audio(file):
    sound = pydub.AudioSegment.from_file(file, foormat="mp3")
    playback.play(sound)

# Function to convert speech to text
def listen():
    r = sr.Recognizer()
    r.energy_threshold =4000 
    with sr.Microphone() as source:
        print("Listening...")
        synthesize_speech("Listening","Listening.mp3")
        play_audio("Listening.mp3")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print(f"User said: {text}")
        return text.lower()
    except sr.UnknownValueError:
        print("Sorry, I could not understand what you said.")
        synthesize_speech("Sorry, I could not understand what you said.","Not.mp3")
        play_audio("Not.mp3")
        listen()
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        synthesize_speech("Could not request results","N.mp3")
        play_audio("N.mp3")
        listen()

# Function to run the AI voice assistant [ I DONT HAVE VOICE ASSISTANT SO A SIMPLE PROGRAM TO PRINT A PATTERN]
def open_jarvis():
    # os.system("python code1.py")
    path_to_code1 = "C:\ecs\AI_Voice_Assistance.py"
    os.system(f"python {path_to_code1}")
    

# Function to run object detection   , presss q to stop
def open_object_detection():
    path_to_code2 = "C:\ecs\Object.py"
    os.system(f"python {path_to_code2}")


def open_hand_detection():
    path_to_code3 = "C:\ecs\Hand.py"
    os.system(f"python {path_to_code3}")

def open_browser_and_search(query):
    url = f'https://www.google.com/search?q={query}'
    webbrowser.open_new_tab(url)

def open_browser_and_open(query):
    url = f'https://www.{query}.com/'
    webbrowser.open_new_tab(url)

def play_youtube_or_spotify():
    path_to_code1 = "C:\ecs\youtube.py"
    os.system(f"python {path_to_code1}")

# Main function to listen to user commands2
if __name__ == "__main__":
    while True:
        user_input = listen()
        user_input = user_input.lower()
        words = user_input.split()
        
        if 'tanya' in user_input:
            open_jarvis()
        if 'object' in user_input:
            open_object_detection()
        if 'hand' in user_input:
            open_hand_detection()
            
        if len(words) > 0 and words[0].lower() == "search":
            words = ' '.join(words[1:])
            open_browser_and_search(words)
            if input("Press Enter to continue after closing the browser..."):
                continue

        if len(words) > 0 and words[0].lower() == "open":
            words = ' '.join(words[1:])
            open_browser_and_open(words)
            if input("Press Enter to continue after closing the browser..."):
                continue
            
        if 'youtube' in user_input or 'spotify' in user_input:
            play_youtube_or_spotify()
            
        if 'stop' in user_input:
            os.system("taskkill /im python.exe /f")  # Terminate all running Python scripts
        if 'quit' in user_input:
            break

