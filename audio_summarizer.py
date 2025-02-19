import openai
import os
import speech_recognition as sr
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        transcription = recognizer.recognize_google(audio)
        return transcription
    except sr.RequestError as e:
        return f"Error: {e}"
    except sr.UnknownValueError:
        return "Could not understand audio."

def summarize_text(text):
    prompt = f"Summarize the following transcription concisely:\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a summarization assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    audio_path = input("Enter the path to your audio file: ")
    transcription = transcribe_audio(audio_path)
    print("Transcription:", transcription)
    summary = summarize_text(transcription)
    print("\nSummary:\n", summary)
