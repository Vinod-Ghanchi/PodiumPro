from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import librosa
import speech_recognition as sr
import nltk
from textblob import TextBlob
import re
import os

app = Flask(__name__)

# Load the trained SVM model and feature coefficients
model = joblib.load('svm2.joblib')
feature_coefficients = joblib.load('svm2_coefficients.joblib')

# Function to extract features from audio (similar to your feature extraction logic)
def extract_audio_features(audio_file):
    # Load and process the audio file using Librosa
    sample_rate = librosa.get_samplerate(audio_file)
    y, sr_librosa = librosa.load(audio_file, sr=sample_rate)

    # Speech-to-Text Conversion
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)

    transcribed_text = recognizer.recognize_google(audio)

    # Tokenize the transcribed text into words for articulation rate
    words = nltk.word_tokenize(transcribed_text)

    # Calculate the articulation rate (words per second)
    duration_of_speech = len(y) / sr_librosa  # in seconds
    articulation_rate = len(words) / duration_of_speech

    # Calculate speech rate (words per minute)
    speech_rate = len(words) / (duration_of_speech / 60)

    # Sentiment Analysis (Speech Mood)
    text_blob = TextBlob(transcribed_text)
    speech_mood = text_blob.sentiment.polarity

    # Filler Words Detection
    # Define a list of common filler words
    filler_words = ["um", "uh", "like", "you know", "so", "actually", "basically", "literally", "totally", "seriously", "well", "anyway", "apparently", "honestly", "right", "I mean", "sort of", "kind of"]

    # Count the occurrences of filler words
    filler_word_count = 0
    for word in filler_words:
        filler_word_count += len(re.findall(r'\b' + word + r'\b', transcribed_text, flags=re.IGNORECASE))

    # F0 Statistics (Fundamental Frequency)
    # You may need a separate library or model to calculate F0 statistics
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    average_f0 = f0[f0 > 0].mean()


    # Return extracted features as a NumPy array
    return np.array([articulation_rate, filler_word_count, average_f0, speech_rate, speech_mood])

# Provide feedback based on the predicted class and feature coefficients
def provide_feedback(predicted_class, feature_feedback):
    feedback_text = f"Predicted Class: {predicted_class[0]}\n\nFeature Feedback:\n"

    for feature, coefficient in feature_feedback.items():
        if feature == "Articulation Rate":
            if coefficient > 0:
                feedback_text += f"{feature}: Your articulation rate is relatively high. This is good for clarity and precision in speech. Consider maintaining or refining this aspect.\n"
            elif coefficient < 0:
                feedback_text += f"{feature}: Your articulation rate is relatively low. Consider focusing on improving clarity and precision in your speech.\n"
            else:
                feedback_text += f"{feature}: Your articulation rate is at a neutral level.\n"

        elif feature == "Filler Word Count":
            if coefficient > 0:
                feedback_text += f"{feature}: There is a relatively high occurrence of filler words in your speech. Consider reducing their usage for clearer communication.\n"
            elif coefficient < 0:
                feedback_text += f"{feature}: Your usage of filler words is relatively low. Maintain or continue minimizing their use.\n"
            else:
                feedback_text += f"{feature}: Your filler word count is at a neutral level.\n"

        elif feature == "Average F0":
            if coefficient > 0:
                feedback_text += f"{feature}: Your average fundamental frequency is relatively high. Maintain or refine your pitch variation for engaging speech.\n"
            elif coefficient < 0:
                feedback_text += f"{feature}: Your average fundamental frequency is relatively low. Consider adding more pitch variation for expressive speech.\n"
            else:
                feedback_text += f"{feature}: Your average fundamental frequency is at a neutral level.\n"

        elif feature == "Speech Rate":
            if coefficient > 0:
                feedback_text += f"{feature}: Your speech rate is relatively high. Maintain or refine your current speech pace.\n"
            elif coefficient < 0:
                feedback_text += f"{feature}: Your speech rate is relatively low. Consider adjusting your speech pace for better engagement.\n"
            else:
                feedback_text += f"{feature}: Your speech rate is at a neutral level.\n"

        elif feature == "Speech Mood":
            if coefficient > 0:
                feedback_text += f"{feature}: Positive sentiment detected. Maintain a positive and engaging tone in your speech.\n"
            elif coefficient < 0:
                feedback_text += f"{feature}: Negative sentiment detected. Consider incorporating more positive expressions for a more favorable communication style.\n"
            else:
                feedback_text += f"{feature}: Your speech mood is at a neutral level.\n"

    return feedback_text

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for processing audio file and getting feedback
@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'})

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded file to the 'uploads' directory
    uploads_dir = os.path.join(os.getcwd(), 'uploads')
    audio_path = os.path.join(uploads_dir, 'recorded_audio.wav')
    audio_file.save(audio_path)

    #  Extract features from the new audio file
    new_audio_data = extract_audio_features(audio_path)

    # Make predictions using the loaded model
    predicted_class = model.predict([new_audio_data])
    
    # Provide feature feedback based on feature coefficients
    feature_feedback = {
        "Articulation Rate": feature_coefficients[0, 0],
        "Filler Word Count": feature_coefficients[0, 1],
        "Average F0": feature_coefficients[0, 2],
        "Speech Rate": feature_coefficients[0, 3],
        "Speech Mood": feature_coefficients[0, 4],
    }

    # Generate feedback text
    feedback_text = provide_feedback(predicted_class, feature_feedback)

    return jsonify({'predicted_class': predicted_class[0], 'feedback_text': feedback_text})

    

if __name__ == '__main__':
    app.run(debug=True)
