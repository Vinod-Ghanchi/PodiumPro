from flask import Flask, request, jsonify, render_template
import os
import joblib
import librosa
import speech_recognition as sr
import nltk
from textblob import TextBlob
import re
import numpy as np
import tempfile  # Import the tempfile module


# Initialize Flask application
app = Flask(__name__)

# Load the trained SVM model and feature coefficients
model = joblib.load('svm2.joblib')
feature_coefficients = joblib.load('svm2_coefficients.joblib')

def extract_audio_features(audio_file):
    """
    Extract features from the audio file, similar to the initially provided logic.
    """
    sample_rate = librosa.get_samplerate(audio_file)
    y, sr_librosa = librosa.load(audio_file, sr=sample_rate)

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)

    transcribed_text = recognizer.recognize_google(audio)
    words = nltk.word_tokenize(transcribed_text)
    duration_of_speech = len(y) / sr_librosa
    articulation_rate = len(words) / duration_of_speech
    speech_rate = len(words) / (duration_of_speech / 60)
    text_blob = TextBlob(transcribed_text)
    speech_mood = text_blob.sentiment.polarity

    filler_words = ["um", "uh", "like", "you know", "so", "actually", "basically", "literally", "totally", "seriously", "well", "anyway", "apparently", "honestly", "right", "I mean", "sort of", "kind of"]
    filler_word_count = sum(len(re.findall(r'\b' + word + r'\b', transcribed_text, flags=re.IGNORECASE)) for word in filler_words)

    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    average_f0 = np.nanmean(f0[f0 > 0])

    return np.array([articulation_rate, filler_word_count, average_f0, speech_rate, speech_mood])

# Provide feedback based on the predicted class and feature coefficients
def provide_feedback(predicted_class, feature_coefficients):
    print(feature_coefficients, predicted_class)

    feedback_text = f"Predicted Class: {predicted_class[0]}\n\nFeature Feedback:\n"

    for feature, coefficient in feature_coefficients.items():
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



    print('---------------------------------------------------------')
    print(feedback_text)

    return feedback_text

@app.route('/')
def index():
    return render_template('index.html')
import logging



@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        logging.error('No audio file provided')
        return jsonify({'error': 'No audio file provided'})

    file = request.files['audio']
    if file.filename == '':
        logging.error('No selected file')
        return jsonify({'error': 'No selected file'})

    if file:
        # Use tempfile to create a temporary file
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        filepath = temp_audio_file.name
        try:
            file.save(filepath)
        except Exception as e:
            logging.error(f'Error saving temporary file: {str(e)}')
            return jsonify({'error': str(e)})

        try:
            new_audio_data = extract_audio_features(filepath)
            new_audio_data = new_audio_data.reshape(1, -1)  # Reshape the input data to 2D
            predicted_class = model.predict(new_audio_data)
            feedback_text = provide_feedback(predicted_class, feature_coefficients)
            logging.info(f"Predicted Class: {predicted_class[0]}")
            logging.info(f"Feedback Text: {feedback_text}")

            # Check if feedback_text is empty or undefined
            if not feedback_text:
                logging.error("Feedback text is empty or undefined")
                temp_audio_file.close()
                os.remove(filepath)
                return jsonify({'error': 'Failed to generate feedback text'})

            temp_audio_file.close()
            os.remove(filepath)
            return jsonify({'predicted_class': predicted_class[0], 'feedback_text': feedback_text})
        except Exception as e:
            logging.error(f'Error processing audio file: {str(e)}')
            temp_audio_file.close()
            os.remove(filepath)
            return jsonify({'error': str(e)})
        
if __name__ == '__main__':
    app.run(debug=True ,port=5000,use_reloader=False)
