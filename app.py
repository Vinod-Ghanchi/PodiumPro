import streamlit as st
import librosa
import numpy as np
import joblib
import speech_recognition as sr
import nltk
from textblob import TextBlob
import re
from tempfile import NamedTemporaryFile

nltk.download('punkt')  # Make sure to download necessary NLTK data

# Define the feature extraction function
def extract_audio_features(audio_file):
    # Load and process the audio file using Librosa
    sample_rate = librosa.get_samplerate(audio_file)
    y, sr_librosa = librosa.load(audio_file, sr=sample_rate)

    # Speech-to-Text Conversion
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    transcribed_text = recognizer.recognize_google(audio_data)

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
    filler_words = ["um", "uh", "like", "you know", "so", "actually", "basically", "literally", "totally", "seriously", "well", "anyway", "apparently", "honestly", "right", "I mean", "sort of", "kind of"]
    filler_word_count = sum(transcribed_text.lower().count(fw) for fw in filler_words)

    # F0 Statistics (Fundamental Frequency)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    average_f0 = np.nanmean(f0[f0 > 0])

    # Return extracted features as a list along with transcribed text and raw feature values
    return np.array([articulation_rate, filler_word_count, average_f0, speech_rate, speech_mood]), transcribed_text, {
    "Articulation Rate": articulation_rate,
    "Filler Word Count": filler_word_count,
    "Average F0": average_f0,
    "Speech Rate": speech_rate,
    "Speech Mood": speech_mood
}
# Load the trained SVM model and feature coefficients
model = joblib.load('svm2.joblib')
feature_coefficients = joblib.load('svm2_coefficients.joblib')

feature_feedback = {
    "Articulation Rate": feature_coefficients[0, 0],
    "Filler Word Count": feature_coefficients[0, 1],
    "Average F0": feature_coefficients[0, 2],
    "Speech Rate": feature_coefficients[0, 3],
    "Speech Mood": feature_coefficients[0, 4],
}


# Prediction and feedback generation code unchanged
def provide_feedback(predicted_class, feature_feedback):
    feedback_text = f"Predicted Class: {predicted_class[0]}\n\nFeature Feedback:\n"

    for feature, coefficient in feature_feedback.items():
        if feature == "Articulation Rate":
            if coefficient > 0:
                feedback_text += f"{feature}:\nYour articulation rate is relatively high. This is good for clarity and precision in speech. Consider maintaining or refining this aspect.\n\n"
            elif coefficient < 0:
                feedback_text += f"{feature}:\nYour articulation rate is relatively low. Consider focusing on improving clarity and precision in your speech.\n\n"
            else:
                feedback_text += f"{feature}:\nYour articulation rate is at a neutral level.\n\n"

        elif feature == "Filler Word Count":
            if coefficient > 0:
                feedback_text += f"{feature}:\nThere is a relatively high occurrence of filler words in your speech. Consider reducing their usage for clearer communication.\n\n"
            elif coefficient < 0:
                feedback_text += f"{feature}:\nYour usage of filler words is relatively low. Maintain or continue minimizing their use.\n\n"
            else:
                feedback_text += f"{feature}:\nYour filler word count is at a neutral level.\n\n"

        elif feature == "Average F0":
            if coefficient > 0:
                feedback_text += f"{feature}:\nYour average fundamental frequency is relatively high. Maintain or refine your pitch variation for engaging speech.\n\n"
            elif coefficient < 0:
                feedback_text += f"{feature}:\nYour average fundamental frequency is relatively low. Consider adding more pitch variation for expressive speech.\n\n"
            else:
                feedback_text += f"{feature}:\nYour average fundamental frequency is at a neutral level.\n\n"

        elif feature == "Speech Rate":
            if coefficient > 0:
                feedback_text += f"{feature}:\nYour speech rate is relatively high. Maintain or refine your current speech pace.\n\n"
            elif coefficient < 0:
                feedback_text += f"{feature}:\nYour speech rate is relatively low. Consider adjusting your speech pace for better engagement.\n\n"
            else:
                feedback_text += f"{feature}:\nYour speech rate is at a neutral level.\n\n"

        elif feature == "Speech Mood":
            if coefficient > 0:
                feedback_text += f"{feature}:\nPositive sentiment detected. Maintain a positive and engaging tone in your speech.\n\n"
            elif coefficient < 0:
                feedback_text += f"{feature}:\nNegative sentiment detected. Consider incorporating more positive expressions for a more favorable communication style.\n\n"
            else:
                feedback_text += f"{feature}:\nYour speech mood is at a neutral level.\n\n"



    return feedback_text

def main():
    # Setup page configuration and theme
    st.set_page_config(
        page_title="Speech Analysis App",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Define the home page
    def home_page():
        st.title("🗣️ Speech Analysis Application")

        with st.container():
            st.markdown("""
                <style>
                .big-font {
                    font-size:20px !important;
                    color: #346751;
                }
                </style>
                """, unsafe_allow_html=True)
            
            st.markdown('<p class="big-font">This application analyzes various aspects of speech to provide insights into speech patterns, articulation rate, and emotional tone.</p>', unsafe_allow_html=True)

        # Use columns to layout the text for aesthetic appeal
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("What is articulation rate?")
            st.write("A measure of speech speed, indicating fluency.")
            
            st.header("What is Filler word count?")
            st.write('Counts hesitant sounds like "um", "uh", affecting clarity.')

        with col2:
            st.header("What is Average F0?")
            st.write("The average pitch of a person's voice.")
            
            st.header("What is Speech rate?")
            st.write("The speed of speech, impacting listener engagement.")

        st.header("What is Speech mood?")
        st.write("The emotional tone conveyed, affecting communication effectiveness.")
        
        if st.button('🎙️ Test an Audio'):
            st.session_state.current_page = 'audio_analysis_page'


    # Define the audio analysis page
    def audio_analysis_page():
        st.title('📊 Audio Feature Analysis and Classification')
        uploaded_file = st.file_uploader("Upload an audio file for analysis", type=["wav"], help="Accepts WAV format audio files")

        if uploaded_file is not None:
            # Processing feedback
            with st.spinner('Analyzing audio...'):
                # Use a temporary file to safely handle the uploaded file
                with NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    audio_file_path = tmp.name

                try:
                    # Extract features from the audio file
                    new_audio_data, transcribed_text, feature_values = extract_audio_features(audio_file_path)
                    
                    # Predict the class using the extracted features
                    predicted_class = model.predict([new_audio_data])

                    st.success("Analysis complete! See below for details and feedback.")

                    # Display the transcribed text and raw values of the extracted features in Streamlit
                    st.subheader("Extracted Feature Values:")
                    for feature, value in feature_values.items():
                        st.write(f"{feature}: {value:.2f}")

                    # Make predictions and generate feedback
                    feedback_text = provide_feedback(predicted_class, feature_values)
                    st.markdown(f"**Feedback:**\n{feedback_text}\n")

                except Exception as e:
                    st.error(f"An error occurred while processing the audio file: {e}")

    # Navigation logic
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home_page'

    if st.session_state.current_page == 'home_page':
        home_page()
    elif st.session_state.current_page == 'audio_analysis_page':
        audio_analysis_page()

if __name__ == "__main__":
    main()









