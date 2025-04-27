import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from PIL import Image
import os
import glob
import time
import pyperclip
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# ===========================
# Load Custom CSS for Styling
# ===========================
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Skipping custom styling.")

local_css("style.css")

st.markdown("""
    <style>
        /* When hovering over the text input box, change cursor to pointer */
        textarea {
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

# ===========================
# Initialize MediaPipe hands and drawing utilities
# ===========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ===========================
# Load Trained Model and Scaler
# ===========================
@st.cache_resource
# Load the Keras model
def load_sign_language_model(model_path):
    model = tf.keras.models.load_model(model_path)  # Load the .keras file model
    return model

# Load the Scaler
def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)  # Load the scaler from .pkl file
    return scaler

# File paths to your model and scaler
model_path = 'final_model.keras'  # Path to your model file
scaler_path = 'scaler.pkl'  # Path to your scaler file

# Load model and scaler
model = load_sign_language_model(model_path)
scaler = load_scaler(scaler_path)

with open('sign_language_features.pkl', 'rb') as f:
    data = pickle.load(f)
    labels = data['labels']



# ===========================
# Define Class Labels (Update if necessary)
# ===========================
class_labels = {
    0: 'are', 1: 'did', 2: 'doing', 3: 'eat', 
    4: 'going', 5: 'How', 6: 'is', 7: 'name',
    8: 'What', 9: 'Where', 10: 'Which', 11: 'you', 12: 'your'
}

# ===========================
# Preprocessing Functions
# ===========================
def extract_keypoints_from_image(image):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        image_np = np.array(image)
        if image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        elif image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        results = hands.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

        keypoints = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])

        if len(keypoints) == 63:
            keypoints += [0.0] * 63
        if len(keypoints) == 126:
            return np.array(keypoints).reshape(1, 126).astype(np.float32)  # <== Fix here


    return None

def extract_keypoints_from_frame(frame):
    # Use MediaPipe to process the frame and extract hand landmarks
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7) as hands:
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    keypoints = []

    # Check if hands are detected
    if results.multi_hand_landmarks:
        # Iterate through the detected hands and extract their landmarks
        for hand_landmarks in sorted(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x):
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

        # If only one hand is detected, pad the keypoints for the second hand
        if len(results.multi_hand_landmarks) == 1:
            keypoints.extend([0.0] * (63))  # Padding for the second hand
    else:
        # If no hands are detected, return None
        return None

    # Ensure the keypoints list has 126 values (63 per hand)
    if len(keypoints) != 126:
        return None  # In case of unexpected results

    # Reshape and return keypoints as a numpy array
    return np.array(keypoints).reshape(1, 126).astype(np.float32)
# ===========================
# Streamlit UI Setup
# ===========================
st.sidebar.image("apac_logo.jpg", width=120)
st.sidebar.title("APAC")
if st.sidebar.button("Explore APAC", key="explore_apac"):
    st.success("Explore APAC coming soon...")
if st.sidebar.button("New Chat", key="new_chat"):
    st.success("New chat feature coming soon...")
if st.sidebar.button("Recent", key="recent"):
    st.success("Recent chats will be available soon...")
st.sidebar.markdown("---")
if st.sidebar.button("APAC Manager", key="apac_manager"):
    st.success("APAC Manager is under development...")
if st.sidebar.button("Help", key="help"):
    st.info("Help and support will be available soon.")
if st.sidebar.button("Activity", key="activity"):
    st.success("Activity logs will be available shortly...")

# ===========================
# Main Page UI
# ===========================
st.title("Introducing APAC – AI-Powered Accessibility Chatbot")
st.write(
    "APAC is a chatbot that responds to sign language, using AI to convert sign language gestures "
    "to text and vice versa. Whether you use sign language, text, or commands, APAC ensures seamless interaction."
)

# ===========================
# Sidebar Input Selection
# ===========================
input_choice = st.selectbox("Select Input Type", ("Upload Image", "Live Webcam", "Text to Sign"))
if "sentence" not in st.session_state:
    st.session_state.sentence = ""

# ===========================
# Upload Image Handling
# ===========================
if input_choice == "Upload Image":
    uploaded_file = st.file_uploader("Upload a sign language image:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        keypoints = extract_keypoints_from_image(image)
        if keypoints is not None:
            keypoints_scaled = scaler.transform(keypoints.reshape(1, -1))  # Important
            prediction = model.predict(keypoints_scaled, verbose=0)
            predicted_class = int(np.argmax(prediction, axis=1)[0])
            predicted_label = class_labels.get(predicted_class, "Unknown")
            st.text_area("Translated Output", value=predicted_label, height=100, key="output_box", disabled=True)
            if st.button("Copy Text", key="copy_text_upload"):
                pyperclip.copy(predicted_label)
                st.success("Text copied to clipboard!")
        else:
            st.warning("No hand detected or unable to extract keypoints.")

# ===========================
# Live Webcam Handling (Updated with Chatbot)
# ===========================
if input_choice == "Live Webcam":
    if st.button("Start Webcam for Live Translation", key="start_webcam"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        sentence = ""

        if "stop_webcam" not in st.session_state:
            st.session_state.stop_webcam = False

        if st.session_state.stop_webcam:
            cap.release()
            st.info("Webcam stopped.")


        if st.button("Reset", key="reset"):
            sentence = ""
            st.session_state.sentence = ""

        st.info("Webcam started! Show your signs.")

        if 'counter' not in st.session_state:
            st.session_state.counter = 0

        last_prediction_time = time.time()
        previous_word = None
        sentence_output = st.empty()
        chatbot_response = st.empty()

        def extract_keypoints_2hands(multi_hand_landmarks):
            keypoints = []
            for hand_landmarks in multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
            if len(multi_hand_landmarks) == 1:
                keypoints.extend([0.0] * 63)
            return np.array(keypoints).reshape(1, 126).astype(np.float32)  # <== Fix here


        def get_chatbot_response(sentence):
            query = sentence.strip().lower()
            responses = {
                "how are you": "I'm doing great! 😊",
                "what are you doing": "I'm helping you translate signs!",
                "did you eat": "Yes, I had some data bytes! 😄",
                "where are you going": "I'm staying right here to assist you!",
                "what is your name": "I'm APAC, your sign language assistant!"
            }
            return responses.get(query, None)

        with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or st.session_state.stop_webcam:
                    st.info("Webcam Stopped.")
                    break


                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                current_time = time.time()

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    if current_time - last_prediction_time >= 4:
                        keypoints_array = extract_keypoints_2hands(results.multi_hand_landmarks)
                        if keypoints_array is not None and keypoints_array.shape == (1, 126):
                            keypoints_flatten = keypoints_array.reshape(1, -1)
                            keypoints_scaled = scaler.transform(keypoints_flatten)
                            prediction = model.predict(keypoints_scaled, verbose=0)

                            predicted_class = np.argmax(prediction)  # <== Add this
                            predicted_label = labels[predicted_class]  # <== Add this

                            if predicted_label != previous_word:
                                if previous_word in ["How", "how", "What", "what", "where", "Where"] and predicted_label == "your":
                                    sentence += " are"
                                    previous_word = "are"  # Update previous_word also
                                else:
                                    sentence += " " + predicted_label
                                    previous_word = predicted_label

                                last_prediction_time = current_time


                                # Check for chatbot response
                                cleaned_sentence = sentence.lower().strip()
                                if any(q in cleaned_sentence for q in [
                                    "how are you", "what are you doing", "did you eat", "where are you going", "what is your name"
                                ]):
                                    response = get_chatbot_response(cleaned_sentence)
                                    if response:
                                        chatbot_response.success(f"🤖 APAC: {response}")

                stframe.image(frame, channels="BGR")
                sentence_output.text_area(f"Translated Output {st.session_state.counter}",
                                          value=sentence.strip(), height=150,
                                          key=f"sentence_output_{st.session_state.counter}", disabled=True)

                st.session_state.counter += 1
                st.session_state.sentence = sentence.strip()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    st.text_area("Final Translated Output", value=st.session_state.sentence,
                 height=150, key="final_sentence_output", disabled=True)

    if st.button("Copy Text ", key="copy_text_live"):
        if st.session_state.sentence.strip():
            pyperclip.copy(st.session_state.sentence.strip())
            st.success("Text copied to clipboard!")
        else:
            st.warning("No text to copy!")



# ===========================
# Text-to-Sign Handling
# ===========================
elif input_choice == "Text to Sign":
    text_input = st.text_input("Enter text to translate to sign language:")
    image_directory = "Data"

    if text_input:
        words = text_input.split()
        st.write("Corresponding Sign Language Images:")

        for word in words:
            folder_path = os.path.join(image_directory, word.lower())
            if os.path.exists(folder_path):
                image_files = glob.glob(os.path.join(folder_path, "*.*"))
                if image_files:
                    st.image(image_files[0], caption=word.capitalize(), use_column_width=True)
                else:
                    st.warning(f"No images found inside '{word}' folder.")
            else:
                st.warning(f"No folder found for '{word}'.")
