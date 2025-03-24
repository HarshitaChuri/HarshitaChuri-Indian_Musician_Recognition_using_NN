import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
import os

# ✅ Load the trained model
model = tf.keras.models.load_model("musician_recognition_model.h5")

# ✅ Artist Labels (as per dataset)
artist_names = [
    "Shankar Mahadevan", "Lata Mangeshkar", "Kishore Kumar", "Arijit Singh", 
    "Shreya Ghoshal", "A.R. Rahman", "Alka Yagnik", "Neha Kakkar", 
    "Asha Bhosle", "Sonu Nigam", "Mohammed Rafi"
]

# 🎵 Function to Convert Audio to Spectrogram
def audio_to_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig, ax = plt.subplots(figsize=(2, 2))  # 128x128 for model input
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    plt.axis("off")

    # Save spectrogram
    spectrogram_path = "temp_spectrogram.jpg"
    plt.savefig(spectrogram_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # Load and preprocess for model
    img = cv2.imread(spectrogram_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    return img

# ✅ Streamlit UI
st.title("🎵 Indian Musician Recognition 🎤")
st.write("Upload an **audio file** to predict the singer 🎶")

# 🎵 Upload Audio File
uploaded_file = st.file_uploader("Upload an MP3/WAV file", type=["mp3", "wav"])

if uploaded_file is not None:
    # Save uploaded file
    file_path = "uploaded_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 🎼 Convert Audio to Spectrogram
    st.write("🔄 **Processing Audio...**")
    spectrogram = audio_to_spectrogram(file_path)

    # 🎤 Predict the Musician
    prediction = model.predict(spectrogram)
    predicted_artist = artist_names[np.argmax(prediction)]

    # ✅ Display Prediction
    st.success(f"🎤 **Predicted Artist: {predicted_artist}**")
    st.image("temp_spectrogram.jpg", caption="Generated Spectrogram", use_column_width=True)

    # Cleanup
    os.remove(file_path)
    os.remove("temp_spectrogram.jpg")
