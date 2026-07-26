import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model sekali saja
model = tf.keras.models.load_model("cnn_model_optimized_extra.keras")

# Sesuaikan dengan kelas model kalian
classes = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

left, center, right = st.columns([2, 3, 2])



st.title("Prediksi Gambar CNN")


uploaded_file = st.file_uploader(
    "Upload gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Gambar yang diupload", width=300)

    image = image.resize((96,96))

    image = np.array(image).astype("float32")

    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    index = np.argmax(prediction)

    confidence = prediction[0][index]

    st.success(f"Prediksi : {classes[index]}")

    st.write(f"Confidence : {confidence*100:.2f}%")