# app.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import json
import os

st.set_page_config(page_title="CNN Image Classifier", layout="centered")
st.title("CNN Image Classifier (Streamlit)")
st.write("Upload an image to get predictions.")

# Load model
@st.cache_resource
def load_model():
    if os.path.exists("saved_model"):
        return tf.keras.models.load_model("saved_model")
    elif os.path.exists("best_model.h5"):
        return tf.keras.models.load_model("best_model.h5")
    else:
        st.error("No model found. Please run train.py first.")
        return None

@st.cache_data
def load_class_names():
    if os.path.exists("class_names.json"):
        with open("class_names.json", "r") as f:
            return json.load(f)
    return []

model = load_model()
class_names = load_class_names()

IMG_SIZE = (224, 224)

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded and model is not None and class_names:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image (NO /255 because model already includes Rescaling)
    img = ImageOps.fit(image, IMG_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(img)

    input_tensor = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(input_tensor)
    topk = 3
    indices = preds[0].argsort()[-topk:][::-1]

    st.write("### 🔍 Top Predictions:")
    for idx in indices:
        st.write(f"- **{class_names[idx]}** — {preds[0][idx] * 100:.2f}%")

elif uploaded:
    st.warning("Model or class mapping missing. Please run train.py.")
