import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# link do modelo
MODEL_URL = "https://drive.google.com/uc?id=1Rhz9jC899ORxUtq4UNu33hh_AB6vtzWC"

# baixar modelo se não existir
if not os.path.exists("model.h5"):
    gdown.download(MODEL_URL, "model.h5", quiet=False)

# carregar modelo
model = tf.keras.models.load_model("model.h5")

classes = [
    "Glioma Tumor",
    "Meningioma Tumor",
    "No Tumor",
    "Pituitary Tumor"
]

st.title("🧠 Brain Tumor Detection")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, use_container_width=True)

    img = image.resize((224, 224))
    img = np.array(img)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    predicted_class = classes[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
