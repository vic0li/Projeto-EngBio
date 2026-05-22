import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown
import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # ✅ IMPORTANTE

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(
    page_title="Detecção de Tumor Cerebral",
    page_icon="🧠",
    layout="wide"
)

# =========================
# ESTILO (CSS)
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(135deg, #020617, #0f172a);
    color: white;
}

.main-title {
    text-align: center;
    color: #3B82F6;
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    color: #CBD5E1;
    font-size: 18px;
    margin-bottom: 20px;
}

.main-box {
    background-color: #111827;
    padding: 35px;
    border-radius: 20px;
}

.result-box {
    background: linear-gradient(135deg, #1D4ED8, #2563EB);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
}

.result-title {
    font-size: 30px;
    font-weight: bold;
}

.result-confidence {
    font-size: 20px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## 🧠 Sobre o Projeto")
    st.write("""
    IA para auxiliar na detecção de tumores em exames de MRI.
    """)

    st.markdown("### 📌 Classes")
    st.write("• Glioma")
    st.write("• Meningioma")
    st.write("• Pituitária")
    st.write("• Sem Tumor")

    st.warning("Projeto acadêmico.")

# =========================
# CABEÇALHO
# =========================
st.markdown('<div class="main-title">Detecção de Tumor Cerebral</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Engenharia Biomédica — PUC Campinas</div>',
    unsafe_allow_html=True
)

st.markdown("---")

st.markdown('<div class="main-box">', unsafe_allow_html=True)

# =========================
# DOWNLOAD MODELO
# =========================
MODEL_URL = "https://drive.google.com/uc?id=1pB3o65554q1ntOKH0P4QvB6RswRPiGVG"

if not os.path.exists("model.h5"):
    with st.spinner("Baixando modelo..."):
        gdown.download(MODEL_URL, "model.h5", quiet=False)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

classes = [
    "Glioma",
    "Meningioma",
    "Sem Tumor",
    "Pituitária"
]

# =========================
# TAMANHO INPUT
# =========================
input_shape = model.input_shape
img_height = input_shape[1]
img_width = input_shape[2]

# =========================
# UPLOAD
# =========================
st.markdown("## 📤 Upload")

uploaded_file = st.file_uploader(
    "Envie a imagem",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PROCESSAMENTO
# =========================
if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Imagem enviada", width=300)

        # ✅ PREPROCESSAMENTO CORRETO
        img = image.resize((img_width, img_height))
        img = np.array(img)

        img = preprocess_input(img)  
        img = np.expand_dims(img, axis=0)

        # previsão
        with st.spinner("Analisando..."):
            prediction = model.predict(img)

        predicted_class = classes[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        with col2:
            st.markdown("### 📊 Resultado")

            st.markdown(f"""
            <div class="result-box">
                <div class="result-title">
                    {predicted_class}
                </div>
                <div class="result-confidence">
                    Confiança: <b>{confidence*100:.2f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(confidence)

            # gráfico
            df = pd.DataFrame({
                "Classe": classes,
                "Probabilidade (%)": prediction[0] * 100
            })

            st.bar_chart(df.set_index("Classe"))

    except Exception as e:
        st.error("Erro ao processar imagem.")
        st.write(e)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Projeto acadêmico — Engenharia Biomédica (2026)")
