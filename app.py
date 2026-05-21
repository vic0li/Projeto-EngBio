import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(
    page_title="Detecção de Tumor Cerebral",
    page_icon="🧠",
    layout="centered"
)

# =========================
# ESTILO (CSS)
# =========================
st.markdown("""
    <style>
        body {
            background-color: #ffffff;
        }
        .main-title {
            text-align: center;
            color: #0B3D91;
            font-size: 36px;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            color: #333333;
            font-size: 18px;
        }
        .box {
            border-radius: 10px;
            padding: 20px;
            background-color: #F4F6FA;
            border-left: 5px solid #0B3D91;
        }
        .result {
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            color: #0B3D91;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# CABEÇALHO
# =========================
col1, col2 = st.columns([1,3])

with col1:
    # 🔁 Substitua pelo caminho da sua logo
    st.image("logo_puc.png", width=120)

with col2:
    st.markdown('<div class="main-title">Detecção de Tumor Cerebral</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Engenharia Biomédica - PUC Campinas<br>Emily Ferreira | Isabela Haga</div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# =========================
# DOWNLOAD DO MODELO
# =========================
MODEL_URL = "https://drive.google.com/uc?id=1Rhz9jC899ORxUtq4UNu33hh_AB6vtzWC"

if not os.path.exists("model.h5"):
    with st.spinner("Baixando modelo..."):
        gdown.download(MODEL_URL, "model.h5", quiet=False)

# carregar modelo
model = tf.keras.models.load_model("model.h5")

classes = [
    "Glioma",
    "Meningioma",
    "Sem Tumor",
    "Pituitária"
]

# =========================
# UPLOAD
# =========================
st.markdown("### 📤 Upload da Imagem de Ressonância Magnética")

uploaded_file = st.file_uploader(
    "Selecione uma imagem (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PROCESSAMENTO
# =========================
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.markdown("### 🖼️ Imagem carregada")
    st.image(image, use_container_width=True)

    # pré-processamento
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    with st.spinner("Analisando imagem..."):
        prediction = model.predict(img)

    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # =========================
    # RESULTADO
    # =========================
    st.markdown("### 📊 Resultado da Análise")

    st.markdown(f"""
        <div class="box">
            <div class="result">Diagnóstico: {predicted_class}</div>
            <br>
            <div style="text-align:center; font-size:18px;">
                Confiança: <b>{confidence:.2f}%</b>
            </div>
        </div>
    """, unsafe_allow_html=True)

# =========================
# RODAPÉ
# =========================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 14px;'>Projeto acadêmico — Engenharia Biomédica | PUC Campinas (2026)</div>",
    unsafe_allow_html=True
)
