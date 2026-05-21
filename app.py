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
    background-color: #FFFFFF;
}

.main-title {
    text-align: center;
    color: #0B3D91;
    font-size: 36px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #444;
    font-size: 18px;
}

.box {
    border-radius: 12px;
    padding: 25px;
    background-color: #F4F7FF;
    border-left: 6px solid #0B3D91;
    margin-top: 15px;
    color: black;
}

.result {
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    color: #0B3D91;
}

.footer {
    text-align: center;
    font-size: 13px;
    margin-top: 40px;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

# =========================
# CABEÇALHO
# =========================
col1, col2 = st.columns([1,3])

with col1:
    st.image("logo_puc.png", width=110)

with col2:
    st.markdown('<div class="main-title">Detecção de Tumor Cerebral</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Engenharia Biomédica — PUC Campinas<br>Emily Ferreira | Isabela Haga</div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# =========================
# DOWNLOAD DO MODELO
# =========================
MODEL_URL = "https://drive.google.com/uc?id=1pB3o65554q1ntOKH0P4QvB6RswRPiGVG"

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
# DESCOBRIR TAMANHO DO MODELO (AUTOMÁTICO)
# =========================
input_shape = model.input_shape
img_height = input_shape[1]
img_width = input_shape[2]

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
    try:
        image = Image.open(uploaded_file)

        # ✅ garantir RGB
        image = image.convert("RGB")

        st.markdown("### 🖼️ Imagem carregada")
        st.image(image, use_container_width=True)

        # ✅ mostrar info da imagem
        st.caption(f"Resolução original: {image.size}")

        # ✅ resize automático baseado no modelo
        img = image.resize((img_width, img_height))

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

    except Exception as e:
        st.error("Erro ao processar a imagem. Tente outra imagem.")
        st.write(e)

# =========================
# RODAPÉ
# =========================
st.markdown("---")
st.markdown(
    "<div class='footer'>Projeto acadêmico — Engenharia Biomédica | PUC Campinas (2026)</div>",
    unsafe_allow_html=True
)
