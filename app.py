import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown
import pandas as pd

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
    margin-bottom: 5px;
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
    box-shadow: 0px 0px 25px rgba(0,0,0,0.25);
    margin-top: 20px;
    animation: fadeIn 0.8s ease-in;
}

.result-box {
    background: linear-gradient(135deg, #1D4ED8, #2563EB);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin-top: 20px;
    box-shadow: 0px 0px 20px rgba(37,99,235,0.3);
}

.result-title {
    font-size: 30px;
    font-weight: bold;
}

.result-confidence {
    font-size: 20px;
    margin-top: 10px;
}

.footer {
    text-align: center;
    font-size: 13px;
    margin-top: 40px;
    color: #94A3B8;
}

.sidebar .sidebar-content {
    background-color: #111827;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }

    to {
        opacity: 1;
        transform: translateY(0px);
    }
}

</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:

    st.image("logo_puc.png", width=180)

    st.markdown("## 🧠 Sobre o Projeto")

    st.write("""
    Este projeto utiliza Inteligência Artificial e Deep Learning
    para auxiliar na detecção de tumores cerebrais em imagens
    de ressonância magnética (MRI).
    """)

    st.markdown("### 📌 Classes Detectadas")

    st.write("• Glioma")
    st.write("• Meningioma")
    st.write("• Pituitária")
    st.write("• Sem Tumor")

    st.markdown("---")

    st.warning(
        "Projeto acadêmico. Não substitui avaliação médica profissional."
    )

# =========================
# CABEÇALHO
# =========================
st.markdown('<div class="main-title">🧠 Detecção de Tumor Cerebral</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Engenharia Biomédica — PUC Campinas<br>Emily Ferreira | Isabela Haga</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# =========================
# CONTAINER PRINCIPAL
# =========================
st.markdown('<div class="main-box">', unsafe_allow_html=True)

# =========================
# DOWNLOAD DO MODELO
# =========================
MODEL_URL = "https://drive.google.com/uc?id=1Rhz9jC899ORxUtq4UNu33hh_AB6vtzWC"

if not os.path.exists("model.h5"):
    with st.spinner("Baixando modelo de IA..."):
        gdown.download(MODEL_URL, "model.h5", quiet=False)

# =========================
# CARREGAR MODELO
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
# DESCOBRIR TAMANHO DO MODELO
# =========================
input_shape = model.input_shape
img_height = input_shape[1]
img_width = input_shape[2]

# =========================
# INTRODUÇÃO
# =========================
st.info(
    "Faça upload de uma imagem de ressonância magnética para análise automática pelo modelo de Inteligência Artificial."
)

# =========================
# UPLOAD
# =========================
st.markdown("## 📤 Upload da Imagem")

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

        # garantir RGB
        image = image.convert("RGB")

        col1, col2 = st.columns([1,1])

        with col1:

            st.markdown("### 🖼️ Imagem Carregada")

            st.image(image, width=350)

            st.caption(f"Resolução original: {image.size}")

        # resize
        img = image.resize((img_width, img_height))

        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        with st.spinner("A IA está analisando a ressonância magnética..."):
            prediction = model.predict(img)

        predicted_class = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        with col2:

            st.markdown("### 📊 Resultado")

            st.markdown(f"""
            <div class="result-box">
                <div class="result-title">
                    {predicted_class}
                </div>

                <div class="result-confidence">
                    Confiança: <b>{confidence:.2f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(float(confidence/100))

            st.markdown("### 📈 Probabilidades")

            df = pd.DataFrame({
                "Classe": classes,
                "Probabilidade": prediction[0] * 100
            })

            st.bar_chart(df.set_index("Classe"))

    except Exception as e:

        st.error("Erro ao processar a imagem. Tente outra imagem.")
        st.write(e)

# =========================
# FECHAR CONTAINER
# =========================
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# RODAPÉ
# =========================
st.markdown("---")

st.markdown(
    "<div class='footer'>Projeto acadêmico — Engenharia Biomédica | PUC Campinas (2026)</div>",
    unsafe_allow_html=True
)
