import streamlit as st
from PIL import Image  # Для картинок
import numpy as np  # Для имитации модельки

st.set_page_config(page_title="Image Classifier", layout="centered")

st.title("🖼 Классификатор изображений")
st.write("Загрузите фото, чтобы для классификации")


@st.cache_resource
def load_model():
    import time
    with st.spinner('Подключаемся к серверу и загружаем веса...'):
        # time.sleep(4)
        return "Модель загружена!"


def predict(image):
    img_resized = image.resize((224, 224))

    confidence = np.random.random()
    label = "Кресло для котика 🐱" if confidence > 0.5 else "Кресло для песика 🐶"

    return label, confidence


model = load_model()

uploaded_file = st.file_uploader("Выберите картинку...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.session_state.im = image
    st.write("Фото загружено")

    # st.image(image, caption='Ваше фото')

    col1, col2 = st.columns(2)

    with col1:
        st.image(st.session_state.im, caption='Ваше фото', use_container_width=True)

    with col2:
        st.write("### Результат анализа")
        if st.button('Классифицировать'):
            with st.spinner('Думаю...'):
                label, score = predict(image)

                st.success(f"Это **{label}**")
                st.metric(label="Уверенность", value=f"{score:.2%}")

                st.progress(score)

                st.balloons()

