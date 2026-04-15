import streamlit as st

st.title("Виджеты")

if st.button('Кнопка'):
    st.write("Кнопка нажата")

mode = st.radio(
    "Выберите режим анализа:",
    ["Быстрый (CPU)", "Глубокий (GPU)", "Ансамбль моделей"],
    index=0  # Какая кнопка нажата по умолчанию
)
st.write(f"Выбран: {mode}")

text = st.text_input("Введите ваш текст: обратную связь, комментарий", placeholder="Text is here")
if text:
    st.write(f"Спасибо")

checkbox = st.checkbox("Показать вероятности всех классов")
if checkbox:
    st.info("Здесь могли бы быть подробные логи модели...")

model_version = st.selectbox(
    "Какую картинку анализируем?",
    ("Первую", "Вторую", "Минус первую")
)