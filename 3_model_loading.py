# --------------- Hugging Face ---------------
import streamlit as st
from transformers import AutoModelForImageClassification, AutoImageProcessor

@st.cache_resource
def load_hf_model():
    model_name = "username/my-cool-model" # Ссылка на репозиторий в HF
    model = AutoModelForImageClassification.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor

# --------------- Google disk ---------------
import torch
import streamlit as st
import os
import urllib.request

MODEL_URL = "https://your-direct-link.com/model_id"
MODEL_PATH = "model.pth"

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Загружаю веса... Это займет минуту."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH

@st.cache_resource
def create_model_architecture():
    # Инициализация из кода, что вы будете использовать, из гугла подгружаем только веса
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model


@st.cache_resource
def get_full_model():
    # Скачиваем веса (путь к файлу)
    weights_path = download_model()

    # Создаем архитектуру
    model = create_model_architecture()

    # Загружаем веса в архитектуру
    # map_location='cpu' — критично для работы на серверах без GPU!
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)

    model.eval()  # Переводим в режим предсказания
    return model


# Здесь будет сама модель
my_model = get_full_model()