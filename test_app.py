"""
Тестовый скрипт для проверки работоспособности
"""
import pickle
import numpy as np
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTModel, AutoTokenizer, AutoModel

def test_models():
    print("🔍 Тестирование загрузки моделей...")
    
    # Проверка XGBoost
    try:
        with open("models/xgboost_model.pkl", "rb") as f:
            model = pickle.load(f)
        print("✅ XGBoost модель загружена")
    except Exception as e:
        print(f"❌ Ошибка загрузки XGBoost: {e}")
        return False
    
    # Проверка ViT
    try:
        model_name = "google/vit-base-patch16-224"
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTModel.from_pretrained(model_name)
        print("✅ ViT модель загружена")
    except Exception as e:
        print(f"❌ Ошибка загрузки ViT: {e}")
        return False
    
    # Проверка текстовой модели
    try:
        model_name = "intfloat/multilingual-e5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        print("✅ Текстовая модель загружена")
    except Exception as e:
        print(f"❌ Ошибка загрузки текстовой модели: {e}")
        return False
    
    print("✅ Все модели успешно загружены!")
    return True

if __name__ == "__main__":
    test_models()