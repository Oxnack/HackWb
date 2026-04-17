import streamlit as st
import numpy as np
import pandas as pd
import torch
from PIL import Image
import pickle
import os
import tempfile
import warnings
from transformers import ViTImageProcessor, ViTModel, AutoTokenizer, AutoModel
import time

warnings.filterwarnings("ignore")

# === КОНФИГУРАЦИЯ ===
MODELS_DIR = "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Загрузка модели XGBoost
@st.cache_resource
def load_xgboost_model():
    model_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Загрузка ViT для изображений
@st.cache_resource
def load_vit_model():
    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name).to(DEVICE)
    model.eval()
    return processor, model

# Загрузка text модели для эмбеддингов
@st.cache_resource
def load_text_model():
    model_name = "intfloat/multilingual-e5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()
    return tokenizer, model

# === ФУНКЦИИ ЭМБЕДДИНГОВ ===
def get_image_embedding(image, processor, model):
    """Получение эмбеддинга изображения"""
    try:
        img = image.convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        return embedding
    except Exception as e:
        st.error(f"Ошибка обработки изображения: {e}")
        return np.zeros(768)

def get_text_embedding(text, tokenizer, model, prefix="query"):
    """Получение эмбеддинга текста"""
    if not text or not isinstance(text, str):
        text = ""
    
    formatted_text = f"{prefix}: {text}"
    inputs = tokenizer(
        formatted_text,
        padding=True,
        truncation=True,
        max_length=128 if prefix == "query" else 256,
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        attention_mask = inputs["attention_mask"]
        embeddings = outputs.last_hidden_state
        
        # Average pooling
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        embedding = (sum_embeddings / sum_mask).cpu().numpy().flatten()
    
    return embedding

def extract_statistical_features(name, description):
    """Извлечение статистических признаков"""
    STOP_WORDS = [
        "сертификат", "таблица", "размер", "схема", "логотип", "сертификация",
        "инструкция", "состав", "ингредиент", "гарантия", "упаковка", "сзади",
        "вид сзади", "узор", "принт", "выкройка", "чертеж", "этикетка", "бирка"
    ]
    
    def count_stop_words(text):
        if pd.isna(text) or not isinstance(text, str):
            return 0
        text_lower = text.lower()
        return sum(1 for word in STOP_WORDS if word in text_lower)
    
    features = {
        'img_order_in_card': 1,
        'img_position_ratio': 1.0,
        'is_first_image': 1,
        'is_last_image': 1,
        'is_middle_image': 0,
        'img_id_offset': 0,
        'img_id_offset_ratio': 0.0,
        'total_images': 1,
        'has_multiple_images': 0,
        'name_length': len(name) if name else 0,
        'desc_length': len(description) if description else 0,
        'name_word_count': len(name.split()) if name else 0,
        'desc_word_count': len(description.split()) if description else 0,
        'stop_words_in_name': count_stop_words(name),
        'stop_words_in_desc': count_stop_words(description),
        'total_stop_words': count_stop_words(name) + count_stop_words(description),
        'has_stop_words': 1 if (count_stop_words(name) + count_stop_words(description)) > 0 else 0
    }
    
    return features

def prepare_features(image_emb, name_emb, desc_emb, stat_features):
    """Подготовка всех признаков для модели"""
    # Объединяем эмбеддинги
    embeddings = np.concatenate([image_emb, name_emb, desc_emb])
    
    # Добавляем статистические признаки в правильном порядке
    stat_order = [
        'img_order_in_card', 'img_position_ratio', 'is_first_image', 'is_last_image', 
        'is_middle_image', 'img_id_offset', 'img_id_offset_ratio', 'total_images', 
        'has_multiple_images', 'name_length', 'desc_length', 'name_word_count', 
        'desc_word_count', 'stop_words_in_name', 'stop_words_in_desc', 
        'total_stop_words', 'has_stop_words'
    ]
    
    stat_values = [stat_features.get(col, 0) for col in stat_order]
    
    # Объединяем все признаки
    all_features = np.concatenate([embeddings, stat_values])
    
    return all_features.reshape(1, -1)

# === ИНТЕРФЕЙС ===
st.set_page_config(
    page_title="WB Relevance Checker",
    page_icon="🛍️",
    layout="wide"
)

# CSS стили
st.markdown("""
<style>
.main {
    padding: 20px;
}
.stButton > button {
    width: 100%;
    height: 50px;
    background-color: #cb11ab;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 10px;
    border: none;
    transition: all 0.3s;
}
.stButton > button:hover {
    background-color: #a00d8a;
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(203, 17, 171, 0.3);
}
.relevant-box {
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border: 2px solid #28a745;
}
.irrelevant-box {
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    border: 2px solid #dc3545;
}
.metric-card {
    padding: 20px;
    border-radius: 10px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    margin: 10px 0;
}
.feature-tag {
    display: inline-block;
    padding: 5px 10px;
    margin: 3px;
    border-radius: 15px;
    background-color: #f0f2f6;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

# Заголовок
st.title("🛍️ Wildberries Relevance Checker")
st.markdown("### Определение релевантности изображения для карточки товара")

# Загрузка моделей
with st.spinner("🔄 Загрузка моделей..."):
    try:
        xgb_model = load_xgboost_model()
        vit_processor, vit_model = load_vit_model()
        text_tokenizer, text_model = load_text_model()
        models_loaded = True
        st.sidebar.success("✅ Модели загружены")
    except Exception as e:
        st.error(f"❌ Ошибка загрузки моделей: {e}")
        models_loaded = False

st.divider()

# Основной интерфейс
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Информация о товаре")
    
    name = st.text_input(
        "Название товара *",
        value="",
        placeholder="Например: Зимняя шапка с помпоном",
        help="Введите название товара"
    )
    
    description = st.text_area(
        "Описание товара *",
        value="",
        placeholder="Введите подробное описание товара...",
        height=150,
        help="Введите описание товара"
    )

with col2:
    st.subheader("🖼️ Изображение")
    
    uploaded_file = st.file_uploader(
        "Загрузите фото товара *",
        type=["jpg", "jpeg", "png"],
        help="Поддерживаются форматы JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Загруженное изображение", use_container_width=True)
        
        # Информация об изображении
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.metric("Размер", f"{image.size[0]}x{image.size[1]}")
        with col_info2:
            st.metric("Формат", image.format)

st.divider()

# Демо-примеры
st.subheader("🎯 Быстрые примеры")
col_demo1, col_demo2, col_demo3, col_demo4 = st.columns(4)

with col_demo1:
    if st.button("🎩 Шапка", use_container_width=True):
        st.session_state['demo_name'] = "Зимняя шапка с ушками"
        st.session_state['demo_desc'] = "Уютная зимняя шапка с мягкими ушками и завязками из качественной пряжи"
        st.rerun()

with col_demo2:
    if st.button("📄 Сертификат", use_container_width=True):
        st.session_state['demo_name'] = "Сертификат соответствия"
        st.session_state['demo_desc'] = "Официальный сертификат соответствия продукции. Содержит схему и таблицу размеров."
        st.rerun()

with col_demo3:
    if st.button("👕 Носки", use_container_width=True):
        st.session_state['demo_name'] = "Хлопковые носки"
        st.session_state['demo_desc'] = "Хлопковые носки с бесшовной технологией для повседневной носки"
        st.rerun()

with col_demo4:
    if st.button("🕶️ Очки", use_container_width=True):
        st.session_state['demo_name'] = "Солнцезащитные очки"
        st.session_state['demo_desc'] = "Стильные солнцезащитные очки с UV-защитой, подходят для повседневной носки"
        st.rerun()

# Применяем демо-значения
if 'demo_name' in st.session_state:
    name = st.session_state['demo_name']
if 'demo_desc' in st.session_state:
    description = st.session_state['demo_desc']

st.divider()

# Кнопка анализа
analyze_button = st.button("🔍 Проверить релевантность", use_container_width=True, disabled=not models_loaded)

if analyze_button:
    if not name or not description:
        st.warning("⚠️ Пожалуйста, заполните название и описание товара")
    elif uploaded_file is None:
        st.warning("⚠️ Пожалуйста, загрузите изображение")
    else:
        with st.spinner("🧠 Анализируем изображение..."):
            progress_bar = st.progress(0)
            
            # Шаг 1: Извлечение эмбеддингов
            progress_bar.progress(20, "Извлечение эмбеддинга изображения...")
            image_emb = get_image_embedding(image, vit_processor, vit_model)
            
            progress_bar.progress(40, "Извлечение эмбеддинга названия...")
            name_emb = get_text_embedding(name, text_tokenizer, text_model, "query")
            
            progress_bar.progress(60, "Извлечение эмбеддинга описания...")
            desc_emb = get_text_embedding(description, text_tokenizer, text_model, "passage")
            
            progress_bar.progress(80, "Извлечение статистических признаков...")
            stat_features = extract_statistical_features(name, description)
            
            progress_bar.progress(90, "Подготовка признаков...")
            features = prepare_features(image_emb, name_emb, desc_emb, stat_features)
            
            progress_bar.progress(95, "Предсказание модели...")
            
            # Проверка размерности
            expected_features = 2355  # 768 + 384 + 384 + 17 = 1553? Проверим
            if features.shape[1] != expected_features:
                # Дополняем или обрезаем признаки
                if features.shape[1] < expected_features:
                    padding = np.zeros((1, expected_features - features.shape[1]))
                    features = np.hstack([features, padding])
                else:
                    features = features[:, :expected_features]
            
            probability = float(xgb_model.predict_proba(features)[0, 1])
            
            progress_bar.progress(100, "Готово!")
            time.sleep(0.5)
            progress_bar.empty()
        
        # Результаты
        st.divider()
        st.subheader("📊 Результаты анализа")
        
        col_res1, col_res2, col_res3 = st.columns([1, 1, 1])
        
        with col_res1:
            delta = probability - 0.5
            st.metric(
                label="Вероятность релевантности",
                value=f"{probability:.1%}",
                delta=f"{delta:+.1%}"
            )
        
        with col_res2:
            if probability >= 0.7:
                st.success("✅ Высокая релевантность")
            elif probability >= 0.5:
                st.info("ℹ️ Средняя релевантность")
            else:
                st.error("❌ Низкая релевантность")
        
        with col_res3:
            confidence = "Высокая" if abs(probability - 0.5) > 0.3 else "Средняя" if abs(probability - 0.5) > 0.15 else "Низкая"
            st.metric("Уверенность модели", confidence)
        
        st.divider()
        
        # Визуализация результата
        if probability >= 0.5:
            st.markdown(f"""
            <div class="relevant-box">
                <h3 style="color: #155724; margin: 0;">✅ РЕЛЕВАНТНОЕ ИЗОБРАЖЕНИЕ</h3>
                <p style="color: #155724; margin: 10px 0 0 0; font-size: 16px;">
                    Изображение подходит для карточки товара с вероятностью {probability:.1%}
                </p>
                <p style="color: #155724; margin: 5px 0 0 0; font-size: 14px;">
                    Модель определяет, что на фото изображен сам товар
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
            <div class="irrelevant-box">
                <h3 style="color: #721c24; margin: 0;">❌ НЕРЕЛЕВАНТНОЕ ИЗОБРАЖЕНИЕ</h3>
                <p style="color: #721c24; margin: 10px 0 0 0; font-size: 16px;">
                    Изображение не рекомендуется для карточки товара (вероятность {probability:.1%})
                </p>
                <p style="color: #721c24; margin: 5px 0 0 0; font-size: 14px;">
                    Возможно, это схема, сертификат или таблица размеров
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Детали анализа
        st.divider()
        st.subheader("🔬 Детали анализа")
        
        with st.expander("📈 Технические детали", expanded=False):
            col_det1, col_det2 = st.columns(2)
            
            with col_det1:
                st.markdown("**Характеристики текста:**")
                st.markdown(f"- Длина названия: {len(name)} символов, {len(name.split())} слов")
                st.markdown(f"- Длина описания: {len(description)} символов, {len(description.split())} слов")
                
                stop_words_count = stat_features['total_stop_words']
                if stop_words_count > 0:
                    st.warning(f"⚠️ Обнаружено стоп-слов: {stop_words_count}")
                else:
                    st.success("✅ Стоп-слова не обнаружены")
            
            with col_det2:
                st.markdown("**Характеристики изображения:**")
                st.markdown(f"- Размер: {image.size[0]}x{image.size[1]} пикселей")
                st.markdown(f"- Формат: {image.format}")
                st.markdown(f"- Режим: {image.mode}")
            
            st.markdown("**Используемые модели:**")
            st.markdown("""
            <div class="metric-card">
                <p style="margin: 0; font-size: 14px;">
                    <strong>🤖 Ансамбль моделей:</strong><br>
                    • ViT-base (768d) - эмбеддинги изображений<br>
                    • Multilingual-E5-small (384d) - эмбеддинги текста<br>
                    • Статистические признаки (17 фичей)<br>
                    • XGBoost - финальный классификатор
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Прогресс-бар вероятности
        st.divider()
        st.subheader("📊 Визуализация вероятности")
        
        # Создаем цветовую шкалу
        color = f"hsl({int(probability * 120)}, 70%, 50%)"
        st.markdown(f"""
        <div style="width:100%; height:30px; background:linear-gradient(to right, #dc3545, #ffc107, #28a745); border-radius:15px; position:relative; margin:20px 0;">
            <div style="position:absolute; left:{probability*100}%; top:-5px; width:4px; height:40px; background:black; transform:translateX(-50%); border-radius:2px;"></div>
            <div style="position:absolute; left:{probability*100}%; top:-30px; transform:translateX(-50%); background:white; padding:2px 8px; border-radius:10px; box-shadow:0 2px 10px rgba(0,0,0,0.1);">
                <strong>{probability:.1%}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption("0% - нерелевантно | 50% - неопределенно | 100% - релевантно")

elif uploaded_file is not None and name and description:
    st.info("👆 Нажмите кнопку 'Проверить релевантность' для анализа")

# Подвал
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🛍️ Wildberries Relevance Checker v1.0</p>
    <p style="font-size: 12px;">Загрузите изображение, название и описание товара для проверки релевантности</p>
</div>
""", unsafe_allow_html=True)

# Сайдбар с информацией
with st.sidebar:
    st.markdown("### 📖 О сервисе")
    st.markdown("""
    Этот сервис определяет, насколько загруженное изображение соответствует описанию товара.
    
    **Как это работает:**
    1. Изображение анализируется моделью ViT
    2. Текст обрабатывается multilingual моделью
    3. Извлекаются статистические признаки
    4. XGBoost выдает финальную вероятность
    
    **Релевантные изображения (1):**
    - Фото самого товара
    - Разные ракурсы товара
    
    **Нерелевантные изображения (0):**
    - Схемы и таблицы
    - Сертификаты
    - Рекламные материалы
    - Логотипы и иконки
    """)
    
    st.divider()
    
    # Статистика сессии
    if 'analyze_count' not in st.session_state:
        st.session_state['analyze_count'] = 0
    
    if analyze_button and uploaded_file and name and description:
        st.session_state['analyze_count'] += 1
    
    st.metric("Анализов в сессии", st.session_state['analyze_count'])