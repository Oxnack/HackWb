
# WB Relevance Checker

Сервис для определения релевантности изображений для карточек товаров Wildberries.

## 🚀 Быстрый старт

### Локальный запуск

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Поместите обученную модель XGBoost в папку `models/`:
```bash
mkdir -p models
cp /path/to/xgboost_model.pkl models/
```

4. Запустите приложение:
```bash
streamlit run app.py
```

5. Откройте браузер и перейдите по адресу: `http://localhost:8501`

### Запуск в Docker

1. Соберите образ:
```bash
docker build -t wb-relevance .
```

2. Запустите контейнер:
```bash
docker run -p 8501:8501 -v $(pwd)/models:/app/models wb-relevance
```

3. Откройте `http://localhost:8501`

### Использование docker-compose

```bash
docker-compose up -d
```

## 📁 Структура проекта

```
.
├── app.py                 # Основное приложение Streamlit
├── Dockerfile             # Конфигурация Docker
├── docker-compose.yml     # Docker Compose конфигурация
├── requirements.txt       # Python зависимости
├── setup.sh              # Скрипт установки
├── test_app.py           # Тестовый скрипт
├── models/               # Папка с моделями
│   └── xgboost_model.pkl # Обученная модель XGBoost
└── README.md             # Документация
```

## 🔧 Требования

- Python 3.8+
- Docker (опционально)
- Минимум 4GB RAM
- Модель XGBoost (`xgboost_model.pkl`) в папке `models/`

## 🎯 Использование

1. Введите название товара
2. Введите описание товара
3. Загрузите изображение
4. Нажмите "Проверить релевантность"
5. Получите вероятность релевантности

## 🧠 Модели

Сервис использует ансамбль моделей:
- **ViT-base** - для извлечения эмбеддингов изображений (768 измерений)
- **Multilingual-E5-small** - для эмбеддингов текста (384 измерения)
- **XGBoost** - финальный классификатор

## 📊 Метрики

Модель обучена на данных Wildberries и показывает ROC AUC ~0.67 на тестовой выборке.

## 🐛 Устранение неполадок

### Ошибка "Модель не найдена"
Убедитесь, что файл `xgboost_model.pkl` находится в папке `models/`

### Ошибка памяти
Попробуйте запустить с ограничением памяти:
```bash
docker run -p 8501:8501 --memory="4g" -v $(pwd)/models:/app/models wb-relevance
```

### Проблемы с GPU
Для отключения GPU в Docker добавьте переменную окружения:
```bash
docker run -p 8501:8501 -e CUDA_VISIBLE_DEVICES=-1 -v $(pwd)/models:/app/models wb-relevance
```

## 📝 Примечания

- Модель автоматически определяет стоп-слова в описании
- Поддерживаются форматы изображений: JPG, JPEG, PNG
- Рекомендуемый размер изображений: до 1024x1024 пикселей

## 🤝 Контакты

При возникновении вопросов обращайтесь к организаторам хакатона.
```

## 📋 Инструкция по запуску

### Локальный запуск (без Docker)

```bash
# 1. Клонирование репозитория
git clone <your-repo>
cd <your-repo>

# 2. Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# 3. Установка зависимостей
pip install -r requirements.txt

# 4. Создание папки для моделей и копирование модели
mkdir -p models
cp /path/to/your/xgboost_model.pkl models/

# 5. Запуск приложения
streamlit run app.py

# 6. Открыть в браузере
# http://localhost:8501
```

### Запуск через Docker

```bash
# 1. Сборка образа
docker build -t wb-relevance .

# 2. Запуск контейнера
docker run -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  --name wb-checker \
  wb-relevance

# 3. Остановка контейнера
docker stop wb-checker
docker rm wb-checker

# 4. Просмотр логов
docker logs wb-checker
```

### Запуск через docker-compose

```bash
# Запуск
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Остановка
docker-compose down
```

## 🔍 Проверка работоспособности

1. Откройте браузер и перейдите на `http://localhost:8501`
2. Введите тестовые данные:
   - Название: "Зимняя шапка"
   - Описание: "Теплая зимняя шапка из шерсти"
3. Загрузите любое изображение
4. Нажмите "Проверить релевантность"
5. Должен отобразиться результат с вероятностью

## ⚠️ Важные замечания

1. **Модель XGBoost**: Убедитесь, что файл `xgboost_model.pkl` находится в папке `models/` и соответствует ожидаемой размерности признаков (2355).

2. **Первоначальная загрузка**: При первом запуске модели ViT и E5 будут загружены из HuggingFace (требуется интернет).

3. **Ресурсы**: Приложение требует минимум 2GB RAM, рекомендуется 4GB+.

4. **GPU**: По умолчанию в Docker GPU отключен. Для использования GPU нужно установить `nvidia-docker` и убрать `CUDA_VISIBLE_DEVICES=-1`.

5. **Порты**: Приложение использует порт 8501. Убедитесь, что он свободен.