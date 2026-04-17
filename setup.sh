#!/bin/bash

echo "🚀 Установка WB Relevance Checker"

# Создание виртуального окружения
echo "📦 Создание виртуального окружения..."
python3 -m venv venv
source venv/bin/activate

# Установка зависимостей
echo "📥 Установка зависимостей..."
pip install --upgrade pip
pip install -r requirements.txt

# Создание папки для моделей
mkdir -p models

echo "✅ Установка завершена!"
echo ""
echo "📝 Для запуска выполните:"
echo "   source venv/bin/activate"
echo "   streamlit run app.py"
echo ""
echo "🐳 Для запуска в Docker:"
echo "   docker build -t wb-relevance ."
echo "   docker run -p 8501:8501 -v $(pwd)/models:/app/models wb-relevance"