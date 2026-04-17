"""
Скрипт для быстрого обучения XGBoost без подбора гиперпараметров
Запуск: python train_fast.py
Результат: модель xgboost_model.pkl, метрики и графики
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report, confusion_matrix
import xgboost as xgb
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# === НАСТРОЙКИ ===
RANDOM_STATE = 42
TEST_SIZE = 0.2
INPUT_CSV = "all_train.csv"
MODEL_PATH = "xgboost_model.pkl"

print("="*60)
print("XGBOOST CLASSIFIER - БЫСТРОЕ ОБУЧЕНИЕ")
print("="*60)

# === ЗАГРУЗКА ДАННЫХ ===
print("\n[1/6] Загрузка данных...")
df = pd.read_csv(INPUT_CSV)
print(f"Загружено: {len(df)} строк, {len(df.columns)} колонок")

# === ПАРСИНГ ЭМБЕДДИНГОВ ===
print("\n[2/6] Парсинг эмбеддингов...")

def parse_embedding_col(col):
    """Парсит колонку с эмбеддингами в numpy array"""
    return np.vstack([np.fromstring(x, sep=',') for x in col])

X_img = parse_embedding_col(df['img_emb'])
X_name = parse_embedding_col(df['name_emb'])
X_desc = parse_embedding_col(df['desc_emb'])

print(f"img_emb shape: {X_img.shape}")
print(f"name_emb shape: {X_name.shape}")
print(f"desc_emb shape: {X_desc.shape}")

# Объединяем эмбеддинги
X_embeddings = np.hstack([X_img, X_name, X_desc])

# === ДОБАВЛЯЕМ СТАТИСТИЧЕСКИЕ ПРИЗНАКИ ===
print("\n[3/6] Добавление статистических признаков...")

# Определяем колонки, которые не являются эмбеддингами или метаданными
exclude_cols = ['id', 'card_identifier_id', 'label', 'img_emb', 'name_emb', 'desc_emb']
stat_cols = [col for col in df.columns if col not in exclude_cols]

if stat_cols:
    X_stats = df[stat_cols].values.astype(np.float32)
    print(f"Статистические признаки: {len(stat_cols)}")
    print(f"  Примеры: {', '.join(stat_cols[:5])}...")
    
    # Объединяем все признаки
    X = np.hstack([X_embeddings, X_stats]).astype(np.float32)
else:
    print("Статистические признаки не найдены")
    X = X_embeddings.astype(np.float32)

y = df['label'].values

print(f"\nФинальный размер признаков: {X.shape}")
print(f"Размер целевой переменной: {y.shape}")

# === РАЗБИЕНИЕ НА TRAIN/VAL ===
print("\n[4/6] Разбиение на train/val...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"Train: {X_train.shape[0]} samples")
print(f"Val:   {X_val.shape[0]} samples")
print(f"Баланс классов в train: 0 - {(y_train==0).sum()}, 1 - {(y_train==1).sum()}")
print(f"Баланс классов в val:   0 - {(y_val==0).sum()}, 1 - {(y_val==1).sum()}")

# === ОБУЧЕНИЕ МОДЕЛИ ===
print("\n[5/6] Обучение XGBoost...")

# Дефолтные параметры XGBoost
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)

print("✓ Обучение завершено")

# === ОЦЕНКА МОДЕЛИ ===
print("\n[6/6] Оценка модели...")

# Предсказания
y_pred_proba = model.predict_proba(X_val)[:, 1]
y_pred_class = model.predict(X_val)

# Метрики
roc_auc = roc_auc_score(y_val, y_pred_proba)

print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ НА ВАЛИДАЦИИ")
print("="*60)
print(f"ROC AUC: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_val, y_pred_class, target_names=['Нерелевантные (0)', 'Релевантные (1)']))

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred_class)
print("\nConfusion Matrix:")
print(f"TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
print(f"FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")

# === СОХРАНЕНИЕ МОДЕЛИ ===
print("\n" + "="*60)
print("СОХРАНЕНИЕ МОДЕЛИ")
print("="*60)

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)
print(f"✓ Модель сохранена: {MODEL_PATH}")

# Сохраняем также список признаков для predict
model_info = {
    'model': model,
    'feature_names': list(range(X.shape[1])),
    'embedding_dims': {
        'img': X_img.shape[1],
        'name': X_name.shape[1],
        'desc': X_desc.shape[1]
    },
    'stat_cols': stat_cols
}

with open('model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)
print("✓ Информация о модели сохранена: model_info.pkl")

# === ВИЗУАЛИЗАЦИЯ ===
print("\n" + "="*60)
print("СОЗДАНИЕ ГРАФИКОВ")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. ROC кривая
fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
roc_auc_val = auc(fpr, tpr)

axes[0, 0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC AUC = {roc_auc_val:.4f}')
axes[0, 0].plot([0, 1], [0, 1], 'r--', linewidth=1, label='Случайно')
axes[0, 0].set_xlabel('False Positive Rate', fontsize=11)
axes[0, 0].set_ylabel('True Positive Rate', fontsize=11)
axes[0, 0].set_title('ROC Кривая', fontsize=13, fontweight='bold')
axes[0, 0].legend(loc='lower right')
axes[0, 0].grid(True, alpha=0.3)

# 2. Распределение предсказаний
axes[0, 1].hist(y_pred_proba[y_val == 0], bins=30, alpha=0.7, 
                label='Класс 0', color='red', edgecolor='black')
axes[0, 1].hist(y_pred_proba[y_val == 1], bins=30, alpha=0.7, 
                label='Класс 1', color='green', edgecolor='black')
axes[0, 1].set_xlabel('Предсказанная вероятность', fontsize=11)
axes[0, 1].set_ylabel('Частота', fontsize=11)
axes[0, 1].set_title('Распределение предсказаний', fontsize=13, fontweight='bold')
axes[0, 1].legend(loc='upper center')
axes[0, 1].grid(True, alpha=0.3)

# 3. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'],
            ax=axes[1, 0])
axes[1, 0].set_xlabel('Predicted', fontsize=11)
axes[1, 0].set_ylabel('Actual', fontsize=11)
axes[1, 0].set_title('Confusion Matrix', fontsize=13, fontweight='bold')

# 4. Feature Importance (Top 15)
importance = model.feature_importances_
top_indices = np.argsort(importance)[-15:][::-1]

axes[1, 1].barh(range(15), importance[top_indices])
axes[1, 1].set_yticks(range(15))
axes[1, 1].set_yticklabels([f'Feature {i}' for i in top_indices])
axes[1, 1].set_xlabel('Importance', fontsize=11)
axes[1, 1].set_title('Top 15 Feature Importance', fontsize=13, fontweight='bold')
axes[1, 1].invert_yaxis()
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('xgboost_results.png', dpi=150, bbox_inches='tight')
print("✓ График сохранен: xgboost_results.png")

# Показываем график (если в интерактивном режиме)
try:
    plt.show()
except:
    pass

# === ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА ===
print("\n" + "="*60)
print("ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА")
print("="*60)

# Порог по умолчанию (0.5)
precision_05 = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
recall_05 = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
f1_05 = 2 * precision_05 * recall_05 / (precision_05 + recall_05) if (precision_05 + recall_05) > 0 else 0

print(f"При пороге 0.5:")
print(f"  Precision: {precision_05:.4f}")
print(f"  Recall:    {recall_05:.4f}")
print(f"  F1-score:  {f1_05:.4f}")

# Оптимальный порог по F1
from sklearn.metrics import f1_score
thresholds = np.arange(0.1, 0.9, 0.05)
best_f1 = 0
best_thresh = 0.5

for thresh in thresholds:
    y_pred_thresh = (y_pred_proba >= thresh).astype(int)
    f1 = f1_score(y_val, y_pred_thresh)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"\nОптимальный порог по F1: {best_thresh:.2f} (F1 = {best_f1:.4f})")

# === ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ ===
print("\n" + "="*60)
print("ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ")
print("="*60)
print("""
# Загрузка модели для предсказаний:
import pickle
import numpy as np

with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Для предсказания нужны:
# - img_emb: эмбеддинг изображения
# - name_emb: эмбеддинг названия
# - desc_emb: эмбеддинг описания
# - stats: статистические признаки (если есть)

# Объединяем в один вектор:
features = np.hstack([img_emb, name_emb, desc_emb, stats])
proba = model.predict_proba([features])[0, 1]
print(f"Вероятность релевантности: {proba:.4f}")
""")

print("\n" + "="*60)
print("✓ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
print("="*60)
