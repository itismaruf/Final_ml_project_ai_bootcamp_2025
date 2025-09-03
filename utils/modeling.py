import joblib
import json
import pandas as pd
import numpy as np

def run_model_pipeline(df=None, model_type="rf"):
    """
    Загружает готовые артефакты (модель, метрики, важности признаков, ROC-кривые и т.д.)
    и возвращает их в формате, который ожидает App.py.
    
    Параметры:
    ----------
    df : pd.DataFrame или None
        Не используется для офлайн-загрузки, но оставлен для совместимости.
    model_type : str
        Тип модели: "rf" (Random Forest) или "stacking".
    """

    # === 1. Определяем имена файлов в зависимости от модели ===
    if model_type == "rf":
        model_path = "rf_model_clean.pkl"
    elif model_type == "stacking":
        model_path = "stacking_model_clean.pkl"
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

    # === 2. Загружаем модель ===
    model = joblib.load(model_path)

    # === 3. Загружаем метрики ===
    with open("metrics_clean.json") as f:
        metrics = json.load(f)

    # === 4. Загружаем важности признаков ===
    feature_importance_df = pd.read_csv("feature_importance.csv")

    # === 5. Матрица ошибок ===
    # Если есть в metrics — берём оттуда, иначе пробуем отдельный файл
    if "Confusion_matrix" in metrics:
        conf_matrix = np.array(metrics["Confusion_matrix"])
    else:
        try:
            with open("conf_matrix.json") as f:
                conf_matrix = np.array(json.load(f))
        except FileNotFoundError:
            conf_matrix = np.array([])

    # === 6. Список признаков ===
    with open("features.json") as f:
        features = json.load(f)

    # === 7. ROC-кривые ===
    roc_curves = metrics.get("ROC_curves", {})

    # === 8. Формируем результат ===
    result = {
        "model": model,
        "metrics": metrics,
        "feature_importance_df": feature_importance_df,
        "conf_matrix": conf_matrix,
        "features": features,
        "classes": model.classes_,
        "roc_curves": roc_curves
    }

    return result