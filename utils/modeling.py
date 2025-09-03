import joblib
import json
import pandas as pd
import numpy as np

def run_model_pipeline(df):
    """
    Вместо обучения модели загружает готовые артефакты,
    сохранённые офлайн, и возвращает их в формате,
    который ожидает App.py.
    """

    # === 1. Загружаем модель и артефакты ===
    model = joblib.load("stacking_model_clean.pkl")

    with open("metrics_clean.json") as f:
        metrics = json.load(f)

    feature_importance_df = pd.read_csv("feature_importance.csv")

    with open("conf_matrix.json") as f:
        conf_matrix = np.array(json.load(f))

    with open("features.json") as f:
        features = json.load(f)

    # === 2. Формируем структуру, как раньше ===
    result = {
        "metrics": metrics,
        "feature_importance_df": feature_importance_df,
        "y_test": None,          # офлайн не сохраняем, можно оставить None
        "y_pred": None,          # офлайн не сохраняем, можно оставить None
        "y_proba": None,         # офлайн не сохраняем, можно оставить None
        "classes": list(model.classes_),
        "roc_curves": None,      # можно добавить, если сохраним офлайн
        "conf_matrix": conf_matrix,
        "features": features,
        "model": model,
        "scaler": None,          # scaler внутри пайплайна
        "best_thresholds": None  # если не используем — None
    }

    return result


def split_features_by_type(df, feature_cols):
    """
    Делит признаки на числовые и категориальные.
    """
    if df is None or df.empty:
        return [], []

    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    return num_cols, cat_cols


def validate_and_prepare_single_input(df, feature_cols, user_input):
    """
    Проверяет значения и формирует DataFrame из одного объекта.
    """
    errors = {}
    row = {}

    for feat in feature_cols:
        if feat not in user_input:
            errors[feat] = "Поле отсутствует во вводе."
            continue

        val = user_input[feat]
        series = df[feat]

        if pd.api.types.is_numeric_dtype(series):
            try:
                num_val = float(val)
                if not np.isfinite(num_val):
                    errors[feat] = "Некорректное число."
                else:
                    row[feat] = num_val
            except Exception:
                errors[feat] = f"Ожидалось число, получено: {val}"
        else:
            allowed = pd.Series(series.dropna().unique()).astype(str).tolist()
            sval = str(val).strip()
            if sval not in allowed:
                errors[feat] = f"Недопустимая категория: {sval}"
            else:
                row[feat] = sval

    if errors:
        return None, errors

    return pd.DataFrame([row], columns=feature_cols), {}


def predict_with_explanation(model, scaler, features, X_input_df, threshold=0.5, top_k=3):
    """
    Предсказание для одного объекта с объяснением вклада признаков.
    """
    # Масштабирование не нужно, если scaler внутри пайплайна
    y_proba_all = model.predict_proba(X_input_df[features])
    pred_class_int = int(np.argmax(y_proba_all[0]))
    proba = float(np.max(y_proba_all[0]))
    pred_class = model.classes_[pred_class_int]

    # Упрощённое объяснение: топ признаков по абсолютному значению
    contribs = dict(zip(features, X_input_df.iloc[0].values))
    top = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    influence_text = ", ".join([f"{feat} ({val:.2f})" for feat, val in top])

    return {
        "pred_class": pred_class,
        "pred_class_int": pred_class_int,
        "proba": proba,
        "explanation": f"Решение объясняется вкладом признаков: {influence_text}.",
        "top_contributions": top,
        "all_proba": y_proba_all[0].tolist()
    }