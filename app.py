import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from utils.data_exploration import (
    load_data, filter_data, dataset_overview,
    plot_style_distribution, plot_result_distribution,
    plot_feature_correlations, top_teams_by_metric
)

from utils.modeling import run_model_pipeline
from utils.team_comparison import render_team_comparison

# Загружаем данные один раз
df = load_data("top_5_soccer_leagues_15_20.csv")
df['win'] = (df['result'] == 'W').astype(int)

# Переключатель страниц в сайдбаре
page = st.sidebar.radio("Выберите раздел", ["📊 Знакомство с данными", "🤖 Моделирование", "⚔️ Сравнение команд"])

# ------------------ СТРАНИЦА 1 ------------------
if page == "📊 Знакомство с данными":
    st.header("📊 Знакомство с данными")

    # Фильтры
    season = st.sidebar.selectbox("Сезон", [None] + sorted(df['season'].unique()))
    league = st.sidebar.selectbox("Лига", [None] + sorted(df['league_name'].unique()))

    filtered_df = filter_data(df, season, league)

    # Общая статистика
    stats = dataset_overview(filtered_df)
    st.write("### Общая информация", stats)

    # Графики
    st.plotly_chart(plot_style_distribution(filtered_df))
    st.plotly_chart(plot_result_distribution(filtered_df))

    with st.expander("📈 Показать матрицу корреляций"):
        st.plotly_chart(plot_feature_correlations(filtered_df))

    # Выбор метрики для топа команд
    numeric_columns = [
        'xGoals_team', 'shots_team', 'shotsOnTarget', 'deep', 'ppda',
        'fouls', 'corners', 'yellowCards', 'redCards'
    ]
    selected_metric = st.selectbox("Выберите метрику для топ-5 команд", numeric_columns)
    st.plotly_chart(top_teams_by_metric(filtered_df, selected_metric, n=5))


# ------------------ СТРАНИЦА 2 ------------------
if page == "🤖 Моделирование":
    st.header("🤖 Моделирование")

    # --- Загружаем всё через общий пайплайн ---
    artifacts = run_model_pipeline(None)  # df не нужен, т.к. офлайн-загрузка

    model = artifacts["model"]
    metrics = artifacts["metrics"]
    feature_importance_df = artifacts["feature_importance_df"]
    conf_matrix = artifacts["conf_matrix"]
    features = artifacts["features"]
    classes = artifacts["classes"]
    roc_curves = artifacts["roc_curves"]

    # -----------------------
    # 1. Описание подхода
    # -----------------------
    st.subheader("📌 Использованный метод")
    st.info("Мы применили - 🌲 Random Forest\n\n")
    st.markdown("---")

    # -----------------------
    # 2. Метрики качества
    # -----------------------
    st.subheader("📊 Метрики качества")
    numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    metrics_df = pd.DataFrame(numeric_metrics.items(), columns=["Метрика", "Значение"])

    fig_metrics = px.bar(
        metrics_df, x="Метрика", y="Значение",
        color="Значение", text="Значение",
        color_continuous_scale="Viridis"
    )
    fig_metrics.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    st.plotly_chart(fig_metrics, use_container_width=True)
    st.markdown("---")

    # -----------------------
    # 3. Confusion Matrix
    # -----------------------
    st.subheader("🔍 Confusion Matrix")
    if conf_matrix is not None and conf_matrix.size > 0:
        fig_cm = px.imshow(
            conf_matrix, text_auto=True,
            x=classes, y=classes,
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(
            title="Матрица ошибок",
            xaxis_title="Предсказано", yaxis_title="Истинное значение"
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.warning("Матрица ошибок недоступна.")
    st.markdown("---")

    # -----------------------
    # 4. Важность признаков
    # -----------------------
    st.subheader("🌟 Важность признаков")
    fig_imp = px.bar(
        feature_importance_df,
        x="Важность", y="Признак",
        orientation="h",
        color="Важность",
        color_continuous_scale="Sunset"
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    st.markdown("---")

    # -----------------------
    # 5. ROC-кривые
    # -----------------------
    st.subheader("📈 ROC-кривые (One-vs-Rest)")
    if roc_curves:
        import plotly.graph_objects as go
        fig_roc = go.Figure()
        for cls, data in roc_curves.items():
            fig_roc.add_trace(go.Scatter(
                x=data["fpr"], y=data["tpr"],
                mode="lines",
                name=f"Класс {cls}"
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Случайное угадывание",
            line=dict(dash="dash", color="gray")
        ))
        fig_roc.update_layout(
            title="ROC-кривые по классам",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend_title="Классы",
            template="plotly_white"
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    else:
        st.warning("ROC-кривые недоступны.")
    st.markdown("---")

# ------------------ СТРАНИЦА 1 ------------------
if page == "⚔️ Сравнение команд":
    render_team_comparison(df)