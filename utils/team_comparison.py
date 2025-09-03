# team_comparison.py
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

def render_team_comparison(df: pd.DataFrame):
    """Отрисовывает блок сравнения команд по лигам."""

    st.subheader("⚔️ Сравнение команд по лигам")

    # --- 1. Выбор лиг ---
    leagues = sorted(df["league_name"].dropna().unique())
    col1, col2 = st.columns(2)
    with col1:
        league1 = st.selectbox("Выберите лигу для команды 1", leagues, key="league1")
    with col2:
        league2 = st.selectbox("Выберите лигу для команды 2", leagues, key="league2")

    # --- 2. Выбор команд ---
    teams1 = sorted(df[df["league_name"] == league1]["team_name"].unique()) if league1 else []
    teams2 = sorted(df[df["league_name"] == league2]["team_name"].unique()) if league2 else []

    col3, col4 = st.columns(2)
    with col3:
        team1 = st.selectbox("Выберите команду 1", teams1, key="team1_select")
    with col4:
        team2 = st.selectbox("Выберите команду 2", teams2, key="team2_select")

    # --- 3. Кнопка запуска сравнения ---
    if st.button("Сравнить команды"):
        if not team1 or not team2 or team1 == team2:
            st.warning("Выберите две разные команды для сравнения.")
            return

        # фильтруем данные по выбранным командам
        team1_data = df[df["team_name"] == team1]
        team2_data = df[df["team_name"] == team2]

        features = [
            "goals_team", "xGoals_team", "shots_team", "shotsOnTarget",
            "deep", "ppda", "fouls", "corners", "yellowCards", "redCards"
        ]
        team1_stats = team1_data[features].mean()
        team2_stats = team2_data[features].mean()

        compare_df = pd.DataFrame({
            "Показатель": features,
            team1: team1_stats.values,
            team2: team2_stats.values
        })

        # 🚀 Радарный график
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=team1_stats.values,
            theta=features,
            fill='toself',
            name=team1
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=team2_stats.values,
            theta=features,
            fill='toself',
            name=team2
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            title=f"Сравнение команд: {team1} vs {team2}"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # 📊 Таблица сравнения с подсветкой
        num_cols = compare_df.select_dtypes(include="number").columns
        st.dataframe(compare_df.style.highlight_max(axis=1, subset=num_cols, color="blue"))

        # 📌 Диаграмма распределения стилей команд (улучшенная версия)
        style_counts = (
            df[df["team_name"].isin([team1, team2])]
            .groupby(["team_name", "team_style"])
            .size()
            .reset_index(name="count")
        )

        fig_styles = go.Figure()

        for team in [team1, team2]:
            team_data = style_counts[style_counts["team_name"] == team]
            fig_styles.add_trace(go.Bar(
                x=team_data["team_style"],
                y=team_data["count"],
                name=team,
                text=team_data["count"],
                textposition="auto"
            ))

        fig_styles.update_layout(
            barmode="group",
            title="📌 Частота использования тактических стилей",
            xaxis_title="Тактический стиль",
            yaxis_title="Количество матчей",
            legend_title="Команда",
            template="plotly_white"
        )

        st.plotly_chart(fig_styles, use_container_width=True)
