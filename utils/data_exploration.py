import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 1. Загрузка данных
def load_data(path: str) -> pd.DataFrame:
    """Загружает CSV и возвращает DataFrame."""
    return pd.read_csv(path)

# 2. Фильтрация данных
def filter_data(df, season=None, league=None, team=None):
    """Фильтрует данные по сезону, лиге и команде."""
    if season:
        df = df[df['season'] == season]
    if league:
        df = df[df['league_name'] == league]
    if team:
        df = df[df['team_name'] == team]
    return df

# 3. Общая информация о датасете
def dataset_overview(df):
    """Возвращает словарь с общей статистикой."""
    return {
        "Матчей": df['gameID'].nunique(),
        "Команд": df['team_name'].nunique(),
        "Сезонов": df['season'].nunique(),
        "Лиг": df['league_name'].nunique()
    }

# 4. Распределение стилей
def plot_style_distribution(df):
    fig = px.histogram(
        df, x='team_style', color='team_style',
        title='Распределение стилей команд',
        text_auto=True
    )
    fig.update_layout(xaxis_title="Стиль", yaxis_title="Количество")
    return fig

# 5. Распределение результатов
def plot_result_distribution(df):
    fig = px.histogram(
        df, x='result', color='result',
        title='Распределение результатов матчей',
        text_auto=True
    )
    fig.update_layout(xaxis_title="Результат", yaxis_title="Количество")
    return fig

# 6. Корреляция признаков
def plot_feature_correlations(df):
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr()
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        title="Тепловая карта корреляций"
    )
    return fig

# 7. Топ команд по метрике
def top_teams_by_metric(df, metric, n=5):
    top_df = (
        df.groupby('team_name')[metric]
        .mean()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    fig = px.bar(
        top_df, x='team_name', y=metric,
        title=f"Топ-{n} команд по {metric}",
        text_auto=".2f"
    )
    return fig