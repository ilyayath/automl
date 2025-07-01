import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from ydata_profiling import ProfileReport

def basic_eda(df):
    st.write("**Розмірність датасету:**", df.shape)
    st.write("**Описові статистики:**")
    st.dataframe(df.describe())
    st.write("**Кількість пропусків:**")
    st.dataframe(df.isnull().sum())
    st.write("**Числові колонки**", df.select_dtypes(include=["int64", "float64"]).columns)
    st.write("**Числові колонки**", df.select_dtypes(include=["object"]).columns)

def pairplot_top_corr(df, n=4):
    num_df = df.select_dtypes(include='number')
    corr = num_df.corr().abs().unstack().sort_values(ascending=False)
    corr = corr[corr < 1].drop_duplicates()
    top_pairs = corr.head(n).index

    cols = list(set([i for pair in top_pairs for i in pair]))

    st.write("🔗 **Pairplot для топ-зв’язаних змінних**")
    fig = sns.pairplot(df[cols])
    st.pyplot(fig)

def plot_distributions(df):
    num_cols = df.select_dtypes(include='number').columns

    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['figure.figsize'] = (10, 6)

    st.write("**Розподіли числових змінних**")

    for col in num_cols:
        fig, ax = plt.subplots()

        sns.histplot(
            data=df[col].dropna(),
            kde=True,
            color='#1f77b4',
            bins=30,
            stat='density',
            alpha=0.6,
            ax=ax
        )

        kde_line = ax.lines[0]
        kde_line.set_color('#ff7f0e')
        kde_line.set_linewidth(2.5)

        ax.set_title(f'Розподіл змінної: {col}', fontsize=16, weight='bold', pad=15)
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel('Щільність', fontsize=12)

        ax.tick_params(axis='both', which='major', labelsize=10)

        ax.grid(True, linestyle='--', alpha=0.7)

        sns.despine(ax=ax, top=True, right=True)

        st.pyplot(fig)

        plt.close(fig)

def group_stats_with_plots(df, target_col):
    st.markdown(f"### 📊 Групова статистика по цільовій змінній: `{target_col}`")

    if target_col not in df.columns:
        st.warning("⛔ Обрана змінна відсутня в датафреймі.")
        return

    if df[target_col].nunique() > 25:
        st.warning("⚠ Занадто багато унікальних значень у цільовій змінній для групування.")
        return

    # Вибираємо числові змінні, окрім target
    num_cols = df.select_dtypes(include='number').columns
    num_cols = [col for col in num_cols if col != target_col]

    if not num_cols:
        st.warning("⛔ У датафреймі немає числових ознак, окрім цільової.")
        return

    # Агрегована таблиця
    st.subheader("📋 Зведена статистика")
    agg_df = df.groupby(target_col)[num_cols].agg(["mean", "std", "count"])
    st.dataframe(agg_df)

    # Візуалізації
    st.subheader("📈 Візуалізація середніх значень")

    for col in num_cols:
        fig, ax = plt.subplots()
        sns.barplot(data=df, x=target_col, y=col, ci='sd', palette="pastel", ax=ax)

        ax.set_title(f"{col} за групами {target_col}", fontsize=14, weight='bold')
        ax.set_xlabel(str(target_col), fontsize=12)
        ax.set_ylabel(f"Середнє значення {col}", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)

        st.pyplot(fig)
        plt.close(fig)

def plot_categorical_counts(df):
    cat_cols = df.select_dtypes(include='object').columns
    st.write("📊 **Розподіли категоріальних змінних**")
    for col in cat_cols:
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind='bar', ax=ax, color='#1f77b4')
        ax.set_title(f'Частоти: {col}', fontsize=14, weight='bold')
        ax.set_ylabel('Кількість')
        st.pyplot(fig)
        plt.close(fig)

def plot_boxplots(df):
    num_cols = df.select_dtypes(include='number').columns

    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['figure.figsize'] = (10, 6)

    st.write("**Boxplots для виявлення викидів**")

    for col in num_cols:
        fig, ax = plt.subplots()

        sns.boxplot(
            x=df[col].dropna(),
            ax=ax,
            color='#1f77b4',
            boxprops=dict(alpha=0.6),
            whiskerprops=dict(color='#1f77b4', linestyle='--'),
            capprops=dict(color='#1f77b4'),
            medianprops=dict(color='#ff7f0e', linewidth=2.5),
            flierprops=dict(marker='o', markersize=8, markerfacecolor='#ff7f0e', alpha=0.7)
        )

        ax.set_title(f'Ящикова діаграма: {col}', fontsize=16, weight='bold', pad=15)
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel('Значення', fontsize=12)

        ax.tick_params(axis='both', which='major', labelsize=10)

        ax.grid(True, linestyle='--', alpha=0.7)

        sns.despine(ax=ax, top=True, right=True)

        st.pyplot(fig)

        plt.close(fig)

def plot_correlation(df):
    num_cols = df.select_dtypes(include='number')

    corr = num_cols.corr()

    sns.set_style("white")
    plt.rcParams['font.family'] = 'Arial'

    st.write(" **Кореляційна теплокарта**")

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        ax=ax,
        square=True,
        vmin=-1, vmax=1,
        cbar_kws={'label': 'Коефіцієнт кореляції'},
        annot_kws={'size': 10},
        linewidths=0.5,
        linecolor='white'
    )

    ax.set_title('Теплова карта кореляцій', fontsize=16, weight='bold', pad=15)
    ax.tick_params(axis='both', which='major', labelsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

def generate_profile_report(df):
    st.write("🔎 **Генеруємо звіт ydata-profiling... Це може зайняти хвилину.**")

    profile = ProfileReport(df, explorative=True, title="EDA звіт")

    report_file = "eda_report.html"
    profile.to_file(report_file)

    st.success("✅ Звіт збережено у файлі eda_report.html")
    with open(report_file, 'rb') as f:
        st.download_button(
            label="Завантажити звіт",
            data=f,
            file_name=report_file,
            mime='text/html'
        )

    st.write("**Попередній перегляд звіту:**")
    st.components.v1.html(
        open(report_file, 'r', encoding='utf-8').read(),
        height=600, scrolling=True
    )

def detect_outlier(df, const_for_iqr):
    st.write("**Виявлення аномалій в даних за допомогою IQR**")
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    outliers_info = {}

    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - const_for_iqr * IQR
        upper_bound = Q3 + const_for_iqr * IQR
        outliers = ((df[col] > upper_bound) | (df[col] < lower_bound)).sum()
        outliers_info[col] = outliers

    outlier_df = pd.DataFrame.from_dict(outliers_info, orient='index', columns=['Кількість аномалій'])
    st.write("Потенціальні аномалії")
    st.dataframe(outlier_df.sort_values(by="Кількість аномалій", ascending=False))
