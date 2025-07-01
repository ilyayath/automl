import streamlit as st
import pandas as pd
from eda_utils import basic_eda, plot_distributions, plot_boxplots, plot_correlation, generate_profile_report, detect_outlier, plot_categorical_counts, group_stats_with_plots, pairplot_top_corr
from ml_utils import run_automl_pipeline

st.set_page_config(page_title="Auto ML Dashboard", layout="wide")

st.sidebar.title("Інструкція")
st.sidebar.info("""
1. Завантаж CSV-файл
2. Перейди на вкладку EDA — переглянь статистику
3. Перейди на вкладку Модель — побудуй ML модель
4. Переглянь висновки та експортуй результати
""")

st.title("AutoML Dashboard для CSV")

uploaded_file = st.file_uploader("Завантаж CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Файл завантажено")
    
    tab1, tab2, tab3 = st.tabs(["EDA", "Модель", "Висновки"])
    
    with tab1:
        basic_eda(df)
        target_col = st.selectbox("Вибери цільову змінну (target)", df.columns, key="eda_target")
        if target_col:
            group_stats_with_plots(df, target_col)
        pairplot_top_corr(df, n=4)
        plot_distributions(df)
        plot_categorical_counts(df)
        plot_boxplots(df)
        plot_correlation(df)
        const_for_iqr = st.number_input(
            label="Введіть константу для міжквантильного розмаху (рекомендовано 1.5)",
            min_value=0.0,
            max_value=3.0,
            value=1.5,
            step=0.1
        )  
        detect_outlier(df, const_for_iqr) 
        if st.button("🔎 Згенерувати звіт EDA"):
            generate_profile_report(df)
            
    with tab2:
        st.write("Завантаж дані на яких модель буде тестуватись")
        upl_test = st.file_uploader("Upload your test CSV file", type=["csv"], key="test_file")
        if upl_test is not None:
            test = pd.read_csv(upl_test)
            target_col = st.selectbox("Вибери цільову змінну (target)", df.columns, key="ml_target")
            run_automl_pipeline(df, target_col, test)
        else:
            st.warning("Please upload a CSV file to continue.")
        
    with tab3:
        st.markdown("📈 **Висновки та експорти** — фінальний крок.")

else:
    st.warning("📁 Будь ласка, завантаж CSV-файл для початку.")
