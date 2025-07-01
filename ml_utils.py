import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

def run_automl_pipeline(df,target_col,test):
    st.markdown("Побудова ML моделі")
    
    reg_models = [DecisionTreeRegressor,Ridge,RandomForestRegressor]
    class_models = [DecisionTreeClassifier,RandomForestClassifier,LogisticRegression]
    
    task_type = "regression"
    if df[target_col].dtype == 'object':
        task_type = "classification"
        model = st.selectbox("Оберіть моделі для класифікації",class_models)
    else:
        model = st.selectbox("Оберіть моделі для регресії",reg_models)
        
    X = df.drop(target_col,axis=1)
    y = df[target_col]
    
    if target_col in test.columns:
        X_test = test.drop(target_col, axis=1)
        y_test = test[target_col]
    else:
        X_test = test
    
    num_cols = X.select_dtypes(include=['int64','float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns
    
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer',SimpleImputer(strategy='mean')),
            ('scaler',StandardScaler())
        ]),num_cols),
        ('cat', Pipeline([
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('one_enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]),cat_cols)
    ])
    
    pipeline = Pipeline([
        ('preprocessor',preprocessor),
        ('model',model)
    ])
    
    pipeline.fit(X,y)
    kf = KFold(n_splits=5,shuffle=True)
    
    scores = cross_val_score(pipeline,X,y,cv=kf).mean()
    st.write(f"📊 Середнє значення cross-validation метрики: {scores:.4f}")