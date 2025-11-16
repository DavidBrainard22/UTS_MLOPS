# app_streamlit_heart.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# CONFIG
st.set_page_config(page_title="Heart Disease Risk System", page_icon="❤️", layout="wide")
DATA_PATH = "heart_cleveland_upload.csv"

# CSS
st.markdown("""
<style>
/* Basic card/header styling */
.main-header {
  font-size: 30px;
  font-weight: 800;
  color: #b91c1c;
  padding: 18px;
  border-radius: 12px;
  text-align: center;
  margin-bottom: 12px;
  background: linear-gradient(90deg,#fff7f7,#fff);
  border: 1px solid #fee2e2;
}
.section-header { font-size:18px; font-weight:700; margin-top:12px; margin-bottom:8px; }
.metric { background:#ffffff; padding:12px; border-radius:8px; border:1px solid #e6e6e6; }
.kpi { font-size:22px; font-weight:700; color:#b91c1c; }
.small { font-size:13px; color:#475569; }
</style>
""", unsafe_allow_html=True)

# DATA LOADING + PREP
@st.cache_data
def load_heart_data(path=DATA_PATH):
    try:
        df = pd.read_csv(path)
        if 'target' not in df.columns and 'num' in df.columns:
            df = df.rename(columns={'num': 'target'})
        if 'target' not in df.columns and 'heartdisease' in df.columns:
            df = df.rename(columns={'heartdisease': 'target'})
        return df
    except Exception:
        cols = [
            "age","sex","cp","trestbps","chol","fbs","restecg","thalach",
            "exang","oldpeak","slope","ca","thal","target"
        ]
        df = pd.read_csv("heart_cleveland_upload.csv", names=cols)
        df = df.replace('?', np.nan)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df

df_raw = load_heart_data()

# quick cleaning/prep
df = df_raw.copy()
if 'target' in df.columns:
    df['target'] = df['target'].apply(lambda x: 1 if x and x > 0 else 0)
else:
    raise ValueError("Dataset must contain 'target' column. Periksa nama kolom file Anda.")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

for c in numeric_cols:
    if df[c].isnull().sum() > 0:
        df[c] = df[c].fillna(df[c].median())

feature_cols = [
    "age","sex","cp","trestbps","chol","fbs","restecg","thalach",
    "exang","oldpeak","slope","ca","thal"
]
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols]
y = df['target']

categorical_feats = [c for c in ["sex","cp","fbs","restecg","exang","slope","ca","thal"] if c in X.columns]
numeric_feats = [c for c in feature_cols if c not in categorical_feats]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_feats),
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_feats)
], remainder='drop')

# TRAIN / MODEL
@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    svc_linear = Pipeline([
        ('pre', preprocessor),
        ('clf', SVC(kernel='linear', C=0.1, probability=True, random_state=42))
    ])
    svc_rbf = Pipeline([
        ('pre', preprocessor),
        ('clf', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
    ])
    lr = Pipeline([
        ('pre', preprocessor),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])

    # Ensemble voting
    ensemble = VotingClassifier(
        estimators=[('rbf', svc_rbf), ('lr', lr), ('linear', svc_linear)],
        voting='soft',
        weights=[2,1,2]
    )

    # Fit ensemble
    ensemble.fit(X_train, y_train)

    X_train_trans = preprocessor.fit_transform(X_train)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_trans, y_train)

    # Test set transform
    X_test_trans = preprocessor.transform(X_test)

    # Metrics
    preds = ensemble.predict(X_test)
    probs = ensemble.predict_proba(X_test)[:,1]

    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds),
        'recall': recall_score(y_test, preds),
        'f1': f1_score(y_test, preds),
        'roc_auc': roc_auc_score(y_test, probs),
        'confusion_matrix': confusion_matrix(y_test, preds),
        'roc_curve': roc_curve(y_test, probs)
    }

    ohe = preprocessor.named_transformers_['cat']
    cat_feature_names = []
    if hasattr(ohe, 'categories_'):
        for i, cat in enumerate(categorical_feats):
            cats = ohe.categories_[i]
            cat_feature_names += [f"{cat}_{val}" for val in cats]
    feature_names_transformed = list(numeric_feats) + cat_feature_names

    return {
        'ensemble': ensemble,
        'rf': rf,
        'preprocessor': preprocessor,
        'X_test': X_test,
        'y_test': y_test,
        'metrics': metrics,
        'feature_names_transformed': feature_names_transformed
    }

models = train_models(X, y)

# UI / DASHBOARD
st.markdown('<div class="main-header">HEART DISEASE RISK PREDICTION</div>', unsafe_allow_html=True)

col1, col2 = st.columns([0.1,1])
    # compute and plot feature importances
    rf = models['rf']
    fname = models['feature_names_transformed']
    importances = rf.feature_importances_
    imp_df = pd.DataFrame({'feature': fname, 'importance': importances}).sort_values('importance', ascending=True)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh(imp_df['feature'], imp_df['importance'])
    ax.set_xlabel('Importance')
    ax.set_title('Feature importance (after preprocessing)')
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.markdown('<div class="section-header">🔎 Input Patient Parameters</div>', unsafe_allow_html=True)
    with st.form('input_form'):
        # Provide inputs for common heart attributes
        age = st.slider('Age', 20, 100, 55)
        sex = st.selectbox('Sex', options=[0,1], format_func=lambda x: 'Male' if x==1 else 'Female')
        cp = st.selectbox('Chest pain type (cp)', options=[0,1,2,3], index=1)
        trestbps = st.number_input('Resting blood pressure (trestbps)', min_value=80, max_value=220, value=130)
        chol = st.number_input('Cholesterol (mg/dl)', min_value=100, max_value=600, value=246)
        fbs_label = st.selectbox('Fasting blood sugar > 120 mg/dl (fbs)', ['Tidak', 'Ya'])
        fbs = 1 if fbs_label == 'Ya' else 0
        restecg = st.selectbox('Resting ECG (restecg)', options=[0,1,2])
        thalach = st.number_input('Max heart rate achieved (thalach)', min_value=60, max_value=230, value=150)
        exang_label = st.selectbox('Exercise induced angina (exang)', ['Tidak', 'Ya'])
        exang = 1 if exang_label == 'Ya' else 0
        oldpeak = st.number_input('ST depression (oldpeak)', min_value=0.0, max_value=10.0, value=1.0, format="%.2f")
        slope = st.selectbox('Slope of peak exercise ST segment (slope)', options=[0,1,2])
        ca = st.selectbox('Number of major vessels colored by fluoroscopy (ca)', options=[0,1,2,3,4])
        thal = st.selectbox('Thalassemia (thal)', options=[0,1,2,3])

        submitted = st.form_submit_button('Run Heart Risk Assessment')

    if submitted:
        input_dict = {}
        for f in feature_cols:
            if f == 'age': input_dict['age'] = age
            elif f == 'sex': input_dict['sex'] = sex
            elif f == 'cp': input_dict['cp'] = cp
            elif f == 'trestbps': input_dict['trestbps'] = trestbps
            elif f == 'chol': input_dict['chol'] = chol
            elif f == 'fbs': input_dict['fbs'] = fbs
            elif f == 'restecg': input_dict['restecg'] = restecg
            elif f == 'thalach': input_dict['thalach'] = thalach
            elif f == 'exang': input_dict['exang'] = exang
            elif f == 'oldpeak': input_dict['oldpeak'] = oldpeak
            elif f == 'slope': input_dict['slope'] = slope
            elif f == 'ca': input_dict['ca'] = ca
            elif f == 'thal': input_dict['thal'] = thal

        input_df = pd.DataFrame([input_dict])[feature_cols]  # preserve order

        # Prediction
        ensemble = models['ensemble']
        proba = ensemble.predict_proba(input_df)[:,1][0]
        pred = 1 if proba >= 0.5 else 0
        proba_pct = proba * 100

        # simple risk categorization
        if proba_pct >= 70:
            risk_label = "HIGH RISK"
            color = "#b91c1c"
            advice = "Segera konsultasikan ke dokter kardiologi. Pemeriksaan lanjutan direkomendasikan."
        elif proba_pct >= 35:
            risk_label = "MEDIUM RISK"
            color = "#ea580c"
            advice = "Pertimbangkan evaluasi medis lebih lanjut dan perubahan gaya hidup."
        else:
            risk_label = "LOW RISK"
            color = "#16a34a"
            advice = "Pertahankan gaya hidup sehat dan pemeriksaan berkala."

        st.markdown(f"<div class='metric' style='border-left:4px solid {color};'><div class='kpi'>{proba_pct:.1f}%</div><div class='small'>{risk_label}</div><div style='margin-top:8px'>{advice}</div></div>", unsafe_allow_html=True)
