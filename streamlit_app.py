# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

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

warnings.filterwarnings("ignore")

# -----------------------
# CONFIG / PATH
# -----------------------
st.set_page_config(page_title="Heart Disease Risk System", page_icon="❤", layout="wide")
DATA_PATH = DATA_PATH = "heart_cleveland_upload.csv"
# -----------------------
# CSS Styling
# -----------------------
st.markdown("""
<style>
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

# -----------------------
# DATA LOADING
# -----------------------
@st.cache_data
def load_heart_data(path=DATA_PATH):
    try:
        df = pd.read_csv(path)
           # Rename kolom target
        if "condition" in df.columns:
            df = df.rename(columns={"condition": "target"})
        elif "num" in df.columns:
            df = df.rename(columns={"num": "target"})
        elif "target" not in df.columns:
            raise ValueError("Dataset tidak memiliki kolom target ('condition', 'num', atau 'target').")
        return df
    
    except Exception:
        cols = [
            "age","sex","cp","trestbps","chol","fbs","restecg","thalach",
            "exang","oldpeak","slope","ca","thal","target"
        ]
        df = pd.read_csv(path, names=cols)
        df = df.replace('?', np.nan)
        df = df.apply(pd.to_numeric, errors='coerce')
        return df

df_raw = load_heart_data()

# -----------------------
# BASIC CLEANING + PREP
# -----------------------
df = df_raw.copy()

if 'target' not in df.columns:
    raise ValueError("Dataset must contain a 'target' column (atau 'num').")

# Ensure target is binary 0/1
# Many Heart datasets use 0 = no disease, 1..4 = disease
df['target'] = df['target'].apply(lambda x: 1 if (pd.notna(x) and x > 0) else 0)

# fill numeric missing values with median
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for c in numeric_cols:
    df[c] = df[c].fillna(df[c].median())

# candidate features (keamanan bila dataset berbeda)
feature_cols = [
    "age","sex","cp","trestbps","chol","fbs","restecg","thalach",
    "exang","oldpeak","slope","ca","thal","condition"
]
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols]
y = df['target']

# determine categorical and numeric features
categorical_feats = [c for c in ["sex","cp","fbs","restecg","exang","slope","ca","thal","condition"] if c in X.columns]
numeric_feats = [c for c in feature_cols if c not in categorical_feats]

# If no numeric_feats (unlikely), avoid empty transformer
if len(numeric_feats) == 0:
    numeric_feats = []

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_feats) if len(numeric_feats) > 0 else ("num", "passthrough", numeric_feats),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_feats)
])

# -----------------------
# TRAIN MODELS
# -----------------------
@st.cache_resource
def train_models(X, y):
    # train-test split (stratify if possible)
    strat = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=strat
    )

    # pipelines with shared preprocessor object (safe)
    svc_linear = Pipeline([
        ("pre", preprocessor),
        ("clf", SVC(kernel="linear", C=0.1, probability=True))
    ])

    svc_rbf = Pipeline([
        ("pre", preprocessor),
        ("clf", SVC(kernel="rbf", C=1.0, probability=True))
    ])

    lr = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    ensemble = VotingClassifier(
        estimators=[("rbf", svc_rbf), ("lr", lr), ("linear", svc_linear)],
        voting="soft",
        weights=[2,1,2]
    )

    # Fit ensemble (this will fit preprocessor as part of pipelines)
    ensemble.fit(X_train, y_train)

    # For a separate RandomForest we need transformed numeric + ohe features:
    # Ensure preprocessor is fitted (it was by pipelines.fit above). Use transform.
    try:
        X_train_trans = preprocessor.transform(X_train)
    except Exception:
        # if preprocessor not fitted for some reason, fit it explicitly
        X_train_trans = preprocessor.fit_transform(X_train)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_trans, y_train)

    # Evaluate ensemble on test set safely (handle cases with single-class y_test)
    try:
        X_test_trans = preprocessor.transform(X_test)
    except Exception:
        X_test_trans = preprocessor.fit_transform(X_test)

    preds = ensemble.predict(X_test)
    try:
        probs = ensemble.predict_proba(X_test)[:,1]
    except Exception:
        # fallback: if predict_proba not available or single-class, create zeros
        probs = np.zeros(len(preds))

    # compute metrics robustly
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_test, preds)
    metrics["precision"] = precision_score(y_test, preds, zero_division=0)
    metrics["recall"] = recall_score(y_test, preds, zero_division=0)
    metrics["f1"] = f1_score(y_test, preds, zero_division=0)
    metrics["confusion_matrix"] = confusion_matrix(y_test, preds).tolist()
    try:
        metrics["roc_auc"] = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else None
        metrics["roc_curve"] = roc_curve(y_test, probs) if len(np.unique(y_test)) > 1 else (None, None, None)
    except Exception:
        metrics["roc_auc"] = None
        metrics["roc_curve"] = (None, None, None)

    # Extract transformed feature names (numeric + OHE)
    cat_feature_names = []
    try:
        ohe = preprocessor.named_transformers_["cat"]
        if hasattr(ohe, "categories_"):
            for name, cats in zip(categorical_feats, ohe.categories_):
                cat_feature_names += [f"{name}_{v}" for v in cats]
    except Exception:
        # if not available, leave empty
        cat_feature_names = []

    feature_names_transformed = numeric_feats + cat_feature_names

    return {
        "ensemble": ensemble,
        "rf": rf,
        "preprocessor": preprocessor,
        "metrics": metrics,
        "feature_names_transformed": feature_names_transformed,
        "X_train_cols": X.columns.tolist()
    }

models = train_models(X, y)

# -----------------------
# UI
# -----------------------
st.markdown('<div class="main-header">HEART DISEASE RISK PREDICTION</div>', unsafe_allow_html=True)

st.markdown('<div class="section-header">✏ Input Patient Parameters</div>', unsafe_allow_html=True)

cp_desc = {
    0: "Typical angina",
    1: "Atypical angina",
    2: "Non-anginal pain",
    3: "Asymptomatic"
}
restecg_desc = {
    0: "Normal",
    1: "ST-T wave abnormality",
    2: "Left ventricular hypertrophy"
}
slope_desc = {
    0: "Upsloping",
    1: "Flat",
    2: "Downsloping"
}
thal_desc = {
    0: "Unknown",
    1: "Normal",
    2: "Fixed defect",
    3: "Reversible defect"
}

with st.form("input_form"):
    age = st.slider("Age", 20, 100, 55)
    sex = st.selectbox("Sex", ["Female", "Male"])
    sex_val = 1 if sex == "Male" else 0

    cp = st.selectbox("Chest pain type (cp)", [0,1,2,3], index=1, format_func=lambda x: f"{x} — {cp_desc.get(x,'')}")
    trestbps = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 130)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 246)
    fbs_label = st.selectbox("Fasting blood sugar > 120 mg/dl?", ["Tidak", "Ya"])
    fbs = 1 if fbs_label == "Ya" else 0

    restecg = st.selectbox("Resting ECG", [0,1,2], format_func=lambda x: f"{x} — {restecg_desc.get(x,'')}")
    thalach = st.number_input("Max heart rate achieved", 60, 230, 150)
    exang_label = st.selectbox("Exercise induced angina?", ["Tidak", "Ya"])
    exang = 1 if exang_label == "Ya" else 0

    oldpeak = st.number_input("ST depression induced by exercise relative to rest", 0.0, 10.0, 1.0, format="%.2f")
    slope = st.selectbox("Slope of the peak exercise ST segment", [0,1,2], format_func=lambda x: f"{x} — {slope_desc.get(x,'')}")
    ca = st.selectbox("Number of major vessels colored by fluoroscopy (0-4)", [0,1,2,3,4])
    thal = st.selectbox("Thalassemia", [0,1,2,3], format_func=lambda x: f"{x} — {thal_desc.get(x,'')}")
    condition_label = st.selectbox('Condition (apakah ada kondisi lain?)', ['Tidak', 'Ya']) if 'condition' in X.columns else None
    condition = 1 if condition_label == 'Ya' else 0 if condition_label is not None else None

    submit = st.form_submit_button("Run Heart Risk Assessment")

if submit:
    input_dict = {
        "age": age,
        "sex": sex_val,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }
    if 'condition' in X.columns:
        input_dict['condition'] = condition

    # only keep columns that the model expects
    model_cols = models["X_train_cols"]
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[[c for c in model_cols if c in input_df.columns]]

    ensemble = models["ensemble"]
    try:
        proba = ensemble.predict_proba(input_df)[0][1] * 100
    except Exception:
        # if predict_proba not available or error, fallback to predict and low confidence
        pred_label = ensemble.predict(input_df)[0]
        proba = 100.0 if pred_label == 1 else 0.0

    pred = 1 if proba >= 50 else 0

    # risk levels
    if proba >= 70:
        risk = ("TINGGI (HIGH RISK)", "#b91c1c", "Segera konsultasi dokter kardiologi.")
    elif proba >= 35:
        risk = ("SEDANG (MEDIUM RISK)", "#ea580c", "Perlu pemantauan dan evaluasi lebih lanjut.")
    else:
        risk = ("RENDAH (LOW RISK)", "#16a34a", "Risiko rendah, tetap jaga kesehatan dan lakukan pemeriksaan berkala.")

    label, color, advice = risk

    st.markdown(
        f"<div class='metric' style='border-left:4px solid {color};'>"
        f"<div class='kpi'>{proba:.1f}%</div>"
        f"<div class='small'>{label}</div>"
        f"<div style='margin-top:8px'>{advice}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    if pred == 1:
        st.warning("Hasil: *Terindikasi kemungkinan penyakit jantung.* Hasil ini bukan diagnosis definitif — segera konsultasi ke profesional medis untuk pemeriksaan lanjutan.")
    else:
        st.success("Hasil: *Kemungkinan rendah tanda penyakit jantung.* Jaga pola hidup sehat dan lakukan pemeriksaan rutin sesuai anjuran dokter.")
