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

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Heart Disease Risk System", page_icon="❤️", layout="wide")
DATA_PATH = "heart_cleveland_upload.csv"
# -------------------------------------------------------
# CSS Styling
# -------------------------------------------------------
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

# -------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------
@st.cache_data
def load_heart_data(path=DATA_PATH):
    try:
        df = pd.read_csv(path)

        # standardize target naming
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
        df = pd.read_csv(path, names=cols)
        df = df.replace('?', np.nan)
        df = df.apply(pd.to_numeric, errors='coerce')
        return df

df_raw = load_heart_data()

# -------------------------------------------------------
# CLEANING
# -------------------------------------------------------
df = df_raw.copy()

if 'target' not in df.columns:
    raise ValueError("Dataset must contain 'target' column.")

df['target'] = df['target'].apply(lambda x: 1 if x and x > 0 else 0)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for c in numeric_cols:
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

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_feats),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_feats)
])

# -------------------------------------------------------
# TRAIN MODELS
# -------------------------------------------------------
@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

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

    ensemble.fit(X_train, y_train)

    # Train RF on transformed features
    X_train_trans = preprocessor.fit_transform(X_train)

    rf = RandomForestClassifier(n_estimators=200)
    rf.fit(X_train_trans, y_train)

    X_test_trans = preprocessor.transform(X_test)
    preds = ensemble.predict(X_test)
    probs = ensemble.predict_proba(X_test)[:,1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
        "confusion_matrix": confusion_matrix(y_test, preds),
        "roc_curve": roc_curve(y_test, probs)
    }

    # Extract feature names
    ohe = preprocessor.named_transformers_["cat"]
    cat_feature_names = []
    if hasattr(ohe, "categories_"):
        for name, cats in zip(categorical_feats, ohe.categories_):
            cat_feature_names += [f"{name}_{v}" for v in cats]

    feature_names_transformed = numeric_feats + cat_feature_names

    return {
        "ensemble": ensemble,
        "rf": rf,
        "preprocessor": preprocessor,
        "metrics": metrics,
        "feature_names_transformed": feature_names_transformed
    }

models = train_models(X, y)

# -------------------------------------------------------
# UI
# -------------------------------------------------------
st.markdown('<div class="main-header">HEART DISEASE RISK PREDICTION</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

# LEFT COLUMN → FEATURE IMPORTANCE
with col1:
    rf = models["rf"]
    fname = models["feature_names_transformed"]
    importances = rf.feature_importances_

    imp_df = pd.DataFrame({
        "feature": fname,
        "importance": importances
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp_df["feature"], imp_df["importance"])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Preprocessed)")
    plt.tight_layout()
    st.pyplot(fig)

# RIGHT COLUMN → FORM INPUT
with col2:
    st.markdown('<div class="section-header">🔎 Input Patient Parameters</div>', unsafe_allow_html=True)

    with st.form("input_form"):
        age = st.slider("Age", 20, 100, 55)
        sex = st.selectbox("Sex", [0,1], format_func=lambda x: "Male" if x==1 else "Female")
        cp = st.selectbox("Chest pain type (cp)", [0,1,2,3], index=1)
        trestbps = st.number_input("Resting blood pressure", 80, 220, 130)
        chol = st.number_input("Cholesterol", 100, 600, 246)
        fbs = 1 if st.selectbox("Fasting blood sugar", ["Tidak","Ya"]) == "Ya" else 0
        restecg = st.selectbox("Resting ECG", [0,1,2])
        thalach = st.number_input("Max heart rate", 60, 230, 150)
        exang = 1 if st.selectbox("Exercise induced angina", ["Tidak","Ya"]) == "Ya" else 0
        oldpeak = st.number_input("ST depression", 0.0, 10.0, 1.0, format="%.2f")
        slope = st.selectbox("Slope", [0,1,2])
        ca = st.selectbox("Number of major vessels", [0,1,2,3,4])
        thal = st.selectbox("Thalassemia", [0,1,2,3])

        submit = st.form_submit_button("Run Heart Risk Assessment")

    if submit:
        input_dict = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }

        input_df = pd.DataFrame([input_dict])[feature_cols]

        ensemble = models["ensemble"]
        proba = ensemble.predict_proba(input_df)[0][1] * 100
        pred = 1 if proba >= 50 else 0

        # risk levels
        if proba >= 70:
            risk = ("HIGH RISK", "#b91c1c", 
                    "Segera konsultasi dokter kardiologi.")
        elif proba >= 35:
            risk = ("MEDIUM RISK", "#ea580c", 
                    "Perlu pemantauan dan evaluasi lebih lanjut.")
        else:
            risk = ("LOW RISK", "#16a34a",
                    "Risiko rendah, tetap jaga kesehatan.")

        label, color, advice = risk

        st.markdown(
            f"<div class='metric' style='border-left:4px solid {color};'>"
            f"<div class='kpi'>{proba:.1f}%</div>"
            f"<div class='small'>{label}</div>"
            f"<div style='margin-top:8px'>{advice}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
