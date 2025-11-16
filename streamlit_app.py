import streamlit as st
import pandas as pd
import pickle
import numpy as np
from pathlib import Path

# CONFIG
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered",
)

st.markdown("""
    <h1 style='text-align: center; color: #d11b1b;'>❤️ Heart Disease Prediction</h1>
""", unsafe_allow_html=True)

DATA_PATH = "heart_cleveland_upload.csv"

# LOAD DATASET

@st.cache_data
def load_heart_data():
    df = pd.read_csv(DATA_PATH)

    # pastikan kolom target bernama "target"
    if "condition" in df.columns:
        df = df.rename(columns={"condition": "target"})

    if "num" in df.columns:
        df = df.rename(columns={"num": "target"})

    if "target" not in df.columns:
        raise ValueError("Dataset harus memiliki kolom 'condition' atau 'num' atau 'target'.")

    return df

# LOAD DATA
df = load_heart_data()

# TRAIN MODELS (tidak perlu model.pkl)
models = train_models(X, y)

preprocessor = models["preprocessor"]
model = models["rf"]   # RandomForest (atau ensemble, sesuai kebutuhan)


# FORM INPUT PASIEN
st.markdown("<h2>🧪 Input Data Pasien</h2>", unsafe_allow_html=True)

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, 50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", 50, 250, 120)
        chol = st.number_input("Cholesterol", 50, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

    with col2:
        restecg = st.selectbox("Resting ECG", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate Achieved", 50, 250, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Slope", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
        thal = st.selectbox("Thal", [0, 1, 2, 3])

    submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

# PREDIKSI
if submitted:
    sample = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
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
        "thal": thal,
    }

    X = pd.DataFrame([sample])
    X_transformed = preprocessor.transform(X)

    pred = model.predict(X_transformed)[0]
    prob = model.predict_proba(X_transformed)[0][1]

    st.markdown("---")
    st.markdown("<h2>🔎 Hasil Prediksi</h2>", unsafe_allow_html=True)

    if pred == 1:
        st.error(f"""❤️ **Pasien berisiko penyakit jantung**
        Probabilitas: **{prob:.2f}**""")
    else:
        st.success(f"""💚 **Pasien tidak berisiko penyakit jantung**
        Probabilitas: **{prob:.2f}**""")

