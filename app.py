import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# --------------------------------------------------
# Streamlit Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    layout="centered"
)

st.title("📈 Tesla Stock Price Prediction using RNN & LSTM")

# --------------------------------------------------
# Load Models & Scaler
# --------------------------------------------------
@st.cache_resource
def load_all_models():
    models = {
        "rnn_1": load_model("models/rnn_1day.h5"),
        "rnn_5": load_model("models/rnn_5day.h5"),
        "rnn_10": load_model("models/rnn_10day.h5"),
        "lstm_1": load_model("models/lstm_1day.h5"),
        "lstm_5": load_model("models/lstm_5day.h5"),
        "lstm_10": load_model("models/lstm_10day.h5"),
    }
    scaler = joblib.load("scaler/minmax_scaler.pkl")
    return models, scaler


models, scaler = load_all_models()

# --------------------------------------------------
# File Upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Tesla Stock Data (CSV / XLS / XLSX)",
    type=["csv", "xls", "xlsx"]
)

# --------------------------------------------------
# File Reading with SAFE Fallback
# --------------------------------------------------
def read_uploaded_file(file):
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)

        elif file.name.endswith(".xlsx"):
            return pd.read_excel(file, engine="openpyxl")

        elif file.name.endswith(".xls"):
            try:
                return pd.read_excel(file, engine="xlrd")
            except Exception:
                # Fallback: file is actually CSV
                file.seek(0)
                st.warning("⚠️ .xls file contained CSV data. Read as CSV.")
                return pd.read_csv(file)

        else:
            st.error("Unsupported file format")
            st.stop()

    except Exception as e:
        st.error(f"File reading failed: {e}")
        st.stop()


# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------
def make_prediction(model, data, window_size):
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])

    X = np.array(X)
    preds = model.predict(X, verbose=0)
    preds = scaler.inverse_transform(preds)
    return preds


# --------------------------------------------------
# Main App Logic
# --------------------------------------------------
if uploaded_file is not None:
    df = read_uploaded_file(uploaded_file)

    st.success("File uploaded successfully ✅")
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Validation
    if "Close" not in df.columns:
        st.error("❌ Dataset must contain a column named 'Close'")
        st.stop()

    close_prices = df["Close"].values.reshape(-1, 1)
    scaled_close = scaler.transform(close_prices)

    window_size = 60

    st.subheader("📊 Model Predictions")

    # ---------------- RNN ----------------
    st.markdown("### 🔹 SimpleRNN Predictions")

    rnn_1_pred = make_prediction(models["rnn_1"], scaled_close, window_size)
    rnn_5_pred = make_prediction(models["rnn_5"], scaled_close, window_size)
    rnn_10_pred = make_prediction(models["rnn_10"], scaled_close, window_size)

    # ---------------- LSTM ----------------
    st.markdown("### 🔹 LSTM Predictions")

    lstm_1_pred = make_prediction(models["lstm_1"], scaled_close, window_size)
    lstm_5_pred = make_prediction(models["lstm_5"], scaled_close, window_size)
    lstm_10_pred = make_prediction(models["lstm_10"], scaled_close, window_size)

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------
    st.subheader("📈 Actual vs Predicted Closing Price")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(close_prices[window_size:], label="Actual Price")

    ax.plot(rnn_1_pred, label="RNN 1-Day")
    ax.plot(lstm_1_pred, label="LSTM 1-Day")

    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # --------------------------------------------------
    # Final Predictions
    # --------------------------------------------------
    st.subheader("📌 Final Day Predictions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**SimpleRNN**")
        st.write("1-Day:", rnn_1_pred[-1][0])
        st.write("5-Day:", rnn_5_pred[-1][0])
        st.write("10-Day:", rnn_10_pred[-1][0])

    with col2:
        st.markdown("**LSTM**")
        st.write("1-Day:", lstm_1_pred[-1][0])
        st.write("5-Day:", lstm_5_pred[-1][0])
        st.write("10-Day:", lstm_10_pred[-1][0])

    st.success("Prediction completed successfully 🚀")
