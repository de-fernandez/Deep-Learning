import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import joblib

# ============================================================
# CONFIG
# ============================================================

NUM_FEATURES = 51
SEQ_LEN = 20
NUM_CLASSES = 2
HIDDEN_SIZE = 64
CLASS_LABELS = ['Attack', 'Normal']

st.set_page_config(
    page_title="CNN-BiGRU Intrusion Detection System",
    layout="wide"
)

st._config.set_option('server.maxUploadSize', 1024)

# ============================================================
# MODEL
# ============================================================

class CNN_BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.conv = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bigru = nn.GRU(
            input_size=64,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        _, h = self.bigru(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        h = self.dropout(h)
        return self.fc(h)

# ============================================================
# LOAD MODEL + SCALER
# ============================================================

@st.cache_resource
def load_model():
    model = CNN_BiGRU(NUM_FEATURES, HIDDEN_SIZE, NUM_CLASSES)
    model.load_state_dict(torch.load("cnn_bigru_state_dict.pth", map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model()
scaler = load_scaler()

st.success("Model & Scaler loaded successfully!")

# ============================================================
# PREDICTION FUNCTION (FIXED)
# ============================================================

def predict_sequence(data_array):
    """
    data_array shape: (N, 51)
    """
    sequences = []

    for i in range(len(data_array) - SEQ_LEN):
        seq = data_array[i:i+SEQ_LEN]
        sequences.append(seq)

    sequences = np.array(sequences)

    # SCALE (VERY IMPORTANT FIX)
    n_features = sequences.shape[2]
    sequences = scaler.transform(
        sequences.reshape(-1, n_features)
    ).reshape(sequences.shape)

    tensor = torch.tensor(sequences, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).numpy()
        preds = np.argmax(probs, axis=1)

    return preds, probs

# ============================================================
# UI
# ============================================================

st.title("CNN-BiGRU Intrusion Detection System")
st.write("Upload water treatment sensor data for anomaly detection")
st.markdown("---")

mode = st.radio(
    "Choose mode:",
    ["Batch Detection", "Live Simulation"],
    horizontal=True
)

uploaded_file = st.file_uploader(
    "Upload CSV (51 sensor columns only, max 1GB)",
    type=["csv"]
)

if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)

df.columns = df.columns.str.strip()

st.write("### Preview")
st.dataframe(df.head())

st.write("Shape:", df.shape)

# FIX: enforce correct feature count
if df.shape[1] != NUM_FEATURES:
    st.error(f"Expected {NUM_FEATURES} columns, got {df.shape[1]}")
    st.stop()

data = df.values.astype(np.float32)

# ============================================================
# BATCH MODE
# ============================================================

if mode == "Batch Detection":

    with st.spinner("Running model..."):
        preds, probs = predict_sequence(data)

    labels = [CLASS_LABELS[p] for p in preds]

    results = pd.DataFrame({
        "Prediction": labels,
        "Attack %": (probs[:, 0] * 100).round(2),
        "Normal %": (probs[:, 1] * 100).round(2),
    })

    st.write("### Results")
    st.dataframe(results)

    st.metric("Total Predictions", len(labels))
    st.metric("Attacks", labels.count("Attack"))
    st.metric("Normals", labels.count("Normal"))

# ============================================================
# LIVE MODE
# ============================================================

elif mode == "Live Simulation":

    st.write("Simulating real-time sensor stream...")

    speed = st.slider("Speed (sec)", 0.05, 1.0, 0.1)

    max_rows = st.number_input(
        "Rows to simulate",
        min_value=50,
        max_value=len(data),
        value=min(200, len(data))
    )

    if st.button("Start"):

        placeholder = st.empty()
        table = st.empty()

        history = []

        for i in range(SEQ_LEN, min(max_rows, len(data))):

            window = data[i-SEQ_LEN:i]
            window = window.reshape(1, SEQ_LEN, NUM_FEATURES)

            window = scaler.transform(
                window.reshape(-1, NUM_FEATURES)
            ).reshape(window.shape)

            tensor = torch.tensor(window, dtype=torch.float32)

            with torch.no_grad():
                out = model(tensor)
                prob = torch.softmax(out, dim=1).numpy()[0]
                pred = np.argmax(prob)

            label = CLASS_LABELS[pred]

            history.append({
                "Row": i,
                "Prediction": label,
                "Attack %": round(prob[0] * 100, 2),
                "Normal %": round(prob[1] * 100, 2)
            })

            df_live = pd.DataFrame(history)

            color = "🔴" if label == "Attack" else "🟢"

            placeholder.markdown(
                f"Row {i} → {color} {label}"
            )

            table.dataframe(df_live.tail(20), use_container_width=True)

            time.sleep(speed)

        st.success("Simulation complete")