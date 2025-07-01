# app.py

import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ✅ Page Config
st.set_page_config(
    page_title="Alzheimer's GCN Predictor",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ✅ GCN Model Class
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# ✅ Load assets
model = GCN(5, 32, 3)
model.load_state_dict(torch.load("Models/gnn_model_adni.pt", map_location=torch.device('cpu')))
model.eval()

data = torch.load("Models/adni_graph_data.pt", map_location=torch.device('cpu'), weights_only=False)
scaler = joblib.load("Models/scaler_adni.pkl")

# ✅ UI Title
st.markdown("<h1 style='text-align: center;'>🧠 Alzheimer's Disease Prediction Using GCN</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter patient features to predict diagnosis: CN / MCI / AD</p>", unsafe_allow_html=True)

# ✅ Tabs
tab1, tab2 = st.tabs(["📋 Patient Input", "📈 Prediction Summary"])

with tab1:
    mmscore = st.slider("MMSE Score", 0, 30, 25)
    cdrsb = st.slider("CDR-SB", 0.0, 18.0, 0.5, step=0.1)
    educat = st.slider("Years of Education", 0, 30, 16)
    gender = st.radio("Gender", ["Male", "Female"])
    doby = st.number_input("Year of Birth", min_value=1900, max_value=2024, value=1947)

    gender_code = 1 if gender == "Male" else 0

    if st.button("Predict Diagnosis"):
        input_features = [mmscore, cdrsb, educat, gender_code, doby]
        new_scaled = scaler.transform([input_features])
        new_x = torch.tensor(new_scaled, dtype=torch.float)
        x_all = torch.cat([data.x, new_x], dim=0)

        out = model(x_all, data.edge_index)
        pred_logits = out[-1]
        pred_probs = torch.softmax(pred_logits, dim=0).detach().numpy()
        pred_class = pred_probs.argmax()
        label = ["CN", "MCI", "AD"][pred_class]

        st.session_state.predicted = label
        st.session_state.probs = pred_probs
        st.session_state.inputs = input_features

with tab2:
    if "predicted" not in st.session_state:
        st.info("Please make a prediction first.")
    else:
        st.markdown("### 🧠 Prediction Results")
        st.success(f"**Predicted Diagnosis**: {st.session_state.predicted}")

        # Pie Chart
        fig, ax = plt.subplots()
        labels = ["CN", "MCI", "AD"]
        ax.pie(st.session_state.probs, labels=labels, autopct="%1.1f%%", colors=["#66c2a5", "#fc8d62", "#8da0cb"])
        ax.set_title("Prediction Confidence")
        st.pyplot(fig)

        st.markdown("### 📊 Patient Summary")
        mm, cdr, edu, gen, yb = st.session_state.inputs
        gender_text = "Male" if gen == 1 else "Female"
        st.markdown(f"""
        - **MMSE Score**: {mm}  
        - **CDR-SB**: {cdr}  
        - **Education**: {edu} years  
        - **Gender**: {gender_text}  
        - **Year of Birth**: {yb}  
        """)

        st.markdown("⚠️ *This prediction is not a clinical diagnosis. Consult a medical professional for interpretation.*")
