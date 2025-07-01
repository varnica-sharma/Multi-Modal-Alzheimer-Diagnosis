# app.py
import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import joblib
import numpy as np
import matplotlib.pyplot as plt

# 🧠 GCN model class
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

# 🎨 Page config
st.set_page_config(page_title="Alzheimer's GCN Predictor", page_icon="🧠", layout="centered")

st.title("🧠 Alzheimer's Disease Prediction Using GCN")
st.markdown("### 📋 Patient Evaluation Form")

# 🔄 Load model and data
model = GCN(5, 32, 3)
model.load_state_dict(torch.load("Models/gnn_model_adni.pt", map_location=torch.device('cpu')))
model.eval()
data = torch.load("Models/adni_graph_data.pt", map_location=torch.device('cpu'), weights_only=False)
scaler = joblib.load("Models/scaler_adni.pkl")

# 🧠 Memory complaints checklist
st.subheader("Memory Complaints")
short_term = st.checkbox("Short-term memory loss")
concentration = st.checkbox("Difficulty concentrating")
disoriented = st.checkbox("Disoriented in familiar places")

st.markdown("---")

# 🧮 Medical Features
mmscore = st.slider("🧠 MMSE Score", 0, 30, 25)
cdrsb = st.slider("📈 CDR-SB", 0.0, 18.0, 0.5, step=0.1)

# 🎓 Education Dropdown
education_map = {
    "Primary": 8,
    "High School": 12,
    "Graduate": 16,
    "PhD": 20
}
education_level = st.selectbox("🎓 Highest Education", list(education_map.keys()))
educat = education_map[education_level]

# ⚧ Gender
gender = st.radio("⚧ Gender", ["Male", "Female"])
gender_code = 1 if gender == "Male" else 0

# 📅 Year of Birth
doby = st.number_input("📅 Year of Birth", min_value=1900, max_value=2024, value=1954)

# 👉 Predict
if st.button("🧠 Predict Diagnosis"):
    input_features = [mmscore, cdrsb, educat, gender_code, doby]
    new_scaled = scaler.transform([input_features])
    new_x = torch.tensor(new_scaled, dtype=torch.float)
    x_all = torch.cat([data.x, new_x], dim=0)

    out = model(x_all, data.edge_index)
    logits = out[-1]
    pred_class = logits.argmax().item()
    label = ["CN", "MCI", "AD"][pred_class]

    st.markdown(f"### 🧠 Predicted Diagnosis: **{label}**")

    # 🔍 Explanation
    st.markdown("#### 📌 Interpretation:")
    if label == "CN":
        st.info("Normal cognition detected based on low CDR-SB and high MMSE score.")
    elif label == "MCI":
        st.warning("Mild Cognitive Impairment (MCI) is suggested. Monitor symptoms closely.")
    else:
        st.error("High probability of Alzheimer's Disease (AD). Consider clinical follow-up.")

    # 📊 Prediction Confidence Chart
    probs = torch.nn.functional.softmax(logits, dim=0).detach().numpy()
    fig, ax = plt.subplots()
    ax.bar(["CN", "MCI", "AD"], probs * 100, color=["skyblue", "orange", "tomato"])
    ax.set_title("Prediction Confidence")
    ax.set_ylabel("Probability (%)")
    st.pyplot(fig)
