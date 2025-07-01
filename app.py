# streamlit_app.py
import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import joblib
import numpy as np

st.set_page_config(
    page_title="Alzheimer's GNN Predictor",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto",
)

# Title
st.title("🧠 Alzheimer's Disease Prediction Using GCN")
st.markdown("Enter patient features to predict diagnosis: CN / MCI / AD")

# Load model, data, and scaler
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

model = GCN(5, 32, 3)
model.load_state_dict(torch.load("Models/gnn_model_adni.pt", map_location=torch.device('cpu')))
model.eval()
data = torch.load("Models/adni_graph_data.pt", map_location=torch.device('cpu'), weights_only=False)
scaler = joblib.load("Models/scaler_adni.pkl")

# User inputs
mmscore = st.slider("MMSE Score", 0, 30, 25)
cdrsb = st.slider("CDR-SB", 0.0, 18.0, 0.5, step=0.1)
educat = st.slider("Years of Education", 0, 30, 16)
gender = st.radio("Gender", ["Male", "Female"])
doby = st.number_input("Year of Birth", min_value=1900, max_value=2024, value=1947)

# Convert gender
gender_code = 1 if gender == "Male" else 0

# Predict on submit
if st.button("Predict Diagnosis"):
    input_features = [mmscore, cdrsb, educat, gender_code, doby]
    x_scaled = scaler.transform(data.x)
    new_scaled = scaler.transform([input_features])
    new_x = torch.tensor(new_scaled, dtype=torch.float)
    x_all = torch.cat([data.x, new_x], dim=0)
    out = model(x_all, data.edge_index)
    pred_class = out[-1].argmax().item()
    label = ["CN", "MCI", "AD"][pred_class]
    st.success(f"🧠 Predicted Diagnosis: {label}")
