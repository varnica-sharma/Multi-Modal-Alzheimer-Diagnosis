# 🧠 Alzheimer’s Disease Prediction Using GCN

A Streamlit web app that predicts whether a patient is **Cognitively Normal (CN)**, has **Mild Cognitive Impairment (MCI)**, or **Alzheimer’s Disease (AD)** using a **Graph Convolutional Network (GCN)** trained on ADNI data.

---

## 🔗 Live App

👉 [Click here to use the app](https://varnica-sharma-multi-omics-alzheimer-diagnosis.streamlit.app)

---

## 📌 Overview

- Built using **PyTorch Geometric + Streamlit**
- Based on **ADNI** (Alzheimer’s Disease Neuroimaging Initiative) data
- GCN model classifies patients using clinical and demographic features
- Streamlit frontend with form-based inputs and graphical confidence output

---

## 🚀 Features

- 📋 Patient evaluation form with MMSE, CDR-SB, Education, Gender, Birth Year
- 🧠 Predicts AD stage: CN / MCI / AD
- 📊 Displays model confidence with interactive chart
- 🌙 Dark mode UI with custom styling
- 💾 All models and scalers included

---


---

## 📈 Input Features

| Feature          | Description                                         |
|------------------|-----------------------------------------------------|
| **MMSE Score**   | Mini-Mental State Examination (0 to 30)             |
| **CDR-SB**       | Clinical Dementia Rating - Sum of Boxes (0.0 to 18) |
| **Education**    | Total years of formal education                     |
| **Gender**       | Male / Female                                       |
| **Year of Birth**| Year (e.g., 1950)                                   |

---

## 🧠 Model Architecture

```python
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
```

---


🔒 Data Source

ADNI: Alzheimer's Disease Neuroimaging Initiative
👉 [Dataset](https://ida.loni.usc.edu)

📊 Example Prediction

Predicted Class: MCI
Confidence: 84.3%
"High CDR-SB and MMSE < 27 suggest mild cognitive decline"
