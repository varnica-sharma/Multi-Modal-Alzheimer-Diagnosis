# 🧠 Alzheimer’s Disease Prediction Using Multi-Modal GCN

A Streamlit web app backed by a Graph Convolutional Network (GCN) that classifies a patient as **Cognitively Normal (CN)**, **Mild Cognitive Impairment (MCI)**, or **Alzheimer’s Disease (AD)**.  
The model is trained on baseline visits from the ADNI cohort and blends demographics, cognition, cerebrospinal fluid (CSF) biomarkers, and MRI-derived brain volumes for richer context.

> **Note:** This project is for research and educational use only. It is *not* a substitute for professional medical diagnosis.

---

## 🔗 Live App

👉 [Launch the Streamlit app](https://multi-modal-alzheimer-diagnosis.streamlit.app)

---

## 📌 Overview

- Built with **PyTorch Geometric**, **PyTorch**, **scikit-learn**, and **Streamlit**
- Uses **ADNI** baseline data (`ADNIMERGE`, APOE genotypes, CSF Elecsys biomarkers, FreeSurfer MRI summaries)
- Weighted patient similarity graph + two-layer GCN for CN / MCI / AD classification
- Streamlit frontend with guided form inputs, probability charts, modality attributions, and similar-patient lookup
- Subgroup reporting (gender, APOE ε4, age bins) to monitor fairness

---

## 🚀 Key Features

- 📋 **Patient profile form**: demographics, cognitive tests, CSF biomarkers, and MRI volumes (with derived ratios auto-computed)
- 🧠 **Multi-class prediction**: CN / MCI / AD with calibrated probabilities
- 📊 **Interpretability**:
  - Modality contribution table (demographic vs cognitive vs biomarker vs imaging impact)
  - Nearest-neighbour table showing analogous ADNI cases and diagnoses
- 📈 **Model monitoring**: subgroup accuracy metrics in training logs/notebook
- 💾 **Artifacts included**: trained model weights, graph object, preprocessing pipelines, k-NN neighbour model

---

## 📈 Input Features

| Category       | Feature / Derived Metric                     |
|----------------|----------------------------------------------|
| Demographic    | Age, Education (years), Gender, APOE ε4 count |
| Cognitive      | MMSE, CDR-SB, ADAS-13, MoCA, FAQ, RAVLT immediate |
| CSF Biomarkers | Aβ42, Total Tau, p-Tau, Tau/Aβ42 ratio        |
| Imaging        | Hippocampus, Entorhinal, Ventricles, Fusiform, Mid Temporal, Whole Brain, ICV, Hippocampus/ICV, Entorhinal/ICV |

---

## 🧠 Model Snapshot

```python
class GCNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels=96, dropout=0.4):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels // 2)
        self.classifier = nn.Linear(hidden_channels // 2, 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight).relu()
        x = self.dropout(x)
        return self.classifier(x)
```

Weighted k-NN edges are built from modality-specific embeddings (with adjustable weights) and fed into the GCN alongside edge weights.

---

## 🗂️ Repository Layout

```
├── app.py                  # Streamlit UI
├── alzheimer_pipeline.py   # Data fusion, graph construction, training, inference utilities
├── Alzheimer.ipynb         # Notebook: data audit, training, evaluation, interpretability
├── PROJECT_NOTES.md        # Quick usage snippets and deployment tips
├── artifacts/              # Saved model, graph data, preprocessors, metadata (generated)
├── Datasets/               # ADNI CSVs (not included; see Data Access)
├── requirements.txt        # Dependencies for local / Streamlit runs
└── README.md
```

---

## 🔒 Data Access (ADNI)

1. Request ADNI access at https://ida.loni.usc.edu  
2. Download the latest CSVs:
   - `ADNIMERGE_*.csv`
   - `All_Subjects_APOERES_*.csv`
   - `All_Subjects_UPENNBIOMK_ROCHE_ELECSYS_*.csv`
   - `All_Subjects_UCSFFSX7_*.csv`
   - Optional: `All_Subjects_MMSE_*.csv`, `All_Subjects_CDR_*.csv`, `All_Subjects_ADAS_*.csv`
3. Place them inside the `Datasets/` folder (keep filenames intact)

⚠️ **Do not commit ADNI data to a public repository.**

---

## 🛠️ Setup & Training (Local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train or refresh artefacts
python3 -m alzheimer_pipeline
```

The training script handles:

1. Feature engineering + modality fusion
2. Participant-level train/val/test split
3. Weighted k-NN graph construction
4. GCN training with early stopping
5. Metric reporting (macro F1, confusion matrix, subgroup accuracy)
6. Artifact export to `artifacts/`

Open `Alzheimer.ipynb` for a guided walkthrough (data overview → training curves → interpretability → inference example).

---

## 🌐 Streamlit App

Launch locally:

```bash
streamlit run app.py
```

You’ll see:

- Guided sidebar with cohort ranges
- Input form (demographics, cognition, CSF, imaging)
- Probability bar chart for CN / MCI / AD
- Modality contribution table
- Similar cases pulled from the ADNI graph
- Raw JSON output (for advanced users)

### Deploying to Streamlit Cloud

1. Push to GitHub: `app.py`, `alzheimer_pipeline.py`, `requirements.txt`, `alzheimer_pipeline` assets, `artifacts/`, docs/notebooks as needed (do **not** include `Datasets/`).
2. Create a Streamlit Cloud app pointing to `app.py`.
3. Ensure `artifacts/` is part of the repo so the online app can load the trained model.

---

## 🔍 Explainability & Monitoring

- `predict_patient(..., return_explanations=True)` returns probabilities, modality contributions, and nearest neighbours.
- `explain_nodes` (in `alzheimer_pipeline.py`) provides gradient × input attributions per node with modality aggregation and neighbourhood context.
- `compute_subgroup_metrics` prints accuracy/recall across gender, APOE4 carrier status, and age bins.

---

## ⚠️ Limitations & Next Steps

- Baseline visits only (no longitudinal prediction yet)  
- Relies on FreeSurfer volumetrics; adding PET SUVRs or cortical thickness trajectories could improve sensitivity  
- Further calibration / uncertainty estimation (e.g., temperature scaling, conformal prediction) is future work

---

## 📚 Credits

- ADNI: Alzheimer’s Disease Neuroimaging Initiative 👉 [Dataset](https://ida.loni.usc.edu)
- Libraries: PyTorch, PyTorch Geometric, scikit-learn, NumPy, Pandas, Streamlit

