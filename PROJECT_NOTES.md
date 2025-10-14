# Alzheimer GCN Project – Refined Workflow

This document highlights the new project structure and how to reproduce the training / inference pipeline.

## Key Components

- `alzheimer_pipeline.py`: reusable module that loads the latest ADNI multi-modal tables, engineers demographic/cognitive/biomarker/imaging features, builds a weighted patient similarity graph, trains the GCN, evaluates performance, and saves artefacts.
- `Alzheimer.ipynb`: streamlined notebook that leverages the shared module for end-to-end experimentation and reporting.
- `app.py`: Streamlit app that wraps the trained model for interactive inference, showing probabilities, modality contributions, and exemplar neighbours.
- `artifacts/`: directory generated after training containing the model weights, graph object, preprocessing pipeline, k-NN model, feature arrays, and patient indices required for deployment.

## Reproducing Training

```bash
python3 -m alzheimer_pipeline
```

The script will:

1. Pull baseline visits from `ADNIMERGE`, enriching them with APOE genotype, CSF biomarkers, and imaging volumes.
2. Engineer demographic, cognitive, biomarker, and imaging features plus derived ratios (e.g., hippocampus/ICV, tau/abeta).
3. Split visits by `RID` into train/validation/test sets (70/15/15) to avoid patient leakage.
4. Build a weighted k-nearest-neighbour graph (`k=20`) using modality-aware scaling.
5. Train a two-layer GCN with edge weights, dropout, and early stopping.
6. Report validation macro F1 and write the updated artefacts to `artifacts/`.

## Inference

```python
from pathlib import Path
from alzheimer_pipeline import load_model_for_inference, predict_patient

(
    model,
    graph_data,
    preprocessor,
    neighbor_model,
    modality_processors,
    modality_weights,
    distance_scale,
    fused_features,
    patient_table,
    feature_columns,
) = load_model_for_inference(Path("artifacts"))

example = {
    "Age": 72,
    "EducationYears": 16,
    "GenderBinary": 0,
    "APOE4Count": 1,
    "MMSE": 26,
    "CDRSB": 1.5,
    "ADAS13": 18.0,
    "MOCA": 20,
    "FAQ": 5,
    "RAVLT_immediate": 35,
    "CSF_ABETA42": 650,
    "CSF_TAU": 300,
    "CSF_PTAU": 30,
    "Tau_ABeta_Ratio": 0.46,
    "Hippocampus": 3500,
    "Ventricles": 45000,
    "WholeBrain": 1_000_000,
    "Entorhinal": 3800,
    "Fusiform": 22_000,
    "MidTemp": 21_000,
    "ICV": 1_500_000,
    "Hippocampus_ICV": 0.0023,
    "Entorhinal_ICV": 0.0025,
}

probs = predict_patient(
    model,
    graph_data,
    preprocessor,
    neighbor_model,
    modality_processors,
    modality_weights,
    distance_scale,
    feature_columns,
    example,
    patient_table=patient_table,
    return_explanations=True,
)

print(probs["probabilities"])
print(probs["modality_contributions"])
print(probs["nearest_neighbors"][:3])
```

`predict_patient` attaches the new profile to the existing graph via the modality-aware neighbour model, runs it through the trained GCN, and returns class probabilities plus modality and neighbour explanations.

## Interpretability & Fairness

- `compute_feature_attributions` / `explain_nodes` – gradient × input features with modality aggregation and nearest-neighbour context.
- `compute_subgroup_metrics` – assess accuracy/recall across demographic or biomarker-defined cohorts (e.g., gender, APOE carriers, age bins).

## Presenting the Project

- Use the notebook to demonstrate: data overview, training curves, classification reports, confusion matrix, and inference.
- Align interview talking points with the refreshed implementation (multi-modal features, weighted graph construction, patient-level splits, deployment story).
- Update the Streamlit app to call `load_model_for_inference` / `predict_patient` so the live demo mirrors the multi-modal pipeline and exposes probability outputs + modality explanations.

## Streamlit Deployment

- `requirements.txt` lists the dependencies used for local and Streamlit Cloud runs (PyTorch + PyG CPU wheels included).
- Launch locally with `streamlit run app.py`.
- On Streamlit Cloud, push `app.py`, `requirements.txt`, `alzheimer_pipeline.py`, and the `artifacts/` folder; set the entrypoint to `app.py`.
