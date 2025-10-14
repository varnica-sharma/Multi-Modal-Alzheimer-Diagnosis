import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

from alzheimer_pipeline import (
    load_model_for_inference,
    predict_patient,
    LABEL_NAMES,
)


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_artifacts(artifact_dir: Path):
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
    ) = load_model_for_inference(artifact_dir)
    # Keep everything on CPU for inference
    model = model.cpu()
    graph_data = graph_data.cpu()
    return {
        "model": model,
        "graph_data": graph_data,
        "preprocessor": preprocessor,
        "neighbor_model": neighbor_model,
        "modality_processors": modality_processors,
        "modality_weights": modality_weights,
        "distance_scale": distance_scale,
        "patient_table": patient_table,
        "feature_columns": feature_columns,
    }


@st.cache_data(show_spinner=False)
def _load_reference_stats(patient_table: pd.DataFrame) -> Dict[str, Any]:
    summary = {}
    summary["counts"] = patient_table["Label"].map(LABEL_NAMES).value_counts().to_dict()
    summary["ages"] = {
        "min": float(np.nanmin(patient_table["Age"])),
        "max": float(np.nanmax(patient_table["Age"])),
        "median": float(np.nanmedian(patient_table["Age"])),
    }
    summary["education"] = {
        "min": float(np.nanmin(patient_table["EducationYears"])),
        "max": float(np.nanmax(patient_table["EducationYears"])),
        "median": float(np.nanmedian(patient_table["EducationYears"])),
    }
    summary["hippocampus"] = {
        "min": float(np.nanmin(patient_table["Hippocampus"])),
        "max": float(np.nanmax(patient_table["Hippocampus"])),
        "median": float(np.nanmedian(patient_table["Hippocampus"])),
    }
    return summary


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _render_sidebar(reference: Dict[str, Any]):
    st.sidebar.header("How to use")
    st.sidebar.write(
        "Fill in the patient profile using the sliders and dropdowns. "
        "Use approximate values if you do not have an exact measurement. "
        "When you click **Predict diagnosis**, we run the graph model trained on ADNI data "
        "and show the probability for each stage together with the modalities that contributed most."
    )
    st.sidebar.write("---")
    st.sidebar.write("**Reference ranges from the training cohort**")
    counts = reference.get("counts", {})
    for label, count in counts.items():
        st.sidebar.write(f"{label}: {count:,} baseline visits")
    ages = reference.get("ages", {})
    st.sidebar.write(f"Age range: {ages.get('min', 0):.0f} – {ages.get('max', 0):.0f} (median {ages.get('median', 0):.0f})")
    edu = reference.get("education", {})
    st.sidebar.write(
        f"Education (years): {edu.get('min', 0):.0f} – {edu.get('max', 0):.0f} "
        f"(median {edu.get('median', 0):.0f})"
    )
    hippo = reference.get("hippocampus", {})
    st.sidebar.write(
        f"Hippocampus volume: {hippo.get('min', 0):,.0f} – {hippo.get('max', 0):,.0f} mm³ "
        f"(median {hippo.get('median', 0):,.0f})"
    )


def _probability_bar(probabilities: Dict[str, float]):
    st.subheader("Diagnosis probabilities")
    chart_df = pd.DataFrame(
        {
            "Diagnosis": list(probabilities.keys()),
            "Probability": [prob * 100 for prob in probabilities.values()],
        }
    )
    st.bar_chart(chart_df.set_index("Diagnosis"))


def _modality_contributions(modality_contributions: Dict[str, float]):
    st.subheader("Modalities driving the prediction")
    mod_df = (
        pd.DataFrame(
            {
                "Modality": list(modality_contributions.keys()),
                "Contribution (%)": [weight * 100 for weight in modality_contributions.values()],
            }
        )
        .sort_values("Contribution (%)", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(mod_df, use_container_width=True)


def _neighbor_table(neighbors):
    if not neighbors:
        return
    st.subheader("Most similar cases in the training cohort")
    df = pd.DataFrame(neighbors)
    df.rename(columns={"distance": "Graph distance"}, inplace=True)
    st.dataframe(df, use_container_width=True)


def _maybe_warn_missing_features(result_dict: Dict[str, Any], feature_columns: list, input_dict: Dict[str, float]):
    missing = [col for col in feature_columns if pd.isna(input_dict.get(col))]
    if missing:
        with st.expander("Missing inputs we imputed"):
            st.write("These fields were filled in using cohort statistics before prediction:")
            st.write(", ".join(missing))


# ---------------------------------------------------------------------------
# Streamlit layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Multi-Modal Alzheimer's Graph Diagnosis",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Multi-Modal Alzheimer's Diagnosis Explorer")
st.write(
    "This interactive tool combines demographics, cognitive scores, cerebrospinal fluid biomarkers, "
    "and MRI measurements to estimate the probability of a patient being cognitively normal (CN), "
    "having mild cognitive impairment (MCI), or Alzheimer's disease (AD). "
    "Predictions are powered by a graph neural network trained on the ADNI cohort."
)

artifact_state = _load_artifacts(Path("artifacts"))
reference_stats = _load_reference_stats(artifact_state["patient_table"])
_render_sidebar(reference_stats)

with st.form("prediction_form"):
    st.subheader("Patient profile")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age (years)", min_value=45, max_value=100, value=72)
        gender = st.selectbox("Gender", options={"Female": 0.0, "Male": 1.0}, format_func=lambda x: "Female" if x == 0.0 else "Male")
        education = st.number_input("Education (years)", min_value=0, max_value=25, value=16)
        apoe = st.select_slider("APOE ε4 alleles", options=[0, 1, 2], value=1)
    with col2:
        mmse = st.slider("MMSE", min_value=0, max_value=30, value=26)
        cdrsb = st.slider("CDR-SB", min_value=0.0, max_value=18.0, value=2.5, step=0.1)
        adas13 = st.slider("ADAS-13", min_value=0.0, max_value=85.0, value=20.0, step=0.5)
        faq = st.slider("FAQ", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
        moca = st.slider("MoCA", min_value=0.0, max_value=30.0, value=20.0, step=0.5)
        ravlt = st.slider("RAVLT immediate", min_value=0.0, max_value=80.0, value=35.0, step=1.0)
    with col3:
        st.markdown("**CSF biomarkers (pg/mL)**")
        abeta = st.number_input("Aβ42", min_value=200.0, max_value=2000.0, value=650.0, step=10.0)
        tau = st.number_input("Total tau", min_value=0.0, max_value=1000.0, value=300.0, step=5.0)
        ptau = st.number_input("p-tau", min_value=0.0, max_value=200.0, value=30.0, step=1.0)
        st.markdown("**MRI volumes (mm³)**")
        hippocampus = st.number_input("Hippocampus", min_value=2000.0, max_value=9000.0, value=3800.0, step=50.0)
        entorhinal = st.number_input("Entorhinal", min_value=2000.0, max_value=8000.0, value=3500.0, step=50.0)
        ventricles = st.number_input("Ventricles", min_value=10000.0, max_value=120000.0, value=45000.0, step=500.0)
        fusiform = st.number_input("Fusiform", min_value=2000.0, max_value=40000.0, value=22000.0, step=100.0)
        midtemp = st.number_input("Mid temporal", min_value=5000.0, max_value=60000.0, value=21000.0, step=100.0)
        whole_brain = st.number_input("Whole brain", min_value=800000.0, max_value=1500000.0, value=1000000.0, step=1000.0)
        icv = st.number_input("Intracranial volume (ICV)", min_value=1000000.0, max_value=2000000.0, value=1500000.0, step=1000.0)

    submitted = st.form_submit_button("Predict diagnosis", use_container_width=True)

if submitted:
    raw_features = {col: np.nan for col in artifact_state["feature_columns"]}
    raw_features.update(
        {
            "Age": age,
            "GenderBinary": float(gender),
            "EducationYears": education,
            "APOE4Count": float(apoe),
            "MMSE": mmse,
            "CDRSB": cdrsb,
            "ADAS13": adas13,
            "FAQ": faq,
            "MOCA": moca,
            "RAVLT_immediate": ravlt,
            "CSF_ABETA42": abeta,
            "CSF_TAU": tau,
            "CSF_PTAU": ptau,
            "Tau_ABeta_Ratio": tau / abeta if abeta else np.nan,
            "Hippocampus": hippocampus,
            "Entorhinal": entorhinal,
            "Ventricles": ventricles,
            "Fusiform": fusiform,
            "MidTemp": midtemp,
            "WholeBrain": whole_brain,
            "ICV": icv,
            "Hippocampus_ICV": hippocampus / icv if icv else np.nan,
            "Entorhinal_ICV": entorhinal / icv if icv else np.nan,
        }
    )

    with st.spinner("Running graph neural network..."):
        result = predict_patient(
            artifact_state["model"],
            artifact_state["graph_data"],
            artifact_state["preprocessor"],
            artifact_state["neighbor_model"],
            artifact_state["modality_processors"],
            artifact_state["modality_weights"],
            artifact_state["distance_scale"],
            artifact_state["feature_columns"],
            raw_features,
            patient_table=artifact_state["patient_table"],
            return_explanations=True,
            neighbor_k=10,
        )

    probabilities = result["probabilities"]
    modality_contributions = result["modality_contributions"]
    neighbors = result["nearest_neighbors"]

    predicted_class = max(probabilities, key=probabilities.get)
    st.success(f"Most likely diagnosis: **{predicted_class}**")

    col_a, col_b = st.columns(2)
    with col_a:
        _probability_bar(probabilities)
    with col_b:
        _modality_contributions(modality_contributions)

    _neighbor_table(neighbors)
    _maybe_warn_missing_features(result, artifact_state["feature_columns"], raw_features)

    with st.expander("Raw output JSON"):
        st.json(
            {
                "probabilities": probabilities,
                "modality_contributions": modality_contributions,
                "nearest_neighbors": neighbors[:5],
            }
        )

st.write("---")
st.caption(
    "Model trained on ADNI baseline visits (demographics, cognition, CSF biomarkers, MRI volumes). "
    "For research and educational purposes only – not a substitute for clinical diagnosis."
)
