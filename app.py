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
    feature_defaults = (
        patient_table[feature_columns]
        .median(numeric_only=True)
        .to_dict()
    )
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
        "feature_defaults": feature_defaults,
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


def _neighbor_table(neighbors, show_title: bool = True):
    if not neighbors:
        return
    if show_title:
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

    defaults = artifact_state["feature_defaults"]

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=int(defaults.get("Age", 65)),
        )
        gender_label = st.selectbox("Biological sex", ["Female", "Male"])
        gender_numeric = 0.0 if gender_label == "Female" else 1.0
        education = st.number_input(
            "Education (years)",
            min_value=0,
            max_value=30,
            value=int(defaults.get("EducationYears", 16)),
        )
        apoe = st.select_slider(
            "APOE ε4 alleles",
            options=[0, 1, 2],
            value=int(defaults.get("APOE4Count", 0)),
            help="Count of APOE ε4 alleles from genetic testing",
        )
    with col2:
        mmse = st.slider(
            "MMSE",
            min_value=0,
            max_value=30,
            value=int(defaults.get("MMSE", 26)),
        )
        cdrsb = st.slider(
            "CDR-SB",
            min_value=0.0,
            max_value=18.0,
            value=float(defaults.get("CDRSB", 2.5)),
            step=0.1,
        )
        adas13 = st.slider(
            "ADAS-13",
            min_value=0.0,
            max_value=85.0,
            value=float(defaults.get("ADAS13", 20.0)),
            step=0.5,
        )
        faq = st.slider(
            "FAQ",
            min_value=0.0,
            max_value=30.0,
            value=float(defaults.get("FAQ", 5.0)),
            step=0.5,
        )

    st.markdown("### Optional details")
    st.caption("Leave blank to let the model impute typical values.")

    def optional_float(label: str, placeholder: str) -> float:
        raw = st.text_input(label, value="", placeholder=placeholder)
        try:
            return float(raw)
        except ValueError:
            return np.nan

    with st.expander("Cognitive extras", expanded=False):
        moca = optional_float("MoCA", f"e.g., {defaults.get('MOCA', 20):.0f}")
        ravlt = optional_float(
            "RAVLT immediate", f"e.g., {defaults.get('RAVLT_immediate', 35):.0f}"
        )

    with st.expander("CSF biomarkers (pg/mL)", expanded=False):
        abeta = optional_float("Aβ42", "e.g., 650")
        tau = optional_float("Total tau", "e.g., 300")
        ptau = optional_float("p-tau", "e.g., 30")

    with st.expander("MRI volumes (mm³)", expanded=False):
        hippocampus = optional_float("Hippocampus", "e.g., 3800")
        entorhinal = optional_float("Entorhinal", "e.g., 3500")
        ventricles = optional_float("Ventricles", "e.g., 45000")
        fusiform = optional_float("Fusiform", "e.g., 22000")
        midtemp = optional_float("Mid temporal", "e.g., 21000")
        whole_brain = optional_float("Whole brain", "e.g., 1000000")
        icv = optional_float("Intracranial volume (ICV)", "e.g., 1500000")

    submitted = st.form_submit_button("Predict diagnosis", use_container_width=True)

if submitted:
    raw_features = {col: np.nan for col in artifact_state["feature_columns"]}
    raw_features.update(
        {
            "Age": age,
            "GenderBinary": gender_numeric,
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
            "Tau_ABeta_Ratio": tau / abeta if (not np.isnan(tau)) and (not np.isnan(abeta)) and abeta > 0 else np.nan,
            "Hippocampus": hippocampus,
            "Entorhinal": entorhinal,
            "Ventricles": ventricles,
            "Fusiform": fusiform,
            "MidTemp": midtemp,
            "WholeBrain": whole_brain,
            "ICV": icv,
            "Hippocampus_ICV": hippocampus / icv if (not np.isnan(hippocampus)) and (not np.isnan(icv)) and icv > 0 else np.nan,
            "Entorhinal_ICV": entorhinal / icv if (not np.isnan(entorhinal)) and (not np.isnan(icv)) and icv > 0 else np.nan,
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
    predicted_prob = probabilities[predicted_class]
    st.success(f"Most likely diagnosis: **{predicted_class}** ({predicted_prob:.1%} probability)")

    _probability_bar(probabilities)

    with st.expander("What influenced this prediction?", expanded=False):
        _modality_contributions(modality_contributions)
        if neighbors:
            st.markdown("**Most similar ADNI cases**")
            _neighbor_table(neighbors[:5], show_title=False)
        else:
            st.caption("No close neighbours found in the training cohort.")

    _maybe_warn_missing_features(result, artifact_state["feature_columns"], raw_features)

st.write("---")
st.caption(
    "Model trained on ADNI baseline visits (demographics, cognition, CSF biomarkers, MRI volumes). "
    "For research and educational purposes only – not a substitute for clinical diagnosis."
)
