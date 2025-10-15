"""Comprehensive multi-modal pipeline for the Alzheimer's diagnosis project.

This module centralises data ingestion, feature engineering, graph construction,
model training, evaluation utilities, and inference helpers.  It fuses
demographic, cognitive, biomarker, and imaging signals into a patient similarity
graph and trains a weighted GCN for CN/MCI/AD classification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from contextlib import nullcontext
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr
from torch_geometric.nn import GCNConv


# ---------------------------------------------------------------------------
# Constants and lightweight containers
# ---------------------------------------------------------------------------

LABEL_NAMES = {
    0: "Cognitively Normal",
    1: "Mild Cognitive Impairment",
    2: "Alzheimer's Disease",
}

DIAGNOSIS_MAP = {
    "CN": 0,
    "SMC": 0,
    "EMCI": 1,
    "MCI": 1,
    "LMCI": 1,
    "Dementia": 2,
    "AD": 2,
}

MODALITY_FEATURES = {
    "demographic": ["Age", "EducationYears", "GenderBinary", "APOE4Count"],
    "cognitive": ["MMSE", "CDRSB", "ADAS13", "MOCA", "FAQ", "RAVLT_immediate"],
    "biomarker": ["CSF_ABETA42", "CSF_TAU", "CSF_PTAU", "Tau_ABeta_Ratio"],
    "imaging": [
        "Hippocampus",
        "Ventricles",
        "WholeBrain",
        "Entorhinal",
        "Fusiform",
        "MidTemp",
        "ICV",
        "Hippocampus_ICV",
        "Entorhinal_ICV",
    ],
}

MODALITY_WEIGHTS = {
    "demographic": 0.2,
    "cognitive": 0.35,
    "biomarker": 0.2,
    "imaging": 0.25,
}


@dataclass
class ModalityProcessor:
    columns: List[str]
    pipeline: Pipeline


@dataclass
class DatasetArtifacts:
    data: Data
    feature_columns: List[str]
    preprocessor: Pipeline
    modality_processors: Dict[str, ModalityProcessor]
    modality_weights: Dict[str, float]
    neighbor_model: NearestNeighbors
    distance_scale: float
    patient_index: pd.DataFrame
    patient_table: pd.DataFrame
    fused_features: np.ndarray


# ---------------------------------------------------------------------------
# Data loading and preprocessing helpers
# ---------------------------------------------------------------------------

def _find_latest_csv(data_dir: Path, pattern: str) -> Path:
    candidates = sorted(data_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No files matching {pattern} in {data_dir}")
    return candidates[-1]


def load_raw_tables(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the core ADNI tables required for multi-modal features."""
    adnimerge = pd.read_csv(_find_latest_csv(data_dir, "ADNIMERGE*.csv"), low_memory=False)
    apoe = pd.read_csv(_find_latest_csv(data_dir, "*APOERES*.csv"), low_memory=False)
    csf = pd.read_csv(
        _find_latest_csv(data_dir, "*UPENNBIOMK_ROCHE_ELECSYS*.csv"),
        low_memory=False,
    )
    return adnimerge, apoe, csf


def _parse_gender(series: pd.Series) -> pd.Series:
    return series.str.lower().map({"male": 1.0, "female": 0.0})


def _count_apoe4(genotype: str) -> float:
    if isinstance(genotype, str):
        return float(genotype.count("4"))
    return np.nan


def build_patient_feature_table(
    adnimerge: pd.DataFrame,
    apoe: pd.DataFrame,
    csf: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """Return baseline patient table with engineered multi-modal features."""
    base = adnimerge[adnimerge["VISCODE"].str.lower().isin({"bl", "sc"})].copy()
    base["DiagnosisRaw"] = base["DX_bl"].fillna(base["DX"])
    base["DiagnosisRaw"] = base["DiagnosisRaw"].str.upper()
    base["Label"] = base["DiagnosisRaw"].map(DIAGNOSIS_MAP)
    base = base.dropna(subset=["Label"])
    base["Label"] = base["Label"].astype(int)

    rename_map = {
        "RID": "RID",
        "VISCODE": "VISCODE",
        "EXAMDATE": "ExamDate",
        "AGE": "Age",
        "PTGENDER": "Gender",
        "PTEDUCAT": "EducationYears",
        "APOE4": "APOE4Count",
        "MMSE": "MMSE",
        "CDRSB": "CDRSB",
        "ADAS13": "ADAS13",
        "FAQ": "FAQ",
        "MOCA": "MOCA",
        "RAVLT_immediate": "RAVLT_immediate",
        "ABETA": "ABETA_estimate",
        "TAU": "TAU_estimate",
        "PTAU": "PTAU_estimate",
        "Hippocampus": "Hippocampus",
        "Ventricles": "Ventricles",
        "WholeBrain": "WholeBrain",
        "Entorhinal": "Entorhinal",
        "Fusiform": "Fusiform",
        "MidTemp": "MidTemp",
        "ICV": "ICV",
    }

    base = base[list(rename_map.keys()) + ["Label"]].rename(columns=rename_map)

    base["GenderBinary"] = _parse_gender(base["Gender"])
    base["EducationYears"] = pd.to_numeric(base["EducationYears"], errors="coerce")
    base["Age"] = pd.to_numeric(base["Age"], errors="coerce")
    base["APOE4Count"] = pd.to_numeric(base["APOE4Count"], errors="coerce")

    apoe_processed = (
        apoe.assign(APOE4_from_genotype=apoe["GENOTYPE"].apply(_count_apoe4))
        .groupby("RID")[["APOE4_from_genotype"]]
        .max()
        .reset_index()
    )
    base = base.merge(apoe_processed, on="RID", how="left")
    base["APOE4Count"] = base["APOE4Count"].fillna(base["APOE4_from_genotype"])
    base.drop(columns=["APOE4_from_genotype"], inplace=True)

    csf_bl = csf[csf["VISCODE2"].str.lower().isin({"bl", "sc"})].copy()
    for col in ["ABETA42", "TAU", "PTAU"]:
        csf_bl[col] = pd.to_numeric(csf_bl[col], errors="coerce")
    csf_summary = csf_bl.groupby("RID")[["ABETA42", "TAU", "PTAU"]].median().reset_index()
    csf_summary = csf_summary.rename(
        columns={
            "ABETA42": "CSF_ABETA42",
            "TAU": "CSF_TAU",
            "PTAU": "CSF_PTAU",
        }
    )
    base = base.merge(csf_summary, on="RID", how="left")

    base["CSF_ABETA42"] = pd.to_numeric(base["CSF_ABETA42"], errors="coerce")
    base["CSF_TAU"] = pd.to_numeric(base["CSF_TAU"], errors="coerce")
    base["CSF_PTAU"] = pd.to_numeric(base["CSF_PTAU"], errors="coerce")
    base["Tau_ABeta_Ratio"] = base["CSF_TAU"] / base["CSF_ABETA42"]

    volumetric_cols = ["Hippocampus", "Ventricles", "WholeBrain", "Entorhinal", "Fusiform", "MidTemp", "ICV"]
    for col in volumetric_cols:
        base[col] = pd.to_numeric(base[col], errors="coerce")

    base["Hippocampus_ICV"] = base["Hippocampus"] / base["ICV"]
    base["Entorhinal_ICV"] = base["Entorhinal"] / base["ICV"]

    feature_columns = sorted(
        set(
            MODALITY_FEATURES["demographic"]
            + MODALITY_FEATURES["cognitive"]
            + MODALITY_FEATURES["biomarker"]
            + MODALITY_FEATURES["imaging"]
        )
    )

    patient_frame = base[["RID", "VISCODE", "Label", *feature_columns]].copy()
    return patient_frame, feature_columns


def _train_val_test_split(df: pd.DataFrame, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = df["RID"].to_numpy()
    splitter_primary = GroupShuffleSplit(train_size=0.7, random_state=seed, n_splits=1)
    train_idx, temp_idx = next(splitter_primary.split(df, df["Label"], groups))

    temp_groups = groups[temp_idx]
    temp_df = df.iloc[temp_idx]
    splitter_secondary = GroupShuffleSplit(train_size=0.5, random_state=seed + 1, n_splits=1)
    val_rel, test_rel = next(splitter_secondary.split(temp_df, temp_df["Label"], temp_groups))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Modalities -> fused representation
# ---------------------------------------------------------------------------

def _fit_modalities(
    X: pd.DataFrame,
    train_idx: np.ndarray,
    modality_features: Dict[str, List[str]],
    modality_weights: Dict[str, float],
) -> Tuple[Dict[str, ModalityProcessor], Dict[str, float], np.ndarray]:
    processors: Dict[str, ModalityProcessor] = {}
    transformed_arrays: Dict[str, np.ndarray] = {}
    active_weights: Dict[str, float] = {}

    for modality, columns in modality_features.items():
        present_cols = [col for col in columns if col in X.columns]
        if not present_cols:
            continue
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        pipeline.fit(X.iloc[train_idx][present_cols])
        transformed = pipeline.transform(X[present_cols])

        processors[modality] = ModalityProcessor(columns=present_cols, pipeline=pipeline)
        transformed_arrays[modality] = transformed
        active_weights[modality] = modality_weights.get(modality, 1.0)

    if not processors:
        raise ValueError("No modality features available for fusion.")

    total_weight = sum(active_weights.values())
    normalized_weights = {k: v / total_weight for k, v in active_weights.items()}

    fused_parts = [
        np.sqrt(normalized_weights[modality]) * transformed_arrays[modality]
        for modality in processors
    ]
    fused_features = np.hstack(fused_parts)

    return processors, normalized_weights, fused_features


# ---------------------------------------------------------------------------
# Graph preparation and PyG data assembly
# ---------------------------------------------------------------------------

def prepare_dataset(data_dir: Path, k_neighbors: int = 20, seed: int = 42) -> DatasetArtifacts:
    adnimerge, apoe, csf = load_raw_tables(data_dir)
    patient_frame, feature_columns = build_patient_feature_table(adnimerge, apoe, csf)

    train_idx, val_idx, test_idx = _train_val_test_split(patient_frame, seed)

    X = patient_frame[feature_columns]
    y = patient_frame["Label"].to_numpy()

    preprocessor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor.fit(X.iloc[train_idx])
    X_all = preprocessor.transform(X)

    modality_processors, modality_weights, fused_features = _fit_modalities(
        X, train_idx, MODALITY_FEATURES, MODALITY_WEIGHTS
    )

    neighbor_model = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="euclidean")
    neighbor_model.fit(fused_features)
    distances, indices = neighbor_model.kneighbors(fused_features)

    distance_scale = float(np.median(distances[:, 1:]))
    distance_scale = max(distance_scale, 1e-6)

    rows: List[int] = []
    cols: List[int] = []
    weights: List[float] = []

    num_nodes = fused_features.shape[0]
    for i in range(num_nodes):
        for dist, j in zip(distances[i, 1:], indices[i, 1:]):
            weight = float(np.exp(-dist / distance_scale))
            rows.extend([i, j])
            cols.extend([j, i])
            weights.extend([weight, weight])

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)

    loop_index = torch.arange(num_nodes, dtype=torch.long)
    loop_edges = torch.stack([loop_index, loop_index])
    edge_index = torch.cat([edge_index, loop_edges], dim=1)
    edge_weight = torch.cat([edge_weight, torch.ones(num_nodes, dtype=torch.float)], dim=0)

    data = Data(
        x=torch.from_numpy(X_all).float(),
        y=torch.from_numpy(y).long(),
        edge_index=edge_index,
        edge_weight=edge_weight,
    )

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    patient_index = patient_frame.loc[:, ["RID", "VISCODE"]].copy()
    patient_index["split"] = "train"
    patient_index.loc[val_idx, "split"] = "val"
    patient_index.loc[test_idx, "split"] = "test"

    return DatasetArtifacts(
        data=data,
        feature_columns=feature_columns,
        preprocessor=preprocessor,
        modality_processors=modality_processors,
        modality_weights=modality_weights,
        neighbor_model=neighbor_model,
        distance_scale=distance_scale,
        patient_index=patient_index,
        patient_table=patient_frame,
        fused_features=fused_features,
    )


# ---------------------------------------------------------------------------
# Model definition and training
# ---------------------------------------------------------------------------

class GCNClassifier(nn.Module):
    """Weighted two-layer GCN classifier."""

    def __init__(self, in_channels: int, hidden_channels: int = 96, dropout: float = 0.4):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels // 2)
        self.classifier = nn.Linear(hidden_channels // 2, len(LABEL_NAMES))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = torch.relu(x)
        x = self.dropout(x)
        return self.classifier(x)


def _masked_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    if mask.sum() == 0:
        return 0.0
    predictions = logits[mask].argmax(dim=1)
    correct = (predictions == targets[mask]).sum().item()
    return correct / int(mask.sum())


def train_gcn(
    data: Data,
    epochs: int = 400,
    lr: float = 5e-4,
    weight_decay: float = 5e-4,
    patience: int = 60,
    device: torch.device | None = None,
) -> Tuple[GCNClassifier, List[Dict[str, float]]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GCNClassifier(data.num_features).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    moving_data = Data(
        x=data.x.to(device),
        y=data.y.to(device),
        edge_index=data.edge_index.to(device),
        edge_weight=data.edge_weight.to(device),
        train_mask=data.train_mask.to(device),
        val_mask=data.val_mask.to(device),
        test_mask=data.test_mask.to(device),
    )

    history: List[Dict[str, float]] = []
    best_state = None
    best_val_acc = -np.inf
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(moving_data.x, moving_data.edge_index, moving_data.edge_weight)
        loss = criterion(logits[moving_data.train_mask], moving_data.y[moving_data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(moving_data.x, moving_data.edge_index, moving_data.edge_weight)
            train_acc = _masked_accuracy(logits, moving_data.y, moving_data.train_mask)
            val_acc = _masked_accuracy(logits, moving_data.y, moving_data.val_mask)
            val_loss = criterion(logits[moving_data.val_mask], moving_data.y[moving_data.val_mask]).item()

        history.append({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc, "val_loss": val_loss})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def summarise_performance(model: nn.Module, data: Data) -> Dict[str, Dict[str, float]]:
    model.eval()
    device = next(model.parameters()).device
    logits = model(
        data.x.to(device),
        data.edge_index.to(device),
        data.edge_weight.to(device),
    )
    preds = logits.argmax(dim=1).cpu().numpy()
    truth = data.y.cpu().numpy()

    reports = {}
    for split_name, mask in (("train", data.train_mask), ("val", data.val_mask), ("test", data.test_mask)):
        indices = mask.cpu().numpy().astype(bool)
        reports[split_name] = classification_report(
            truth[indices],
            preds[indices],
            target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
            output_dict=True,
            zero_division=0,
        )
    return reports


def confusion_matrix_split(model: nn.Module, data: Data, split: str = "test") -> np.ndarray:
    mask = getattr(data, f"{split}_mask")
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(
            data.x.to(device),
            data.edge_index.to(device),
            data.edge_weight.to(device),
        )
    preds = logits.argmax(dim=1).cpu().numpy()
    truth = data.y.cpu().numpy()
    indices = mask.cpu().numpy().astype(bool)
    return confusion_matrix(truth[indices], preds[indices], labels=list(LABEL_NAMES.keys()))


# ---------------------------------------------------------------------------
# Explainability and subgroup analysis
# ---------------------------------------------------------------------------

def compute_feature_attributions(
    model: nn.Module,
    data: Data,
    feature_columns: List[str],
    node_indices: Iterable[int],
    target_class: Optional[int] = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """Gradient-based feature attributions (gradient * input) for selected nodes."""
    device = next(model.parameters()).device
    model.eval()
    node_list = list(node_indices)
    if not node_list:
        return pd.DataFrame(columns=feature_columns)

    x = data.x.clone().detach().to(device)
    x.requires_grad_(True)
    edge_index = data.edge_index.to(device)
    edge_weight = data.edge_weight.to(device) if getattr(data, "edge_weight", None) is not None else None

    logits = model(x, edge_index, edge_weight=edge_weight)
    if target_class is None:
        class_ids = logits.argmax(dim=1)
    else:
        class_ids = torch.full_like(logits.argmax(dim=1), target_class, device=device)

    attributions = []
    for idx in node_list:
        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        logit = logits[idx, class_ids[idx]]
        logit.backward(retain_graph=True)
        grad = x.grad[idx].detach().cpu()
        feat = x[idx].detach().cpu()
        attr = grad * feat
        if normalize:
            denom = attr.abs().sum()
            if denom > 0:
                attr = attr / denom
        attributions.append(attr.numpy())

    df = pd.DataFrame(attributions, columns=feature_columns, index=node_list)
    return df


def aggregate_modality_attributions(
    feature_attributions: pd.DataFrame,
    modality_map: Dict[str, List[str]],
) -> pd.DataFrame:
    """Aggregate feature attributions into modality-level scores."""
    modality_scores = {}
    for modality, columns in modality_map.items():
        cols_present = [col for col in columns if col in feature_attributions.columns]
        if not cols_present:
            continue
        modality_scores[modality] = feature_attributions[cols_present].abs().sum(axis=1)

    modality_df = pd.DataFrame(modality_scores)
    row_sums = modality_df.sum(axis=1)
    modality_df = modality_df.div(row_sums.replace(0, np.nan), axis=0)
    return modality_df.fillna(0.0)


def describe_node_neighbours(
    neighbor_model: NearestNeighbors,
    fused_features: np.ndarray,
    patient_table: pd.DataFrame,
    node_index: int,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    distances, indices = neighbor_model.kneighbors(
        fused_features[node_index].reshape(1, -1),
        n_neighbors=min(top_k + 1, len(patient_table)),
        return_distance=True,
    )
    neighbours = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == node_index:
            continue
        row = patient_table.iloc[int(idx)]
        neighbours.append(
            {
                "RID": int(row["RID"]),
                "VISCODE": str(row["VISCODE"]),
                "Diagnosis": LABEL_NAMES.get(int(row["Label"]), "Unknown"),
                "distance": float(dist),
            }
        )
        if len(neighbours) >= top_k:
            break
    return neighbours


def explain_nodes(
    model: nn.Module,
    data: Data,
    artifacts: DatasetArtifacts,
    node_indices: Iterable[int],
    target_class: Optional[int] = None,
    top_k: int = 5,
) -> Dict[int, Dict[str, Any]]:
    """Return feature and modality attributions plus neighbour context for nodes."""
    feature_attr = compute_feature_attributions(
        model,
        data,
        artifacts.feature_columns,
        node_indices,
        target_class=target_class,
    )
    modality_attr = aggregate_modality_attributions(feature_attr, MODALITY_FEATURES)

    explanations: Dict[int, Dict[str, Any]] = {}
    for idx in feature_attr.index:
        explanations[idx] = {
            "feature_attributions": feature_attr.loc[idx].to_dict(),
            "modality_attributions": modality_attr.loc[idx].to_dict() if idx in modality_attr.index else {},
            "nearest_neighbors": describe_node_neighbours(
                artifacts.neighbor_model,
                artifacts.fused_features,
                artifacts.patient_table,
                idx,
                top_k=top_k,
            ),
        }
    return explanations


def compute_subgroup_metrics(
    model: nn.Module,
    data: Data,
    patient_table: pd.DataFrame,
    split: str = "test",
    group_by: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Evaluate accuracy by demographic / biomarker subgroups."""
    if group_by is None:
        group_by = ["GenderBinary"]

    mask = getattr(data, f"{split}_mask")
    indices = np.where(mask.cpu().numpy())[0]
    if len(indices) == 0:
        return pd.DataFrame(columns=group_by + ["count", "accuracy"])

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(
            data.x.to(device),
            data.edge_index.to(device),
            data.edge_weight.to(device),
        )
    preds = logits.argmax(dim=1).cpu().numpy()
    truth = data.y.cpu().numpy()

    eval_df = patient_table.iloc[indices].copy()
    eval_df["pred"] = preds[indices]
    eval_df["truth"] = truth[indices]
    eval_df["correct"] = eval_df["pred"] == eval_df["truth"]

    grouped = eval_df.groupby(group_by, observed=False)
    records = []
    for keys, group in grouped:
        key_tuple = (keys,) if not isinstance(keys, tuple) else keys
        record = {col: val for col, val in zip(group_by, key_tuple)}
        record["count"] = int(len(group))
        record["accuracy"] = float(group["correct"].mean())
        for label_id, label_name in LABEL_NAMES.items():
            mask_label = group["truth"] == label_id
            if mask_label.any():
                recall = (group.loc[mask_label, "pred"] == label_id).mean()
                record[f"recall_{label_name}"] = float(recall)
        records.append(record)

    return pd.DataFrame(records).sort_values("count", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Persistence and deployment helpers
# ---------------------------------------------------------------------------

def save_artifacts(
    output_dir: Path,
    artifacts: DatasetArtifacts,
    model: nn.Module,
    history: Iterable[Dict[str, float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_dir / "gnn_model.pt")
    torch.save(artifacts.data, output_dir / "graph_data.pt")

    joblib.dump(artifacts.preprocessor, output_dir / "preprocessor.joblib")
    joblib.dump(artifacts.neighbor_model, output_dir / "neighbors.joblib")
    joblib.dump(artifacts.modality_processors, output_dir / "modality_processors.joblib")

    artifacts.patient_index.to_parquet(output_dir / "patient_index.parquet", index=False)
    artifacts.patient_table.to_parquet(output_dir / "patient_table.parquet", index=False)
    np.save(output_dir / "fused_features.npy", artifacts.fused_features)

    metadata = {
        "feature_columns": artifacts.feature_columns,
        "label_names": LABEL_NAMES,
        "modality_weights": artifacts.modality_weights,
        "distance_scale": artifacts.distance_scale,
        "modality_features": MODALITY_FEATURES,
        "history": list(history),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def load_model_for_inference(
    artifact_dir: Path,
) -> Tuple[
    GCNClassifier,
    Data,
    Pipeline,
    NearestNeighbors,
    Dict[str, ModalityProcessor],
    Dict[str, float],
    float,
    np.ndarray,
    pd.DataFrame,
    List[str],
]:
    data_path = artifact_dir / "graph_data.pt"
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([Data, DataEdgeAttr])
        data: Data = torch.load(data_path, weights_only=False, map_location="cpu")
    else:
        if hasattr(torch.serialization, "safe_globals"):
            with torch.serialization.safe_globals([Data, DataEdgeAttr]):
                data = torch.load(data_path, map_location="cpu")
        else:
            data = torch.load(data_path, map_location="cpu")
    preprocessor: Pipeline = joblib.load(artifact_dir / "preprocessor.joblib")
    neighbor_model: NearestNeighbors = joblib.load(artifact_dir / "neighbors.joblib")
    modality_processors: Dict[str, ModalityProcessor] = joblib.load(
        artifact_dir / "modality_processors.joblib"
    )
    metadata = json.loads((artifact_dir / "metadata.json").read_text())

    modality_weights: Dict[str, float] = metadata["modality_weights"]
    distance_scale: float = metadata["distance_scale"]
    feature_columns: List[str] = metadata["feature_columns"]
    fused_features = np.load(artifact_dir / "fused_features.npy")
    patient_table = pd.read_parquet(artifact_dir / "patient_table.parquet")

    model = GCNClassifier(in_channels=data.num_features)
    model.load_state_dict(torch.load(artifact_dir / "gnn_model.pt", map_location="cpu", weights_only=True))
    model.eval()

    return (
        model,
        data,
        preprocessor,
        neighbor_model,
        modality_processors,
        modality_weights,
        distance_scale,
        fused_features,
        patient_table,
        feature_columns,
    )


def predict_patient(
    model: nn.Module,
    base_data: Data,
    preprocessor: Pipeline,
    neighbor_model: NearestNeighbors,
    modality_processors: Dict[str, ModalityProcessor],
    modality_weights: Dict[str, float],
    distance_scale: float,
    feature_columns: List[str],
    raw_features: Dict[str, float],
    patient_table: Optional[pd.DataFrame] = None,
    return_explanations: bool = False,
    neighbor_k: int = 5,
) -> Dict[str, Any]:
    feature_vector = [raw_features.get(col, np.nan) for col in feature_columns]
    feature_frame = pd.DataFrame([feature_vector], columns=feature_columns)

    processed_features = preprocessor.transform(feature_frame)
    new_x = torch.from_numpy(processed_features).float()
    augmented_x = torch.cat([base_data.x, new_x], dim=0)
    new_node_idx = augmented_x.shape[0] - 1

    fused_parts = []
    modality_vectors = {}
    for modality, processor in modality_processors.items():
        weight = modality_weights.get(modality, 0.0)
        if weight <= 0:
            continue
        transformed = processor.pipeline.transform(feature_frame[processor.columns])
        weighted = np.sqrt(weight) * transformed
        fused_parts.append(weighted)
        modality_vectors[modality] = weighted

    if not fused_parts:
        raise ValueError("No modality features available for inference.")

    fused_vector = np.hstack(fused_parts)
    distances, indices = neighbor_model.kneighbors(fused_vector, n_neighbors=neighbor_k + 1, return_distance=True)
    neighbour_ids = indices[0]
    neighbour_dists = distances[0]

    rows = []
    cols = []
    weights = []
    for dist, neighbour in zip(neighbour_dists, neighbour_ids):
        weight = float(np.exp(-dist / max(distance_scale, 1e-6)))
        rows.extend([new_node_idx, neighbour])
        cols.extend([neighbour, new_node_idx])
        weights.extend([weight, weight])

    new_edges = torch.tensor([rows, cols], dtype=torch.long)
    new_weights = torch.tensor(weights, dtype=torch.float)

    edge_index = torch.cat([base_data.edge_index, new_edges], dim=1)
    edge_weight = torch.cat([base_data.edge_weight, new_weights], dim=0)

    self_loop = torch.tensor([[new_node_idx], [new_node_idx]], dtype=torch.long)
    edge_index = torch.cat([edge_index, self_loop], dim=1)
    edge_weight = torch.cat([edge_weight, torch.tensor([1.0], dtype=torch.float)], dim=0)

    model.eval()
    with torch.no_grad():
        logits = model(augmented_x, edge_index, edge_weight=edge_weight)
        probs = torch.softmax(logits[new_node_idx], dim=0).numpy()

    probabilities = {LABEL_NAMES[i]: float(prob) for i, prob in enumerate(probs)}

    if not return_explanations:
        return probabilities

    modality_contributions = {
        modality: float(np.linalg.norm(vector))
        for modality, vector in modality_vectors.items()
    }
    total = sum(modality_contributions.values())
    if total > 0:
        modality_contributions = {k: v / total for k, v in modality_contributions.items()}

    neighbour_details: List[Dict[str, float]] = []
    if patient_table is not None:
        for dist, node_idx in zip(neighbour_dists, neighbour_ids):
            if node_idx >= len(patient_table):
                continue
            row = patient_table.iloc[int(node_idx)]
            neighbour_details.append(
                {
                    "RID": int(row["RID"]),
                    "VISCODE": str(row["VISCODE"]),
                    "Diagnosis": LABEL_NAMES.get(int(row["Label"]), "Unknown"),
                    "distance": float(dist),
                }
            )

    return {
        "probabilities": probabilities,
        "modality_contributions": modality_contributions,
        "nearest_neighbors": neighbour_details,
    }


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output_dir = Path("artifacts")
    artifacts = prepare_dataset(Path("Datasets"))
    model, history = train_gcn(artifacts.data)
    reports = summarise_performance(model, artifacts.data)
    print("Validation macro F1:", reports["val"]["macro avg"]["f1-score"])
    summary_df = pd.DataFrame(reports["test"]).round(3)
    print("Test metrics:")
    print(summary_df)

    subgroup_frame = artifacts.patient_table.copy()
    subgroup_frame["APOE4Carrier"] = (subgroup_frame.get("APOE4Count", 0) >= 1).astype(int)
    subgroup_frame["AgeBin"] = pd.cut(
        subgroup_frame["Age"],
        bins=[45, 55, 65, 75, 85, 100],
        right=False,
        labels=["45-54", "55-64", "65-74", "75-84", "85+"],
    )
    print("Subgroup metrics (Gender):")
    print(compute_subgroup_metrics(model, artifacts.data, subgroup_frame, split="test", group_by=["GenderBinary"]))
    print("Subgroup metrics (APOE4 carrier):")
    print(compute_subgroup_metrics(model, artifacts.data, subgroup_frame, split="test", group_by=["APOE4Carrier"]))
    print("Subgroup metrics (Age bin):")
    print(compute_subgroup_metrics(model, artifacts.data, subgroup_frame, split="test", group_by=["AgeBin"]))

    save_artifacts(output_dir, artifacts, model, history)
