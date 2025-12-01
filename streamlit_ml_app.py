"""Interactive Streamlit lab for supervised machine learning concepts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path("files/cardio_train.csv")
TARGET_COLUMN = "cardio"
DISPLAY_TARGET = {0: "No disease", 1: "Cardio condition"}
DEFAULT_FEATURES = [
    "age_years",
    "gender",
    "bmi",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "active",
]
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "age_years": "Age converted from days to years; used throughout the lab.",
    "height": "Height in centimeters.",
    "weight": "Weight in kilograms.",
    "ap_hi": "Systolic arterial pressure.",
    "ap_lo": "Diastolic arterial pressure.",
    "pulse_pressure": "Difference between systolic and diastolic pressure.",
    "bmi": "Body mass index computed from height and weight.",
    "cholesterol": "Cholesterol category (1 normal, 3 very high).",
    "gluc": "Glucose category (1 normal, 3 very high).",
    "smoke": "Smoking status flag.",
    "alco": "Alcohol consumption flag.",
    "active": "Physical activity flag.",
    "gender": "1 = female, 2 = male per the original dataset.",
    "cardio_label": "Human-readable label derived from the target column.",
}


@st.cache_data(show_spinner=False)
def load_cardio_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and augment the cardio dataset once per session."""

    if not path.exists():
        raise FileNotFoundError(
            f"Could not locate cardio dataset at '{path}'. Ensure files/cardio_train.csv is present."
        )

    try:
        df = pd.read_csv(path, sep=";")
    except Exception as exc:  # pragma: no cover - defensive branch for corrupted CSVs
        raise RuntimeError("Failed to read cardio dataset. Please verify the CSV format.") from exc
    df["age_years"] = (df["age"] / 365.25).round(1)
    height_m = df["height"] / 100
    df["bmi"] = (df["weight"] / (height_m**2)).round(2)
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    df["cardio_label"] = df[TARGET_COLUMN].map(DISPLAY_TARGET)
    return df


def list_feature_columns(df: pd.DataFrame) -> List[str]:
    ignore = {"id", TARGET_COLUMN, "cardio_label", "age"}
    return [col for col in df.columns if col not in ignore]


def class_balance_chart(df: pd.DataFrame) -> go.Figure:
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    fig = go.Figure(
        data=[
            go.Bar(
                x=[DISPLAY_TARGET[idx] for idx in counts.index],
                y=counts.values,
                marker_color=["#4caf50", "#f44336"],
            )
        ]
    )
    fig.update_layout(title="Class balance", yaxis_title="Patient count")
    return fig


def correlation_heatmap(df: pd.DataFrame, features: List[str]) -> go.Figure:
    corr = df[features + [TARGET_COLUMN]].corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="ρ"),
        )
    )
    fig.update_layout(height=480, title="Feature correlation heatmap")
    return fig


def build_model(model_type: str, params: Dict) -> Pipeline:
    if model_type == "Logistic Regression":
        clf = LogisticRegression(C=params["C"], max_iter=1000)
        return Pipeline([("scaler", StandardScaler()), ("model", clf)])
    if model_type == "Decision Tree":
        clf = DecisionTreeClassifier(
            max_depth=params["max_depth"], min_samples_leaf=params["min_samples_leaf"], random_state=params["random_state"]
        )
        return Pipeline([("model", clf)])
    if model_type == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=params["random_state"],
        )
        return Pipeline([("model", clf)])
    raise ValueError(f"Unsupported model type: {model_type}")


def evaluate_model(model: Pipeline, df: pd.DataFrame, features: List[str], test_size: float, random_state: int) -> Dict:
    X = df[features]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-9)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    return {
        "model": model,
        "metrics": metrics,
        "confusion": cm,
        "roc": (fpr, tpr),
        "thresholds": thresholds,
        "y_test": y_test,
        "probs": y_prob,
        "preds": y_pred,
        "features": features,
    }


def confusion_matrix_fig(cm: np.ndarray) -> go.Figure:
    labels = ["Actual 0", "Actual 1"]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Pred 0", "Pred 1"],
            y=labels,
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues",
        )
    )
    fig.update_layout(title="Confusion matrix", xaxis_title="Predicted", yaxis_title="Actual")
    return fig


def roc_curve_fig(fpr: np.ndarray, tpr: np.ndarray, auc_score: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Model", line=dict(color="#ff9800")))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))
    fig.update_layout(title=f"ROC curve (AUC = {auc_score:.3f})", xaxis_title="False positive rate", yaxis_title="True positive rate")
    return fig


def dataset_overview_tab(df: pd.DataFrame) -> None:
    st.info("Purpose: inspect dataset shape, class balance, and summary statistics before modeling.")
    cols = st.columns(3)
    cols[0].metric("Records", f"{len(df):,}")
    cols[1].metric("Features", f"{len(list_feature_columns(df))}")
    cols[2].metric("Positive rate", f"{df[TARGET_COLUMN].mean():.2%}")

    st.plotly_chart(class_balance_chart(df), use_container_width=True, key="dataset_class_balance")
    st.caption("This bar shows whether the lab must handle class imbalance (≈50/50 is ideal for quick demos).")

    st.subheader("Summary statistics")
    numeric_cols = [col for col in list_feature_columns(df) if pd.api.types.is_numeric_dtype(df[col])]
    st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

    st.subheader("Peek at the raw rows")
    show_cols = st.multiselect(
        "Columns to display",
        options=list_feature_columns(df) + [TARGET_COLUMN, "cardio_label"],
        default=["age_years", "gender", "ap_hi", "ap_lo", "cholesterol", TARGET_COLUMN],
    )
    st.dataframe(df[show_cols].head(20))

    with st.expander("Download processed dataset"):
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="cardio_lab_dataset.csv",
            mime="text/csv",
        )


def feature_explorer_tab(df: pd.DataFrame) -> None:
    st.info("Purpose: relate single features or feature pairs to the cardio outcome interactively.")
    available_features = list_feature_columns(df)
    plot_type = st.radio("Plot type", ["Histogram", "Scatter", "Box"], horizontal=True)
    if plot_type == "Histogram":
        feature = st.selectbox("Feature", available_features, index=available_features.index("age_years"))
        fig = px.histogram(
            df,
            x=feature,
            color="cardio_label",
            barmode="overlay",
            nbins=40,
            color_discrete_sequence=["#4caf50", "#f44336"],
        )
        fig.update_layout(legend_title="Diagnosis")
        st.plotly_chart(fig, use_container_width=True, key="feature_histogram")
    elif plot_type == "Scatter":
        x_feature = st.selectbox("X-axis", available_features, index=available_features.index("age_years"))
        y_feature = st.selectbox("Y-axis", available_features, index=available_features.index("ap_hi"))
        fig = px.scatter(
            df,
            x=x_feature,
            y=y_feature,
            color="cardio_label",
            opacity=0.55,
            color_discrete_sequence=["#4caf50", "#f44336"],
        )
        st.plotly_chart(fig, use_container_width=True, key="feature_scatter")
    else:
        feature = st.selectbox("Feature", available_features, index=available_features.index("cholesterol"))
        fig = px.box(
            df,
            x="cardio_label",
            y=feature,
            color="cardio_label",
            color_discrete_sequence=["#4caf50", "#f44336"],
        )
        st.plotly_chart(fig, use_container_width=True, key="feature_boxplot")

    with st.expander("Correlation heatmap"):
        corr_features = st.multiselect(
            "Select up to 8 columns",
            options=available_features,
            default=available_features[:6],
            max_selections=8,
        )
        if corr_features:
            st.plotly_chart(
                correlation_heatmap(df, corr_features),
                use_container_width=True,
                key="feature_corr_heatmap",
            )
        else:
            st.warning("Pick at least one column to compute correlations.")


def model_playground_tab(df: pd.DataFrame) -> None:
    st.info("Purpose: choose features, train a classifier, and study headline metrics.")
    available_features = list_feature_columns(df)
    selected = st.multiselect(
        "Input features",
        options=available_features,
        default=[feat for feat in DEFAULT_FEATURES if feat in available_features],
    )
    if not selected:
        st.warning("Select at least one feature to train a model.")
        return

    cols = st.columns(3)
    model_type = cols[0].selectbox("Model", ["Logistic Regression", "Decision Tree", "Random Forest"])
    test_size = cols[1].slider("Test size", 0.1, 0.4, 0.2, step=0.05)
    random_state = cols[2].number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

    params: Dict[str, float | int] = {"random_state": int(random_state)}
    if model_type == "Logistic Regression":
        log_c = cols[0].slider("log10(C)", -2.0, 2.0, 0.0, step=0.1)
        params["C"] = float(10 ** log_c)
    elif model_type == "Decision Tree":
        params["max_depth"] = cols[0].slider("Max depth", 2, 20, 6)
        params["min_samples_leaf"] = cols[1].slider("Min samples leaf", 1, 20, 3)
    else:
        params["n_estimators"] = cols[0].slider("Trees", 50, 400, 200, step=10)
        params["max_depth"] = cols[1].slider("Max depth", 2, 20, 8)
        params["min_samples_leaf"] = cols[2].slider("Min samples leaf", 1, 20, 2)

    model = build_model(model_type, params)
    with st.spinner("Training model..."):
        results = evaluate_model(model, df, selected, test_size, int(random_state))
    st.success("Training complete!")

    st.session_state["latest_run"] = {
        "y_test": results["y_test"],
        "probs": results["probs"],
        "features": selected,
        "model_name": model_type,
    }

    metric_cols = st.columns(5)
    metric_cols[0].metric("Accuracy", f"{results['metrics']['accuracy']:.3f}")
    metric_cols[1].metric("Precision", f"{results['metrics']['precision']:.3f}")
    metric_cols[2].metric("Recall", f"{results['metrics']['recall']:.3f}")
    metric_cols[3].metric("F1", f"{results['metrics']['f1']:.3f}")
    metric_cols[4].metric("ROC AUC", f"{results['metrics']['roc_auc']:.3f}")

    charts = st.tabs(["Confusion Matrix", "ROC Curve", "Feature Influence"])
    with charts[0]:
        st.plotly_chart(
            confusion_matrix_fig(results["confusion"]),
            use_container_width=True,
            key="model_confusion_matrix",
        )
    with charts[1]:
        fpr, tpr = results["roc"]
        st.plotly_chart(
            roc_curve_fig(fpr, tpr, results["metrics"]["roc_auc"]),
            use_container_width=True,
            key="model_roc_curve",
        )
    with charts[2]:
        if hasattr(results["model"].named_steps["model"], "feature_importances_"):
            importances = results["model"].named_steps["model"].feature_importances_
            fig = px.bar(x=selected, y=importances, labels={"x": "Feature", "y": "Importance"})
            st.plotly_chart(fig, use_container_width=True, key="model_feature_importance")
        elif hasattr(results["model"].named_steps["model"], "coef_"):
            coefs = results["model"].named_steps["model"].coef_[0]
            fig = px.bar(x=selected, y=coefs, labels={"x": "Feature", "y": "Coefficient"})
            st.plotly_chart(fig, use_container_width=True, key="model_feature_coefficients")
        else:
            st.info("This estimator does not expose feature influence metrics.")


def threshold_lab_tab() -> None:
    st.info("Purpose: slide the classification threshold to visualize the precision/recall trade-off.")
    latest = st.session_state.get("latest_run")
    if not latest:
        st.warning("Train a model in the previous tab to unlock threshold tuning.")
        return

    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.5, step=0.01)
    y_test = latest["y_test"]
    probs = latest["probs"]
    preds = (probs >= threshold).astype(int)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{accuracy:.3f}")
    cols[1].metric("Precision", f"{precision:.3f}")
    cols[2].metric("Recall", f"{recall:.3f}")
    cols[3].metric("F1", f"{f1:.3f}")

    cm = confusion_matrix(y_test, preds)
    st.plotly_chart(confusion_matrix_fig(cm), use_container_width=True, key="threshold_confusion_matrix")

    thresholds = np.linspace(0.05, 0.95, 50)
    precision_vals, recall_vals = [], []
    for thr in thresholds:
        temp_preds = (probs >= thr).astype(int)
        precision_vals.append(precision_score(y_test, temp_preds, zero_division=0))
        recall_vals.append(recall_score(y_test, temp_preds, zero_division=0))
    tradeoff_fig = go.Figure()
    tradeoff_fig.add_trace(go.Scatter(x=thresholds, y=precision_vals, name="Precision"))
    tradeoff_fig.add_trace(go.Scatter(x=thresholds, y=recall_vals, name="Recall"))
    tradeoff_fig.add_vline(x=threshold, line=dict(color="#f44336", dash="dash"))
    tradeoff_fig.update_layout(
        title=f"Precision/Recall vs. Threshold ({latest['model_name']})",
        xaxis_title="Threshold",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
    )
    st.plotly_chart(tradeoff_fig, use_container_width=True, key="threshold_tradeoff_curve")


def render_sidebar(df: pd.DataFrame) -> None:
    st.sidebar.title("Supervised ML Lab")
    st.sidebar.write("Follow the checklist to stay grounded while exploring models.")
    st.sidebar.checkbox("1. Inspect data", value=True, disabled=True)
    st.sidebar.checkbox("2. Explore features", value=True, disabled=True)
    st.sidebar.checkbox("3. Train model", value=True, disabled=True)
    st.sidebar.checkbox("4. Tune threshold", value=True, disabled=True)
    st.sidebar.divider()
    st.sidebar.write("Feature glossary")
    for feature in FEATURE_DESCRIPTIONS:
        with st.sidebar.expander(feature):
            st.write(FEATURE_DESCRIPTIONS[feature])


def main() -> None:
    st.set_page_config(page_title="Supervised ML Lab", layout="wide")
    try:
        df = load_cardio_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    render_sidebar(df)

    tabs = st.tabs([
        "Dataset Tour",
        "Feature Explorer",
        "Model Playground",
        "Threshold Lab",
    ])
    with tabs[0]:
        dataset_overview_tab(df)
    with tabs[1]:
        feature_explorer_tab(df)
    with tabs[2]:
        model_playground_tab(df)
    with tabs[3]:
        threshold_lab_tab()


if __name__ == "__main__":
    main()
