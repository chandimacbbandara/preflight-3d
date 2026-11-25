import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, classification_report

FEATURES_CSV = "data/processed/quality_measurements/extracted_features.csv"
BUNDLE_PATH = "models/preflight_model_bundle.joblib"
COMP_CSV = "logs/model_comparison.csv"
REPORT_MD = "logs/model_evaluation_report.md"


def df_to_md(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def main():
    df = pd.read_csv(FEATURES_CSV)
    df_comp = pd.read_csv(COMP_CSV)
    bundle = joblib.load(BUNDLE_PATH)

    feature_cols = bundle["feature_columns"]
    defect_le = bundle["defect_label_encoder"]
    X = df[feature_cols]
    y = defect_le.transform(df["defect_type"])
    groups = df["sequence_id"]

    # Evaluate on the unseen 8-object holdout split using a fresh clone of the winning algorithm
    from sklearn.base import clone
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y, groups))

    eval_clf = clone(bundle["defect_model"])
    eval_clf.fit(X.iloc[train_idx], y[train_idx])
    y_pred = eval_clf.predict(X.iloc[test_idx])
    y_test = y[test_idx]

    classes = list(defect_le.classes_)
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=[f"True: {c}" for c in classes], columns=classes).reset_index()
    cm_df.rename(columns={"index": "Actual \\ Predicted"}, inplace=True)

    imp_df = (
        pd.Series(bundle["feature_importances"])
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "Feature", 0: "Importance"})
        .round(4)
    )

    report_txt = classification_report(y_test, y_pred, target_names=classes, digits=3)

    md_lines = [
        "# PreFlight-3D: Model Training & Evaluation Report",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Validation Strategy:** 5-Fold `StratifiedGroupKFold` grouped by `sequence_id` (zero object leakage)",
        "",
        "---",
        "",
        "## 1. Model Benchmark Comparison",
        df_to_md(df_comp),
        "",
        f"**Selected Defect Diagnosis Model:** `{bundle['defect_model_name']}`  ",
        f"**Selected Readiness Score Model:** `{bundle['binary_model_name']}`",
        "",
        "---",
        "",
        "## 2. Unseen Object Holdout Classification Report (56 Test Videos)",
        "```text",
        report_txt.strip(),
        "```",
        "",
        "## 3. Holdout Confusion Matrix",
        df_to_md(cm_df),
        "",
        "---",
        "",
        "## 4. Feature Importance Ranking",
        df_to_md(imp_df),
        "",
    ]

    with open(REPORT_MD, "w") as f:
        f.write("\n".join(md_lines))

    print(f"Saved evaluation report to: {REPORT_MD}")


if __name__ == "__main__":
    main()