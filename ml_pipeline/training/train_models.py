import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

FEATURES_CSV = "data/processed/quality_measurements/extracted_features.csv"
MODELS_DIR = "models"
LOGS_DIR = "logs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

NON_FEATURE_COLS = {
    "sample_id",
    "sequence_id",
    "category",
    "defect_type",
    "readiness_label",
    "baseline_3d_points",
}


def get_models_and_search_spaces(is_binary: bool, random_state: int = 42):
    """Return baseline models (single-threaded per worker to prevent OpenMP thrashing) and search spaces."""
    xgb_kwargs = {
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 1.5,
        "objective": "binary:logistic" if is_binary else "multi:softprob",
        "eval_metric": "logloss" if is_binary else "mlogloss",
        "random_state": random_state,
        "n_jobs": 1,
        "verbosity": 0,
    }
    if is_binary:
        xgb_kwargs["scale_pos_weight"] = 6.0

    baselines = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
        "XGBoost": XGBClassifier(**xgb_kwargs),
        "LightGBM": LGBMClassifier(
            n_estimators=150,
            max_depth=4,
            num_leaves=15,
            learning_rate=0.05,
            min_child_samples=10,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.5,
            importance_type="gain",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
            verbose=-1,
        ),
    }

    param_spaces = {
        "RandomForest": {
            "n_estimators": [150, 250, 350],
            "max_depth": [5, 7, 9, None],
            "min_samples_split": [2, 4, 6],
            "min_samples_leaf": [1, 2, 3],
            "max_features": ["sqrt", "log2", 0.7],
        },
        "XGBoost": {
            "n_estimators": [100, 150, 220],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.03, 0.05, 0.08],
            "subsample": [0.75, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "reg_alpha": [0.0, 0.5, 1.0],
            "reg_lambda": [1.0, 2.0, 3.0],
        },
        "LightGBM": {
            "n_estimators": [100, 150, 220],
            "max_depth": [3, 4, 5],
            "num_leaves": [7, 15, 23],
            "learning_rate": [0.03, 0.05, 0.08],
            "min_child_samples": [6, 10, 15],
            "subsample": [0.75, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "reg_lambda": [1.0, 2.0, 3.0],
        },
    }

    return baselines, param_spaces


def score_candidate(estimator, X, y_encoded, groups, sgkf, train_idx, test_idx, stage, model_name, task_name, class_names):
    """Evaluate a model with 5-fold Group CV and the unseen object holdout split."""
    cv_scores = cross_validate(
        clone(estimator),
        X,
        y_encoded,
        cv=sgkf,
        groups=groups,
        scoring=["accuracy", "f1_macro"],
        n_jobs=2,
    )

    holdout_est = clone(estimator)
    holdout_est.fit(X.iloc[train_idx], y_encoded[train_idx])
    y_pred = holdout_est.predict(X.iloc[test_idx])
    y_test = y_encoded[test_idx]

    # Refit on full dataset for final deployment candidate
    final_est = clone(estimator)
    final_est.fit(X, y_encoded)

    record = {
        "task": task_name,
        "stage": stage,
        "model": model_name,
        "cv_accuracy_mean": round(float(np.mean(cv_scores["test_accuracy"])), 4),
        "cv_accuracy_std": round(float(np.std(cv_scores["test_accuracy"])), 4),
        "cv_macro_f1_mean": round(float(np.mean(cv_scores["test_f1_macro"])), 4),
        "holdout_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "holdout_macro_f1": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "y_test": y_test,
        "y_pred": y_pred,
        "class_names": class_names,
    }
    return record, final_est


def run_experiment(X: pd.DataFrame, y_encoded: np.ndarray, groups: pd.Series, class_names: list, task_name: str):
    is_binary = len(class_names) == 2
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y_encoded, groups))

    baselines, param_spaces = get_models_and_search_spaces(is_binary=is_binary)
    all_records = []
    fitted_candidates = {}
    tuned_estimators_for_voting = []
    best_params_log = {}

    # 1. Evaluate Baseline Models
    for name, base_est in baselines.items():
        print(f"[{task_name}] Evaluating baseline {name}...")
        rec, final_est = score_candidate(
            base_est, X, y_encoded, groups, sgkf, train_idx, test_idx,
            stage="1_Baseline", model_name=name, task_name=task_name, class_names=class_names
        )
        all_records.append(rec)
        fitted_candidates[f"Baseline_{name}"] = final_est

    # 2. Hyperparameter Tuning via RandomizedSearchCV (Group-Aware)
    for name, base_est in baselines.items():
        print(f"[{task_name}] Tuning {name}...")
        search = RandomizedSearchCV(
            estimator=clone(base_est),
            param_distributions=param_spaces[name],
            n_iter=12,
            scoring="f1_macro",
            cv=sgkf,
            random_state=42,
            n_jobs=2,
            refit=True,
        )
        search.fit(X, y_encoded, groups=groups)
        best_est = search.best_estimator_
        best_params_log[name] = search.best_params_

        rec, final_est = score_candidate(
            best_est, X, y_encoded, groups, sgkf, train_idx, test_idx,
            stage="2_Tuned", model_name=f"{name}_Tuned", task_name=task_name, class_names=class_names
        )
        all_records.append(rec)
        fitted_candidates[f"Tuned_{name}"] = final_est
        tuned_estimators_for_voting.append((name.lower(), clone(best_est)))

    # 3. Soft-Voting Ensemble combining Tuned RF + XGBoost + LightGBM
    print(f"[{task_name}] Evaluating Soft-Voting Ensemble...")
    voting_clf = VotingClassifier(
        estimators=tuned_estimators_for_voting,
        voting="soft",
        n_jobs=1,
    )
    ens_rec, final_ensemble = score_candidate(
        voting_clf, X, y_encoded, groups, sgkf, train_idx, test_idx,
        stage="3_Ensemble", model_name="SoftVoting_Ensemble", task_name=task_name, class_names=class_names
    )
    all_records.append(ens_rec)
    fitted_candidates["SoftVoting_Ensemble"] = final_ensemble

    # Select overall best candidate by CV Macro F1 (tie-break by Holdout Macro F1)
    best_row = max(all_records, key=lambda r: (r["cv_macro_f1_mean"], r["holdout_macro_f1"]))
    key_lookup = (
        "SoftVoting_Ensemble"
        if best_row["model"] == "SoftVoting_Ensemble"
        else f"{best_row['stage'].split('_')[1]}_{best_row['model'].replace('_Tuned', '')}"
    )
    winner_model = fitted_candidates[key_lookup]

    return all_records, winner_model, best_row, final_ensemble, best_params_log


def compute_consensus_feature_importance(final_ensemble: VotingClassifier, feature_cols: list) -> pd.Series:
    """Average normalized gain/impurity feature importances across all 3 ensemble estimators."""
    norm_importances = []
    for est in final_ensemble.estimators_:
        imp = np.asarray(est.feature_importances_, dtype=float)
        imp_sum = imp.sum()
        if imp_sum > 0:
            norm_importances.append(imp / imp_sum)
    mean_imp = np.mean(norm_importances, axis=0)
    return pd.Series(mean_imp, index=feature_cols).sort_values(ascending=False)


def main():
    df = pd.read_csv(FEATURES_CSV)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols]
    groups = df["sequence_id"]

    print("Running 5-Fold Group CV + Hyperparameter Tuning + Soft-Voting Ensemble...")

    # Task 1: 7-Class Defect Diagnosis
    defect_le = LabelEncoder()
    y_defect = defect_le.fit_transform(df["defect_type"])
    defect_records, best_defect_model, best_defect_row, defect_ensemble, defect_params = run_experiment(
        X, y_defect, groups, list(defect_le.classes_), "Defect_Diagnosis_7Class"
    )

    # Task 2: Binary Readiness Classification (GOOD=1 vs POOR=0)
    y_binary = (df["readiness_label"] == "GOOD").astype(int).values
    binary_records, best_binary_model, best_binary_row, _, binary_params = run_experiment(
        X, y_binary, groups, ["POOR", "GOOD"], "Readiness_Binary"
    )

    # Save full experiment comparison table
    summary_rows = [
        {k: v for k, v in r.items() if k not in {"y_test", "y_pred", "class_names"}}
        for r in (defect_records + binary_records)
    ]
    df_summary = pd.DataFrame(summary_rows)
    comp_csv = os.path.join(LOGS_DIR, "model_comparison.csv")
    df_summary.to_csv(comp_csv, index=False)

    print("\n=== Complete Model Benchmark (Baseline vs. Tuned vs. Soft-Voting Ensemble) ===")
    print(df_summary.to_string(index=False))

    print(
        f"\nWinning Defect Diagnosis Model: {best_defect_row['model']} "
        f"(5-Fold CV Macro F1: {best_defect_row['cv_macro_f1_mean']}, Holdout Acc: {best_defect_row['holdout_accuracy']})"
    )
    print("\nUnseen Object Holdout Classification Report (7-Class Defect Diagnosis):")
    print(
        classification_report(
            best_defect_row["y_test"],
            best_defect_row["y_pred"],
            target_names=best_defect_row["class_names"],
            digits=3,
        )
    )

    importances = compute_consensus_feature_importance(defect_ensemble, feature_cols)
    print("Top 10 Consensus Feature Importances (Across Tuned RF + XGBoost + LightGBM):")
    print(importances.head(10).round(4).to_string())

    # Save production model bundle
    bundle = {
        "feature_columns": feature_cols,
        "defect_model_name": best_defect_row["model"],
        "defect_model": best_defect_model,
        "defect_label_encoder": defect_le,
        "binary_model_name": best_binary_row["model"],
        "binary_model": best_binary_model,
        "feature_importances": importances.to_dict(),
        "tuned_hyperparameters": {
            "defect_diagnosis": defect_params,
            "readiness_binary": binary_params,
        },
    }
    bundle_path = os.path.join(MODELS_DIR, "preflight_model_bundle.joblib")
    joblib.dump(bundle, bundle_path)

    with open(os.path.join(MODELS_DIR, "feature_schema.json"), "w") as f:
        json.dump(
            {
                "feature_columns": feature_cols,
                "defect_classes": list(defect_le.classes_),
                "best_defect_model": best_defect_row["model"],
                "best_binary_model": best_binary_row["model"],
                "tuned_hyperparameters": bundle["tuned_hyperparameters"],
            },
            f,
            indent=2,
        )

    print(f"\nSaved production model bundle to: {bundle_path}")


if __name__ == "__main__":
    main()