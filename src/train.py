"""Train and compare Logistic Regression and Random Forest with MLflow tracking."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)

from src.data_processing import load_and_clean, split_features_target
from src.modeling import build_pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "heart_disease.csv"
ARTIFACTS = ROOT / "artifacts"


def main():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA}. Run: python -m src.download_dataset"
        )
    ARTIFACTS.mkdir(exist_ok=True)
    df = load_and_clean(DATA)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("uci-heart-disease-classification")
    candidates = {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
            {"classifier__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", random_state=42),
            {
                "classifier__n_estimators": [100, 250],
                "classifier__max_depth": [None, 5, 10],
            },
        ),
    }
    summaries = []
    best_cv_auc = -1.0
    best_pipeline = None
    best_name = None
    for name, (estimator, grid) in candidates.items():
        pipeline = build_pipeline(X_train, estimator)
        search = GridSearchCV(
            pipeline, grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True
        )
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("cv_folds", 5)
            search.fit(X_train, y_train)
            fitted = search.best_estimator_
            pred = fitted.predict(X_test)
            proba = fitted.predict_proba(X_test)[:, 1]
            metrics = {
                "cv_roc_auc_mean": float(search.best_score_),
                "test_accuracy": float(accuracy_score(y_test, pred)),
                "test_precision": float(precision_score(y_test, pred, zero_division=0)),
                "test_recall": float(recall_score(y_test, pred, zero_division=0)),
                "test_roc_auc": float(roc_auc_score(y_test, proba)),
            }
            mlflow.log_params(
                {f"best_{k}": str(v) for k, v in search.best_params_.items()}
            )
            mlflow.log_metrics(metrics)
            report = classification_report(
                y_test, pred, output_dict=True, zero_division=0
            )
            (ARTIFACTS / f"{name}_classification_report.json").write_text(
                json.dumps(report, indent=2)
            )
            ConfusionMatrixDisplay.from_predictions(y_test, pred)
            plt.tight_layout()
            cm_path = ARTIFACTS / f"{name}_confusion_matrix.png"
            plt.savefig(cm_path)
            plt.close()
            RocCurveDisplay.from_predictions(y_test, proba)
            plt.tight_layout()
            roc_path = ARTIFACTS / f"{name}_roc_curve.png"
            plt.savefig(roc_path)
            plt.close()
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(roc_path))
            mlflow.log_artifact(str(ARTIFACTS / f"{name}_classification_report.json"))
            summaries.append(
                {"model": name, **metrics, "best_params": search.best_params_}
            )
            if metrics["cv_roc_auc_mean"] > best_cv_auc:
                best_cv_auc, best_pipeline, best_name = (
                    metrics["cv_roc_auc_mean"],
                    fitted,
                    name,
                )
            mlflow.sklearn.log_model(fitted, artifact_path="model")
    joblib.dump(best_pipeline, ARTIFACTS / "model.joblib")
    (ARTIFACTS / "model_metadata.json").write_text(
        json.dumps(
            {
                "selected_model": best_name,
                "selection_metric": "cv_roc_auc_mean",
                "feature_names": X.columns.tolist(),
                "positive_class": 1,
                "note": "Educational demonstration only; not for clinical diagnosis.",
            },
            indent=2,
        )
    )
    (ARTIFACTS / "evaluation_summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))
    print(f"Selected {best_name}; saved pipeline to {ARTIFACTS / 'model.joblib'}")


if __name__ == "__main__":
    main()
