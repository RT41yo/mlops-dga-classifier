from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from clearml import Dataset, Task, OutputModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_NAME = "MLOPS DGA"
TASK_NAME = "train_dga_baseline"

DEFAULT_PARAMS = {
    "dataset_id": "1fb5ea76060e457b876dca5592b9904b",
    "test_size": 0.2,
    "random_state": 42,
    "max_features": 10000,
    "ngram_min": 2,
    "ngram_max": 5,
    "C": 1.0,
    "max_iter": 300,
}

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(dataset_id: str) -> pd.DataFrame:
    dataset = Dataset.get(dataset_id=dataset_id, alias="training_data")
    local_path = Path(dataset.get_local_copy())
    csv_files = list(local_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in dataset path: {local_path}")
    return pd.read_csv(csv_files[0])


def save_confusion_matrix(y_true, y_pred, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    task = Task.init(
        project_name=PROJECT_NAME,
        task_name=TASK_NAME,
        task_type=Task.TaskTypes.training,
    )
    task.output_uri = "http://192.168.1.50:8081"

    params = task.connect(DEFAULT_PARAMS)
    task.execute_remotely(queue_name="students", exit_process=True)

    logger = task.get_logger()

    df = load_dataset(params["dataset_id"])
    df = df.dropna(subset=["domain", "label"]).copy()
    df["domain"] = df["domain"].astype(str).str.strip()
    df = df[df["domain"] != ""]

    X_train, X_test, y_train, y_test = train_test_split(
        df["domain"],
        df["label"],
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=df["label"],
    )

    ngram_range = (params["ngram_min"], params["ngram_max"])

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=ngram_range,
                    max_features=params["max_features"],
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=params["C"],
                    max_iter=params["max_iter"],
                    random_state=params["random_state"],
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logger.report_scalar("metrics", "accuracy", value=accuracy, iteration=0)
    logger.report_scalar("metrics", "f1", value=f1, iteration=0)

    cm_path = ARTIFACTS_DIR / "confusion_matrix.png"
    save_confusion_matrix(y_test, y_pred, cm_path)
    logger.report_image(
        title="confusion_matrix",
        series="validation",
        local_path=str(cm_path),
        iteration=0,
    )

    model_path = ARTIFACTS_DIR / "dga_pipeline.joblib"
    joblib.dump(model, model_path)

    output_model = OutputModel(
        task=task,
        name="dga-char-tfidf-lr",
        tags=["dga", "sklearn", "baseline", "best-candidate", "v1"],
        comment=f"Char TF-IDF + LogisticRegression, f1={f1:.4f}, acc={accuracy:.4f}",
        framework="scikit-learn",
    )

    output_model.update_weights(
        weights_filename=str(model_path),
        target_filename="dga_pipeline.joblib",
        auto_delete_file=False,
    )
    OutputModel.wait_for_uploads()

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1: {f1:.4f}")

    task.close()


if __name__ == "__main__":
    main()
