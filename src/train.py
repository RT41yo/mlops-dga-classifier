from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from clearml import Dataset, Task
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_NAME = "MLOPS DGA"
TASK_NAME = "train_dga_baseline"

DATASET_ID = "1fb5ea76060e457b876dca5592b9904b"

TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 5000
NGRAM_RANGE = (3, 5)
C = 1.0
MAX_ITER = 300

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(dataset_id: str) -> pd.DataFrame:
    dataset = Dataset.get(dataset_id=dataset_id, alias="training_data")
    local_path = Path(dataset.get_local_copy())

    csv_files = list(local_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in dataset path: {local_path}")

    df = pd.read_csv(csv_files[0])
    return df


def save_confusion_matrix(y_true, y_pred, output_path: Path):
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

    task.execute_remotely(queue_name="students", exit_process=True)

    task.connect(
        {
            "dataset_id": DATASET_ID,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "max_features": MAX_FEATURES,
            "ngram_range": NGRAM_RANGE,
            "C": C,
            "max_iter": MAX_ITER,
        }
    )

    logger = task.get_logger()

    df = load_dataset(DATASET_ID)
    df = df.dropna(subset=["domain", "label"]).copy()
    df["domain"] = df["domain"].astype(str).str.strip()
    df = df[df["domain"] != ""]

    X_train, X_test, y_train, y_test = train_test_split(
        df["domain"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=NGRAM_RANGE,
                    max_features=MAX_FEATURES,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=MAX_ITER,
                    random_state=RANDOM_STATE,
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

    task.upload_artifact(
        name="model_pipeline",
        artifact_object=str(model_path),
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1: {f1:.4f}")

    task.close()


if __name__ == "__main__":
    main()
