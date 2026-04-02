from pathlib import Path
from clearml import Dataset


PROJECT_NAME = "MLOPS DGA"
DATASET_NAME = "dga-sample"
DATASET_VERSION = "v1.0.0"
DATA_FILE = Path("data/dga_sample_100k.csv")


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_FILE}")

    dataset = Dataset.create(
        dataset_project=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        dataset_version=DATASET_VERSION,
    )

    dataset.add_files(path=str(DATA_FILE))
    dataset.upload()
    dataset.finalize()

    print("Dataset created successfully")
    print(f"Dataset ID: {dataset.id}")
    print(f"Dataset name: {DATASET_NAME}")
    print(f"Dataset version: {DATASET_VERSION}")


if __name__ == "__main__":
    main()
