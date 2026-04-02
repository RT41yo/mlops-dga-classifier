from pathlib import Path
import pandas as pd


SOURCE_PATH = Path("data/raw/train.csv")
OUTPUT_PATH = Path("data/dga_sample_100k.csv")
SAMPLE_SIZE_PER_CLASS = 50_000
RANDOM_STATE = 42


def main():
    df = pd.read_csv(SOURCE_PATH)

    df = df[["domain", "label"]].copy()
    df = df.dropna(subset=["domain", "label"])
    df["domain"] = df["domain"].astype(str).str.strip()
    df = df[df["domain"] != ""]

    parts = []
    for label_value, group in df.groupby("label"):
        n = min(len(group), SAMPLE_SIZE_PER_CLASS)
        parts.append(group.sample(n=n, random_state=RANDOM_STATE))

    result = pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved sample to: {OUTPUT_PATH}")
    print(result.shape)
    print(result["label"].value_counts())


if __name__ == "__main__":
    main()
