"""Download the UCI Heart Disease dataset using ucimlrepo (dataset ID 45)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def download_dataset(output: Path | None = None) -> Path:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install -r requirements.txt") from exc
    dataset = fetch_ucirepo(id=45)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    if targets.shape[1] == 0:
        raise ValueError("UCI dataset returned no target column")
    target_name = targets.columns[0]
    frame = features.copy()
    frame["target"] = targets[target_name]
    output = output or ROOT / "data" / "heart_disease.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Saved {len(frame)} rows to {output}")
    return output

if __name__ == "__main__":
    download_dataset()
