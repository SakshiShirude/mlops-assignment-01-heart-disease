"""Generate EDA plots from the downloaded UCI dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.data_processing import load_and_clean

ROOT = Path(__file__).resolve().parents[1]


def save_overview(df, output: Path):
    """Save a compact, report-ready view of the key EDA plots and counts."""
    numeric = df.select_dtypes(include="number")
    fig = plt.figure(figsize=(14, 9), layout="constrained")
    fig.suptitle(
        "UCI Heart Disease: Exploratory Data Analysis", fontsize=18, weight="bold"
    )
    grid = fig.add_gridspec(2, 2)

    balance_ax = fig.add_subplot(grid[0, 0])
    counts = df["target"].value_counts().sort_index()
    balance_ax.bar(
        ["No disease", "Disease present"], counts.values, color=["#4472C4", "#ED7D31"]
    )
    balance_ax.bar_label(balance_ax.containers[0], padding=3)
    balance_ax.set_title("Binary target balance")
    balance_ax.set_ylabel("Records")
    balance_ax.tick_params(axis="x", labelrotation=10)

    hist_subfig = fig.add_subfigure(grid[0, 1])
    hist_subfig.suptitle("Selected feature distributions")
    hist_axes = hist_subfig.subplots(2, 3)
    selected = [
        name for name in ("age", "trestbps", "chol", "thalach", "oldpeak") if name in df
    ]
    for ax, column in zip(hist_axes.flat, selected):
        ax.hist(
            df[column].dropna().to_numpy(),
            bins=15,
            color="#70AD47",
            edgecolor="white",
        )
        ax.set_title(column)
        ax.set_ylabel("Count")
    for ax in hist_axes.flat[len(selected) :]:
        ax.set_visible(False)

    corr_ax = fig.add_subplot(grid[1, 0])
    sns.heatmap(
        numeric.corr(),
        ax=corr_ax,
        cmap="coolwarm",
        center=0,
        annot=False,
        cbar_kws={"shrink": 0.8},
    )
    corr_ax.set_title("Correlation across numeric columns")
    corr_ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    corr_ax.tick_params(axis="y", labelsize=8)

    summary_ax = fig.add_subplot(grid[1, 1])
    summary_ax.axis("off")
    summary = (
        f"Records: {len(df)}\n"
        f"Features: {df.shape[1] - 1}\n"
        f"Class 0 (no disease): {int(counts.get(0, 0))}\n"
        f"Class 1 (disease present): {int(counts.get(1, 0))}\n"
        f"Missing ca: {int(df['ca'].isna().sum()) if 'ca' in df else 0}\n"
        f"Missing thal: {int(df['thal'].isna().sum()) if 'thal' in df else 0}"
    )
    summary_ax.text(
        0.08,
        0.82,
        "Dataset snapshot",
        fontsize=15,
        weight="bold",
        transform=summary_ax.transAxes,
    )
    summary_ax.text(
        0.08,
        0.68,
        summary,
        fontsize=13,
        va="top",
        linespacing=1.8,
        transform=summary_ax.transAxes,
    )

    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    data_path = ROOT / "data" / "heart_disease.csv"
    out = ROOT / "artifacts" / "eda"
    out.mkdir(parents=True, exist_ok=True)
    df = load_and_clean(data_path)
    df.describe(include="all").to_csv(out / "descriptive_statistics.csv")
    df["target"].value_counts().sort_index().plot(kind="bar", rot=0)
    plt.title("Heart disease class balance")
    plt.xlabel("Target (0=no disease, 1=disease)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out / "class_balance.png")
    plt.close()
    numeric = df.select_dtypes(include="number")
    numeric.hist(figsize=(14, 10), bins=20)
    plt.tight_layout()
    plt.savefig(out / "histograms.png")
    plt.close()
    plt.figure(figsize=(12, 9))
    sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, annot=False)
    plt.title("Numeric feature correlation heatmap")
    plt.tight_layout()
    plt.savefig(out / "correlation_heatmap.png")
    plt.close()
    save_overview(df, out / "eda_overview.png")
    print(f"EDA artifacts saved in {out}")


if __name__ == "__main__":
    main()
