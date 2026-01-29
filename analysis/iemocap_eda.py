from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=200)
    plt.close(fig)


def _bar_plot(series: pd.Series, title: str, xlabel: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    series.plot(kind="bar", ax=ax, color="#2f4b7c")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    return fig


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="IEMOCAP dataset quick EDA figures")
    parser.add_argument(
        "--csv",
        type=Path,
        default=repo_root / "iemocap_full_dataset.csv",
        help="Path to iemocap_full_dataset.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root / "analysis" / "figures",
        help="Output directory for figures",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
    df["method"] = df["method"].astype(str).str.strip().str.lower()
    df["gender"] = df["gender"].astype(str).str.strip().str.upper()

    df_valid = df.copy()

    # Emotion distribution
    emotion_counts = df_valid["emotion"].value_counts()
    fig = _bar_plot(
        emotion_counts,
        "Emotion Class Distribution (including 'xxx')",
        "Emotion",
        "Count",
    )
    _save_fig(fig, args.out, "emotion_distribution_bar")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        emotion_counts.values,
        labels=emotion_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 9},
    )
    ax.set_title("Emotion Class Proportions (including 'xxx')")
    _save_fig(fig, args.out, "emotion_distribution_pie")

    # Gender distribution
    gender_counts = df_valid["gender"].value_counts()
    fig = _bar_plot(gender_counts, "Gender Distribution", "Gender", "Count")
    _save_fig(fig, args.out, "gender_distribution_bar")

    # Method distribution
    method_counts = df_valid["method"].value_counts()
    fig = _bar_plot(method_counts, "Method Distribution", "Method", "Count")
    _save_fig(fig, args.out, "method_distribution_bar")

    # Session distribution
    session_counts = df_valid["session"].value_counts().sort_index()
    fig = _bar_plot(session_counts, "Session Distribution", "Session", "Count")
    _save_fig(fig, args.out, "session_distribution_bar")

    # Emotion by gender (stacked bar)
    gender_emotion = pd.crosstab(df_valid["gender"], df_valid["emotion"])
    fig, ax = plt.subplots(figsize=(9, 5))
    gender_emotion.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Emotion by Gender (stacked)")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Count")
    ax.legend(title="Emotion", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, args.out, "emotion_by_gender_stacked")

    # Emotion by method (stacked bar)
    method_emotion = pd.crosstab(df_valid["method"], df_valid["emotion"])
    fig, ax = plt.subplots(figsize=(9, 5))
    method_emotion.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Emotion by Method (stacked)")
    ax.set_xlabel("Method")
    ax.set_ylabel("Count")
    ax.legend(title="Emotion", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, args.out, "emotion_by_method_stacked")

    # Agreement distribution
    fig = _bar_plot(
        df["agreement"].value_counts().sort_index(),
        "Agreement Count Distribution",
        "Agreement",
        "Count",
    )
    _save_fig(fig, args.out, "agreement_distribution_bar")

    # Annotator count distribution
    fig = _bar_plot(
        df["n_annotators"].value_counts().sort_index(),
        "Annotator Count Distribution",
        "Number of Annotators",
        "Count",
    )
    _save_fig(fig, args.out, "annotator_count_distribution_bar")

    # Agreement ratio by emotion
    ratio_df = df_valid[df_valid["n_annotators"] > 0].copy()
    ratio_df["agreement_ratio"] = ratio_df["agreement"] / ratio_df["n_annotators"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ratio_df.boxplot(column="agreement_ratio", by="emotion", ax=ax)
    ax.set_title("Agreement Ratio by Emotion")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Agreement Ratio")
    fig.suptitle("")
    _save_fig(fig, args.out, "agreement_ratio_by_emotion_boxplot")

    print(f"Saved figures to: {args.out}")


if __name__ == "__main__":
    main()
