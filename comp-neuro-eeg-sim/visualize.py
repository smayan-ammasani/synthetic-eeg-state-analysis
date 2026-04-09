from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "comp-neuro-eeg-sim-mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from classifier import classify_signal, evaluate_classifier
from generate_signals import EEGSample, MENTAL_STATES, generate_dataset


STATE_COLORS = {
    "relaxed": "#2A9D8F",
    "focused": "#E76F51",
    "noisy/uncertain": "#6C757D",
}


def _choose_showcase_samples(dataset: list[EEGSample]) -> list[EEGSample]:
    selected: list[EEGSample] = []
    for state in MENTAL_STATES:
        candidates = [sample for sample in dataset if sample.state == state]
        best_match = candidates[0]
        for sample in candidates:
            prediction = classify_signal(sample.noisy_channels, sample.sampling_rate)
            if prediction["predicted_state"] == state:
                best_match = sample
                break
        selected.append(best_match)
    return selected


def plot_signal_gallery(samples: list[EEGSample], png_path: Path, svg_path: Path) -> None:
    figure, axes = plt.subplots(len(samples), 3, figsize=(15, 9), sharex="col")
    if len(samples) == 1:
        axes = np.array([axes])

    column_titles = ("Original signal", "Noisy signal", "Classified signal")
    for column_index, title in enumerate(column_titles):
        axes[0, column_index].set_title(title, fontsize=13, fontweight="bold")

    for row_index, sample in enumerate(samples):
        result = classify_signal(sample.noisy_channels, sample.sampling_rate)
        predicted_state = str(result["predicted_state"])
        processed_signal = np.asarray(result["processed_signal"])
        features = result["features"]

        axes[row_index, 0].plot(sample.time, sample.clean_signal, color=STATE_COLORS[sample.state], linewidth=1.7)
        axes[row_index, 1].plot(sample.time, sample.noisy_signal, color="#4F5D75", linewidth=1.1)
        axes[row_index, 2].plot(sample.time, processed_signal, color=STATE_COLORS[predicted_state], linewidth=1.4)

        axes[row_index, 0].set_ylabel(f"{sample.state}\nAmplitude", fontsize=10)
        for axis in axes[row_index]:
            axis.grid(alpha=0.25)

        axes[row_index, 2].text(
            0.02,
            0.95,
            (
                f"Predicted: {predicted_state}\n"
                f"Dominant freq: {features['dominant_frequency']:.1f} Hz\n"
                f"Alpha ratio: {features['alpha_ratio']:.2f}\n"
                f"Beta ratio: {features['beta_ratio']:.2f}"
            ),
            transform=axes[row_index, 2].transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#D9D9D9"},
        )

    for axis in axes[-1, :]:
        axis.set_xlabel("Time (seconds)")

    figure.suptitle("Synthetic EEG Signal Simulation and Neural State Classification", fontsize=16, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)


def plot_multichannel_gallery(samples: list[EEGSample], png_path: Path, svg_path: Path) -> None:
    figure, axes = plt.subplots(len(samples), 1, figsize=(12, 8), sharex=True)
    if len(samples) == 1:
        axes = np.array([axes])

    for row_index, sample in enumerate(samples):
        axis = axes[row_index]
        result = classify_signal(sample.noisy_channels, sample.sampling_rate)
        predicted_state = str(result["predicted_state"])

        for channel_index, channel_signal in enumerate(sample.noisy_channels):
            offset = channel_index * 3.2
            axis.plot(
                sample.time,
                channel_signal + offset,
                linewidth=1.0,
                label=f"Channel {channel_index + 1}" if row_index == 0 else None,
            )

        axis.set_ylabel(sample.state, fontsize=10)
        axis.grid(alpha=0.25)
        axis.text(
            0.01,
            0.92,
            f"Predicted: {predicted_state}",
            transform=axis.transAxes,
            fontsize=9,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#D9D9D9"},
        )

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time (seconds)")
    figure.suptitle("Multi-Channel EEG Simulation", fontsize=16, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(report: dict[str, object], png_path: Path, svg_path: Path) -> None:
    matrix = np.array(
        [
            [report["confusion_matrix"][state][predicted] for predicted in MENTAL_STATES]
            for state in MENTAL_STATES
        ]
    )
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(MENTAL_STATES)), MENTAL_STATES, rotation=15)
    axis.set_yticks(range(len(MENTAL_STATES)), MENTAL_STATES)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Confusion Matrix")

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(column_index, row_index, int(matrix[row_index, column_index]), ha="center", va="center")

    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)


def save_report(report: dict[str, object], output_path: Path) -> None:
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_demo(
    samples_per_state: int = 12,
    duration: float = 4.0,
    sampling_rate: int = 256,
    num_channels: int = 3,
    seed: int = 7,
) -> tuple[dict[str, Path], dict[str, object]]:
    output_dir = PROJECT_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_dataset(
        samples_per_state=samples_per_state,
        duration=duration,
        sampling_rate=sampling_rate,
        num_channels=num_channels,
        seed=seed,
    )
    report = evaluate_classifier(dataset)

    gallery_png_path = output_dir / "eeg_signal_gallery.png"
    gallery_svg_path = output_dir / "eeg_signal_gallery.svg"
    multichannel_png_path = output_dir / "multichannel_eeg_gallery.png"
    multichannel_svg_path = output_dir / "multichannel_eeg_gallery.svg"
    confusion_png_path = output_dir / "confusion_matrix.png"
    confusion_svg_path = output_dir / "confusion_matrix.svg"
    report_path = output_dir / "classification_report.json"
    plot_signal_gallery(_choose_showcase_samples(dataset), gallery_png_path, gallery_svg_path)
    plot_multichannel_gallery(_choose_showcase_samples(dataset), multichannel_png_path, multichannel_svg_path)
    plot_confusion_matrix(report, confusion_png_path, confusion_svg_path)
    save_report(report, report_path)
    return {
        "signal_gallery_png": gallery_png_path,
        "signal_gallery_svg": gallery_svg_path,
        "multichannel_gallery_png": multichannel_png_path,
        "multichannel_gallery_svg": multichannel_svg_path,
        "confusion_matrix_png": confusion_png_path,
        "confusion_matrix_svg": confusion_svg_path,
        "report_json": report_path,
    }, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create EEG visualizations and save a classification report.")
    parser.add_argument("--samples-per-state", type=int, default=12, help="Dataset size for evaluation.")
    parser.add_argument("--duration", type=float, default=4.0, help="Signal duration in seconds.")
    parser.add_argument("--sampling-rate", type=int, default=256, help="Samples per second.")
    parser.add_argument("--num-channels", type=int, default=3, help="Number of EEG channels to simulate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    args = parser.parse_args()

    output_paths, report = run_demo(
        samples_per_state=args.samples_per_state,
        duration=args.duration,
        sampling_rate=args.sampling_rate,
        num_channels=args.num_channels,
        seed=args.seed,
    )
    print(f"Saved signal gallery PNG to {output_paths['signal_gallery_png']}")
    print(f"Saved signal gallery SVG to {output_paths['signal_gallery_svg']}")
    print(f"Saved multi-channel PNG to {output_paths['multichannel_gallery_png']}")
    print(f"Saved multi-channel SVG to {output_paths['multichannel_gallery_svg']}")
    print(f"Saved confusion matrix PNG to {output_paths['confusion_matrix_png']}")
    print(f"Saved confusion matrix SVG to {output_paths['confusion_matrix_svg']}")
    print(f"Saved classification report to {output_paths['report_json']}")
    print(f"Classifier accuracy on synthetic dataset: {report['accuracy']:.1%}")


if __name__ == "__main__":
    main()
