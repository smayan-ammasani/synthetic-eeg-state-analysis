from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from generate_signals import EEGSample, MENTAL_STATES, generate_dataset
from preprocess import preprocess_signal


def _power_spectrum(signal: np.ndarray, sampling_rate: int) -> tuple[np.ndarray, np.ndarray]:
    centered = signal - np.mean(signal)
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(len(centered), d=1 / sampling_rate)
    power = np.abs(spectrum) ** 2
    return freqs, power


def _band_power(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    return float(np.sum(power[mask]))


def _spectral_entropy(power: np.ndarray) -> float:
    usable = power[1:]
    total = float(np.sum(usable))
    if total <= 0 or len(usable) == 0:
        return 0.0
    distribution = usable / total
    distribution = distribution[distribution > 0]
    if len(distribution) == 0:
        return 0.0
    return float(-np.sum(distribution * np.log2(distribution)) / np.log2(len(usable)))


def extract_features(signal: np.ndarray, sampling_rate: int) -> dict[str, float]:
    freqs, power = _power_spectrum(signal, sampling_rate)
    total_power = _band_power(freqs, power, 1, 90) + 1e-9
    alpha_power = _band_power(freqs, power, 8, 12)
    beta_power = _band_power(freqs, power, 13, 30)
    gamma_power = _band_power(freqs, power, 30, 45)
    very_high_power = _band_power(freqs, power, 45, 90)

    usable_mask = (freqs >= 1) & (freqs <= 45)
    usable_freqs = freqs[usable_mask]
    usable_power = power[usable_mask]
    dominant_frequency = float(usable_freqs[np.argmax(usable_power)])

    roughness = float(np.std(np.diff(signal)) / (np.std(signal) + 1e-9))
    return {
        "alpha_ratio": alpha_power / total_power,
        "beta_ratio": beta_power / total_power,
        "gamma_ratio": gamma_power / total_power,
        "very_high_ratio": very_high_power / total_power,
        "spectral_entropy": _spectral_entropy(power),
        "dominant_frequency": dominant_frequency,
        "roughness": roughness,
    }


def classify_from_features(features: dict[str, float]) -> str:
    if (
        features["spectral_entropy"] > 0.72
        or features["very_high_ratio"] > 0.18
        or features["roughness"] > 1.05
    ):
        return "noisy/uncertain"
    if (
        features["alpha_ratio"] < 0.8
        and features["beta_ratio"] > 0.15
        and features["spectral_entropy"] > 0.22
    ):
        return "noisy/uncertain"
    if features["beta_ratio"] > features["alpha_ratio"] * 1.1 and features["dominant_frequency"] >= 13:
        return "focused"
    if features["alpha_ratio"] >= features["beta_ratio"] and 7 <= features["dominant_frequency"] <= 12.5:
        return "relaxed"
    return "noisy/uncertain"


def classify_signal(
    signal: np.ndarray,
    sampling_rate: int,
    already_preprocessed: bool = False,
) -> dict[str, object]:
    if signal.ndim == 1:
        processed_channels = np.expand_dims(signal.copy() if already_preprocessed else preprocess_signal(signal), axis=0)
    else:
        processed_channels = np.vstack(
            [channel.copy() if already_preprocessed else preprocess_signal(channel) for channel in signal]
        )
    processed_signal = np.mean(processed_channels, axis=0)
    features = extract_features(processed_signal, sampling_rate)
    predicted_state = classify_from_features(features)
    return {
        "predicted_state": predicted_state,
        "processed_signal": processed_signal,
        "processed_channels": processed_channels,
        "features": features,
    }


def evaluate_classifier(samples: list[EEGSample]) -> dict[str, object]:
    confusion = Counter()
    correct = 0
    sample_reports: list[dict[str, object]] = []

    for index, sample in enumerate(samples):
        result = classify_signal(sample.noisy_channels, sample.sampling_rate)
        predicted_state = str(result["predicted_state"])
        if predicted_state == sample.state:
            correct += 1
        confusion[(sample.state, predicted_state)] += 1
        sample_reports.append(
            {
                "sample_index": index,
                "true_state": sample.state,
                "predicted_state": predicted_state,
                "features": result["features"],
            }
        )

    matrix = {
        state: {predicted: confusion[(state, predicted)] for predicted in MENTAL_STATES}
        for state in MENTAL_STATES
    }
    return {
        "accuracy": correct / max(1, len(samples)),
        "confusion_matrix": matrix,
        "samples": sample_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-based EEG mental state classifier.")
    parser.add_argument("--samples-per-state", type=int, default=12, help="Dataset size for evaluation.")
    parser.add_argument("--num-channels", type=int, default=3, help="Number of EEG channels to simulate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    args = parser.parse_args()

    dataset = generate_dataset(
        samples_per_state=args.samples_per_state,
        num_channels=args.num_channels,
        seed=args.seed,
    )
    report = evaluate_classifier(dataset)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
