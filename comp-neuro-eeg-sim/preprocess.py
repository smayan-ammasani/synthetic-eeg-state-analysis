from __future__ import annotations

import argparse

import numpy as np

from generate_signals import EEGSample, generate_eeg_signal


def remove_dc_offset(signal: np.ndarray) -> np.ndarray:
    return signal - np.mean(signal)


def moving_average(signal: np.ndarray, window_size: int = 7) -> np.ndarray:
    if window_size <= 1:
        return signal.copy()
    kernel = np.ones(window_size, dtype=float) / window_size
    return np.convolve(signal, kernel, mode="same")


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    std = float(np.std(signal))
    if std < 1e-8:
        return signal.copy()
    return signal / std


def preprocess_signal(signal: np.ndarray, smoothing_window: int = 7) -> np.ndarray:
    centered = remove_dc_offset(signal)
    smoothed = moving_average(centered, window_size=smoothing_window)
    return normalize_signal(smoothed)


def preprocess_sample(sample: EEGSample, smoothing_window: int = 7) -> np.ndarray:
    return preprocess_signal(sample.noisy_signal, smoothing_window=smoothing_window)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview EEG preprocessing on a synthetic sample.")
    parser.add_argument("--state", choices=("relaxed", "focused", "noisy/uncertain"), default="relaxed")
    parser.add_argument("--window", type=int, default=7, help="Moving-average smoothing window.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    args = parser.parse_args()

    sample = generate_eeg_signal(args.state, seed=args.seed)
    processed = preprocess_sample(sample, smoothing_window=args.window)
    print(
        f"Preprocessed '{sample.state}' signal with window={args.window}. "
        f"Mean={processed.mean():.3f}, Std={processed.std():.3f}"
    )


if __name__ == "__main__":
    main()
