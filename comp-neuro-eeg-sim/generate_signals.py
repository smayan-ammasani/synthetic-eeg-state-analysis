from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import numpy as np


MENTAL_STATES = ("relaxed", "focused", "noisy/uncertain")


@dataclass
class EEGSample:
    state: str
    time: np.ndarray
    clean_channels: np.ndarray
    noisy_channels: np.ndarray
    clean_signal: np.ndarray
    noisy_signal: np.ndarray
    sampling_rate: int
    num_channels: int


def _normalize(signal: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(signal)))
    if peak < 1e-9:
        return signal.copy()
    return signal / peak


def _simulate_relaxed(time: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    alpha = 1.3 * np.sin(2 * np.pi * 10 * time + rng.uniform(0, 2 * np.pi))
    theta = 0.45 * np.sin(2 * np.pi * 6 * time + rng.uniform(0, 2 * np.pi))
    drift = 0.2 * np.sin(2 * np.pi * 1.5 * time)
    return _normalize(alpha + theta + drift)


def _simulate_focused(time: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    beta = 1.1 * np.sin(2 * np.pi * 20 * time + rng.uniform(0, 2 * np.pi))
    upper_beta = 0.45 * np.sin(2 * np.pi * 28 * time + rng.uniform(0, 2 * np.pi))
    envelope = 1.0 + 0.18 * np.sin(2 * np.pi * 2 * time)
    support = 0.2 * np.sin(2 * np.pi * 12 * time)
    return _normalize(envelope * (beta + upper_beta) + support)


def _simulate_uncertain(time: np.ndarray, rng: np.random.Generator, sampling_rate: int) -> np.ndarray:
    mixed = (
        0.65 * np.sin(2 * np.pi * 9 * time + rng.uniform(0, 2 * np.pi))
        + 0.55 * np.sin(2 * np.pi * 23 * time + rng.uniform(0, 2 * np.pi))
        + 0.35 * np.sin(2 * np.pi * 38 * time + rng.uniform(0, 2 * np.pi))
    )
    artifact = np.zeros_like(time)
    spike_width = max(3, int(0.02 * sampling_rate))
    for center in rng.integers(0, len(time), size=3):
        left = max(0, center - spike_width)
        right = min(len(time), center + spike_width)
        artifact[left:right] += 0.9 * np.hanning(max(2, right - left))
    return _normalize(mixed + artifact)


def generate_eeg_signal(
    state: str,
    duration: float = 4.0,
    sampling_rate: int = 256,
    num_channels: int = 3,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> EEGSample:
    if state not in MENTAL_STATES:
        raise ValueError(f"Unknown state '{state}'. Expected one of {MENTAL_STATES}.")

    active_rng = rng if rng is not None else np.random.default_rng(seed)
    time = np.arange(0, duration, 1 / sampling_rate)

    if state == "relaxed":
        base_clean_signal = _simulate_relaxed(time, active_rng)
        noise_scale = 0.22
    elif state == "focused":
        base_clean_signal = _simulate_focused(time, active_rng)
        noise_scale = 0.18
    else:
        base_clean_signal = _simulate_uncertain(time, active_rng, sampling_rate)
        noise_scale = 0.55

    clean_channels: list[np.ndarray] = []
    noisy_channels: list[np.ndarray] = []
    for channel_index in range(num_channels):
        channel_gain = 1.0 + active_rng.normal(0.0, 0.08)
        channel_baseline = active_rng.normal(0.0, 0.03)
        channel_shape_variation = active_rng.normal(0.0, 0.04, size=time.shape)
        clean_signal = channel_gain * base_clean_signal + channel_shape_variation + channel_baseline
        noisy_signal = clean_signal + active_rng.normal(0.0, noise_scale + 0.01 * channel_index, size=time.shape)
        if state == "noisy/uncertain":
            line_noise = 0.12 * np.sin(2 * np.pi * 60 * time + active_rng.uniform(0, 2 * np.pi))
            noisy_signal = noisy_signal + line_noise

        clean_channels.append(clean_signal)
        noisy_channels.append(noisy_signal)

    clean_channel_array = np.vstack(clean_channels)
    noisy_channel_array = np.vstack(noisy_channels)
    clean_signal = np.mean(clean_channel_array, axis=0)
    noisy_signal = np.mean(noisy_channel_array, axis=0)

    return EEGSample(
        state=state,
        time=time,
        clean_channels=clean_channel_array,
        noisy_channels=noisy_channel_array,
        clean_signal=clean_signal,
        noisy_signal=noisy_signal,
        sampling_rate=sampling_rate,
        num_channels=num_channels,
    )


def generate_dataset(
    samples_per_state: int = 12,
    duration: float = 4.0,
    sampling_rate: int = 256,
    num_channels: int = 3,
    seed: int = 7,
) -> list[EEGSample]:
    rng = np.random.default_rng(seed)
    dataset: list[EEGSample] = []
    for state in MENTAL_STATES:
        for _ in range(samples_per_state):
            dataset.append(
                generate_eeg_signal(
                    state=state,
                    duration=duration,
                    sampling_rate=sampling_rate,
                    num_channels=num_channels,
                    rng=rng,
                )
            )
    return dataset


def summarize_samples(samples: Iterable[EEGSample]) -> str:
    sample_list = list(samples)
    if not sample_list:
        return "No samples generated."
    counts = {state: sum(sample.state == state for sample in sample_list) for state in MENTAL_STATES}
    example = sample_list[0]
    return (
        f"Generated {len(sample_list)} samples at {example.sampling_rate} Hz "
        f"across {len(example.time)} time points each with {example.num_channels} channels.\n"
        f"State counts: {counts}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic EEG-like signals.")
    parser.add_argument("--state", choices=MENTAL_STATES, help="Generate a single sample for one state.")
    parser.add_argument("--samples-per-state", type=int, default=12, help="Dataset size for each state.")
    parser.add_argument("--duration", type=float, default=4.0, help="Signal duration in seconds.")
    parser.add_argument("--sampling-rate", type=int, default=256, help="Samples per second.")
    parser.add_argument("--num-channels", type=int, default=3, help="Number of EEG channels to simulate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    args = parser.parse_args()

    if args.state:
        sample = generate_eeg_signal(
            state=args.state,
            duration=args.duration,
            sampling_rate=args.sampling_rate,
            num_channels=args.num_channels,
            seed=args.seed,
        )
        print(
            f"Generated '{sample.state}' signal with {len(sample.time)} samples "
            f"at {sample.sampling_rate} Hz across {sample.num_channels} channels."
        )
        return

    dataset = generate_dataset(
        samples_per_state=args.samples_per_state,
        duration=args.duration,
        sampling_rate=args.sampling_rate,
        num_channels=args.num_channels,
        seed=args.seed,
    )
    print(summarize_samples(dataset))


if __name__ == "__main__":
    main()
