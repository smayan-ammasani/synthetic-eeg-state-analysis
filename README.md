# EEG Signal Simulation + Neural State Classification

This project simulates simple EEG-style brain signals, adds realistic noise, classifies the signals into mental states, and visualizes the results. It uses NumPy for signal processing and Matplotlib for scientific plotting.

## Mental states

- `relaxed`: smoother alpha-dominant waves with mild noise
- `focused`: faster beta-dominant waves with a more structured rhythm
- `noisy/uncertain`: mixed frequencies plus stronger noise and artifacts

## Project structure

```text
comp-neuro-eeg-sim/
|- generate_signals.py
|- preprocess.py
|- classifier.py
|- visualize.py
|- requirements.txt
|- README.md
`- results/
```

## What each file does

- `generate_signals.py`: creates synthetic EEG-like waveforms using sine waves, NumPy noise, and simple multi-channel state-specific patterns
- `preprocess.py`: removes DC offset, smooths the signal with a moving average, and normalizes it
- `classifier.py`: extracts frequency-band features with a NumPy FFT and uses simple rule-based logic to predict the mental state
- `visualize.py`: generates Matplotlib plots for the signals, a multi-channel example, and a confusion matrix, then saves them in `results/`

## How the classifier works

The classifier looks at the signal in the frequency domain after preprocessing:

- more alpha-band power around 8-12 Hz suggests `relaxed`
- more beta-band power around 13-30 Hz suggests `focused`
- high spectral entropy or heavy high-frequency noise suggests `noisy/uncertain`

The simulation now creates multiple channels for each sample. The classifier preprocesses each channel, combines them with a simple average, and then extracts FFT-based features from that combined signal.


## Quick start

1. Open a terminal in `comp-neuro-eeg-sim`
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run:

```bash
python visualize.py
```

That command will:

- generate a synthetic dataset
- classify the signals
- save a PNG visualization to `results/eeg_signal_gallery.png`
- save an SVG copy to `results/eeg_signal_gallery.svg`
- save a multi-channel gallery to `results/multichannel_eeg_gallery.png`
- save a confusion matrix to `results/confusion_matrix.png`
- save evaluation metrics to `results/classification_report.json`

## Dependencies

- `numpy`
- `matplotlib`

Install them with:

```bash
pip install numpy matplotlib
```

You can also install them from:

```bash
pip install -r requirements.txt
```

