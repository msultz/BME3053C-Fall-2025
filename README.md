# BME3053C Streamlit Labs

This repository hosts two Streamlit experiences for BME3053C:

- **Fourier Transform Lab** – explore frequency-domain intuition through guided signal-building exercises.
- **Supervised ML Lab** – load the cardio dataset, probe features, train classifiers, and study threshold trade-offs.

## Setup

1. Create and activate a Python 3.10+ environment (virtualenv, conda, or Codespaces).
2. Install the dependencies:
	```bash
	pip install -r requirements.txt
	```

## Run the apps

- Fourier lab:
	```bash
	streamlit run streamlit_app.py
	```
- Supervised ML lab:
	```bash
	streamlit run streamlit_ml_app.py
	```

Streamlit prints a local URL (and a public sharing URL inside Codespaces). Open it in your browser to access the selected lab.

## Tab-by-tab guide

### Fourier Transform Lab

#### 1. Signal Builder
- **Purpose:** Compose a signal as a sum of sinusoids, read the LaTeX expression, and watch time/frequency plots update live.
- **Key controls:** Sidebar presets, component sliders (amplitude/frequency/phase), waveform toggles, enable/disable checkboxes, and a band-selection tool that reconstructs brushed frequencies.
- **Learning goals:** Connect each slider to the analytic expression, observe peaks move as frequencies change, and see how hovering on peaks highlights their matching sinusoid.

#### 2. Sampling & Aliasing
- **Purpose:** Demonstrate Nyquist, aliasing, duration-based resolution, and zero padding effects.
- **Key controls:** Sliders for sampling rate `fs`, observation time `T`, total samples `N`, zero-padding toggle plus factor. Sample markers overlay the continuous waveform and alias warnings explain folded frequencies.
- **Learning goals:** Understand how `fs`, `T`, and `N` interact (`N = fs · T`), recognize when aliasing occurs, and see how zero padding only interpolates the FFT display.

#### 3. Windowing & Leakage
- **Purpose:** Compare window functions, show leakage due to misaligned tones, and report main-lobe/sidelobe metrics.
- **Key controls:** Window type dropdown (Rectangular, Hann, Hamming, Blackman, Kaiser), Kaiser β slider, and a “misalign frequency” toggle that forces leakage. The tab plots the window shape, the resulting spectrum, and quantitative metrics.
- **Learning goals:** Explain why tapering reduces edge discontinuities, how window choice affects spectral leakage, and why the main-lobe width vs. sidelobe level trade-off matters.

#### 4. Band Reconstruction
- **Purpose:** Treat the spectrum as an interactive filter. Students brush a frequency band, reconstruct only that energy, and interpret the resulting waveform overlay.
- **Key controls:** Plotly brush on the magnitude spectrum (plus reset button).
- **Learning goals:** Relate selected frequency spans to time-domain structures and reinforce the concept of filtering via ideal band selection.

### Supervised ML Lab

#### 1. Dataset Tour
- **Purpose:** Inspect dataset size, class balance, and descriptive statistics for every numeric feature.
- **Key controls:** Metrics row, class-balance bar chart, and customizable column preview.
- **Learning goals:** Emphasize the need to understand cohort mix and data quality before modeling.

#### 2. Feature Explorer
- **Purpose:** Compare how single features or feature pairs relate to the cardio diagnosis.
- **Key controls:** Histogram/scatter/box toggles, color-by-target overlays, and a correlation heatmap expander.
- **Learning goals:** Practice spotting separability and multicollinearity before committing to a model.

#### 3. Model Playground
- **Purpose:** Select features, pick a classifier (logistic, tree, or random forest), and view core metrics/plots.
- **Key controls:** Feature multiselect, test-split slider, hyperparameter sliders, confusion matrix, ROC curve, and feature-importance visuals.
- **Learning goals:** Relate hyperparameters to performance and connect metrics to visuals.

#### 4. Threshold Lab
- **Purpose:** Show how moving the probability threshold trades precision against recall on the held-out set.
- **Key controls:** Threshold slider, live confusion matrix, and precision/recall vs. threshold chart.
- **Learning goals:** Reinforce that classifier scores must be paired with domain-specific thresholds, not just default 0.5 decisions.

