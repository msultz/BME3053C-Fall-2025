# Fourier Transform Streamlit Lab

An educational Streamlit application that lets BME3053C students explore Fourier transforms interactively. The UI mirrors the lesson prompt and provides explanatory text for every tab, widget, and plot so the experience feels like a guided lab notebook.

## Setup

1. Create and activate a Python 3.10+ environment (virtualenv, conda, or Codespaces).
2. Install the dependencies:
	```bash
	pip install -r requirements.txt
	```

## Run the app

Launch the Streamlit server from the project root:

```bash
streamlit run streamlit_app.py
```

Streamlit prints a local URL (and a public sharing URL inside Codespaces). Open it in your browser to access the lab.

## Tab-by-tab guide

### 1. Signal Builder
- **Purpose:** Compose a signal as a sum of sinusoids, read the LaTeX expression, and watch time/frequency plots update live.
- **Key controls:** Sidebar presets, component sliders (amplitude/frequency/phase), waveform toggles, enable/disable checkboxes, and a band-selection tool that reconstructs brushed frequencies.
- **Learning goals:** Connect each slider to the analytic expression, observe peaks move as frequencies change, and see how hovering on peaks highlights their matching sinusoid.

### 2. Sampling & Aliasing
- **Purpose:** Demonstrate Nyquist, aliasing, duration-based resolution, and zero padding effects.
- **Key controls:** Sliders for sampling rate `fs`, observation time `T`, total samples `N`, zero-padding toggle plus factor. Sample markers overlay the continuous waveform and alias warnings explain folded frequencies.
- **Learning goals:** Understand how `fs`, `T`, and `N` interact (`N = fs · T`), recognize when aliasing occurs, and see how zero padding only interpolates the FFT display.

### 3. Windowing & Leakage
- **Purpose:** Compare window functions, show leakage due to misaligned tones, and report main-lobe/sidelobe metrics.
- **Key controls:** Window type dropdown (Rectangular, Hann, Hamming, Blackman, Kaiser), Kaiser β slider, and a “misalign frequency” toggle that forces leakage. The tab plots the window shape, the resulting spectrum, and quantitative metrics.
- **Learning goals:** Explain why tapering reduces edge discontinuities, how window choice affects spectral leakage, and why the main-lobe width vs. sidelobe level trade-off matters.

### 4. Band Reconstruction
- **Purpose:** Treat the spectrum as an interactive filter. Students brush a frequency band, reconstruct only that energy, and interpret the resulting waveform overlay.
- **Key controls:** Plotly brush on the magnitude spectrum (plus reset button).
- **Learning goals:** Relate selected frequency spans to time-domain structures and reinforce the concept of filtering via ideal band selection.

## Helper functions

To keep the code organized, the app implements the helper API required by the prompt:

- `generate_time_vector()` – builds the base time axis and returns the effective sampling rate.
- `build_signal()` – sums all enabled components, returns individual contributions, the LaTeX expression, and the list of active components for highlighting.
- `fft_spectrum()` – applies a window, optionally zero pads, and returns frequency, magnitude, phase, and the complex FFT.
- `reconstruct_band()` – ideal band-pass filtering via FFT masking and inverse transform.
- `compute_window()` – generates standard windows (Rectangular, Hann, Hamming, Blackman, Kaiser) including the tunable Kaiser β.

Each tab reuses these helpers so plots stay synchronized while Streamlit’s session state keeps hover selections and band choices consistent across the entire interface.

