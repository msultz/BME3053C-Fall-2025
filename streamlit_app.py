"""Interactive Streamlit lab for exploring Fourier transforms."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import get_window
import streamlit as st
from streamlit_plotly_events import plotly_events


# ---------------------------------------------------------------------------
# Data structures and presets
# ---------------------------------------------------------------------------
@dataclass
class SignalComponent:
    """Container describing one sinusoid in the custom sum."""

    label: str
    amplitude: float = 1.0
    frequency: float = 5.0
    phase: float = 0.0
    waveform: str = "sine"
    enabled: bool = True
    uid: str = field(default_factory=lambda: f"comp-{np.random.randint(0, 1_000_000)}")


PRESETS: Dict[str, Dict] = {
    "Square Wave": {
        "description": "Square waves use only odd harmonics with amplitudes that decay as 1/n, so their spectra contain strong odd spikes.",
        "generator": lambda base: [
            (4 / (np.pi * n), base * n, 0.0, "sine")
            for n in range(1, 12, 2)
        ],
    },
    "Triangle Wave": {
        "description": "Triangle waves also rely on odd harmonics, but amplitudes fall much faster (1/n²), giving a smoother spectrum.",
        "generator": lambda base: [
            ((8 / (np.pi ** 2)) * ((-1) ** ((n - 1) // 2)) / (n ** 2), base * n, 0.0, "sine")
            for n in range(1, 12, 2)
        ],
    },
    "Sawtooth Wave": {
        "description": "Sawtooth waves contain both even and odd harmonics with amplitudes that decay as 1/n, so their comb spectrum is dense.",
        "generator": lambda base: [
            (2 / (np.pi * n) * ((-1) ** (n + 1)), base * n, 0.0, "sine")
            for n in range(1, 12)
        ],
    },
    "Gaussian Pulse": {
        "description": "Gaussian pulses are localized in time and therefore spread broadly in frequency—perfect for discussing the uncertainty principle.",
        "generator": lambda base: [
            (1.0, base, 0.0, "sine"),
            (0.8, base * 1.5, np.pi / 4, "sine"),
            (0.6, base * 2.0, np.pi / 2, "sine"),
        ],
    },
    "Step": {
        "description": "Step functions emphasize low-frequency energy that decays slowly, so their spectra follow a 1/f profile and ring at the edges.",
        "generator": lambda base: [
            (2 / (np.pi * n), base * n, 0.0, "sine")
            for n in range(1, 16, 2)
        ],
    },
    "Impulse/Spike": {
        "description": "An impulse packs energy into one instant, spreading it almost uniformly across frequencies.",
        "generator": lambda base: [
            (np.random.uniform(0.2, 0.8), base * np.random.uniform(1, 10), np.random.uniform(0, 2 * np.pi), "cosine")
            for _ in range(8)
        ],
    },
    "Chirp": {
        "description": "Chirps sweep frequency over time, so their spectra cover a band instead of a single spike.",
        "generator": lambda base: [
            (1.0, base, 0.0, "sine"),
            (0.9, base * 1.5, np.pi / 4, "sine"),
            (0.8, base * 2.0, np.pi / 2, "sine"),
            (0.7, base * 2.7, np.pi * 0.75, "sine"),
        ],
    },
}

WINDOW_INFO = {
    "rectangular": {
        "label": "Rectangular",
        "description": "No tapering: tightest bins but worst leakage because the window ends abruptly.",
        "main_lobe": 0.89,
        "sidelobe": -13.0,
    },
    "hann": {
        "label": "Hann",
        "description": "Cosine taper that lowers sidelobes while modestly widening the main lobe.",
        "main_lobe": 1.44,
        "sidelobe": -31.5,
    },
    "hamming": {
        "label": "Hamming",
        "description": "Balances leakage and resolution with moderate sidelobe suppression.",
        "main_lobe": 1.30,
        "sidelobe": -42.7,
    },
    "blackman": {
        "label": "Blackman",
        "description": "Aggressive taper with very low sidelobes but a wide main lobe—great for weak tones in noise.",
        "main_lobe": 1.68,
        "sidelobe": -58.0,
    },
    "kaiser": {
        "label": "Kaiser",
        "description": "Adjustable via β. Higher β reduces sidelobes at the expense of main-lobe width.",
        "main_lobe": None,
        "sidelobe": None,
    },
}


# ---------------------------------------------------------------------------
# Numerical utilities
# ---------------------------------------------------------------------------
def generate_time_vector(fs: float, duration: float, n_samples: int) -> Tuple[np.ndarray, float]:
    n_samples = max(128, int(n_samples))
    duration = max(duration, 0.1)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    effective_fs = n_samples / duration
    return t, effective_fs


def build_signal(
    components: Sequence[SignalComponent],
    t: np.ndarray,
    misalign: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray], str, List[SignalComponent]]:
    total = np.zeros_like(t)
    pieces: List[np.ndarray] = []
    active_components: List[SignalComponent] = []
    terms = []
    for idx, comp in enumerate(components):
        if not comp.enabled:
            continue
        freq = comp.frequency + (0.37 if misalign and idx == 0 else 0.0)
        if comp.waveform == "cosine":
            contrib = comp.amplitude * np.cos(2 * np.pi * freq * t + comp.phase)
            term = f"{comp.amplitude:.2f}\\cos(2\\pi {freq:.2f} t + {comp.phase:.2f})"
        else:
            contrib = comp.amplitude * np.sin(2 * np.pi * freq * t + comp.phase)
            term = f"{comp.amplitude:.2f}\\sin(2\\pi {freq:.2f} t + {comp.phase:.2f})"
        total += contrib
        pieces.append(contrib)
        active_components.append(comp)
        terms.append(term)
    if not terms:
        terms.append("0")
    latex = "x(t) = " + " + ".join(terms)
    return total, pieces, latex, active_components


def compute_window(name: str, n: int, beta: float) -> np.ndarray:
    name = name.lower()
    if name == "rectangular":
        return np.ones(n)
    if name == "kaiser":
        return get_window((name, beta), n, fftbins=True)
    return get_window(name, n, fftbins=True)


def fft_spectrum(
    signal: np.ndarray,
    fs: float,
    window: np.ndarray,
    zero_pad_factor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    zero_pad_factor = max(1.0, float(zero_pad_factor))
    tapered = signal * window
    pad_len = int(len(tapered) * zero_pad_factor)
    padded = np.zeros(pad_len)
    padded[: len(tapered)] = tapered
    spectrum = fft(padded)
    freqs = fftfreq(pad_len, 1.0 / fs)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    return freqs, magnitude, phase, spectrum


def positive_freq_slice(freqs: np.ndarray, magnitude: np.ndarray, phase: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = freqs >= 0
    return freqs[mask], magnitude[mask], phase[mask]


def reconstruct_band(spectrum: np.ndarray, freqs: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    low, high = sorted(band)
    mask = (np.abs(freqs) >= low) & (np.abs(freqs) <= high)
    filtered = np.where(mask, spectrum, 0.0)
    recon = ifft(filtered).real
    return recon


def alias_frequency(freq: float, fs: float) -> float:
    if fs <= 0:
        return freq
    half = fs / 2
    if freq <= half:
        return freq
    aliased = ((freq + half) % fs) - half
    return abs(aliased)


def window_metrics(name: str, beta: float) -> Tuple[float, float]:
    info = WINDOW_INFO.get(name, {})
    if name == "kaiser":
        sidelobe = -20 * math.log10(beta + 0.1)
        main_lobe = 0.885 + 0.0187 * beta
        return main_lobe, sidelobe
    return info.get("main_lobe", 1.0), info.get("sidelobe", -20.0)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
def init_components() -> None:
    if "components" not in st.session_state:
        base_freqs = [3, 5, 7, 11, 13]
        st.session_state.components = [
            SignalComponent(label=f"Component {i + 1}", amplitude=1.0, frequency=base_freqs[i])
            for i in range(5)
        ]


def preset_controls() -> None:
    st.subheader("Quick-add presets")
    st.caption("Pick a canonical waveform to auto-populate the component list and study its spectrum.")
    preset_name = st.selectbox(
        "Preset",
        list(PRESETS.keys()),
        help="Each preset highlights a classic waveform and its harmonic recipe.",
        key="preset_select",
    )
    st.info(PRESETS[preset_name]["description"])
    base_freq = st.slider(
        "Preset base frequency (Hz)",
        min_value=1.0,
        max_value=60.0,
        value=5.0,
        help="Fundamental frequency used when generating harmonic-based presets.",
        key="preset_base",
    )
    st.caption("Changing the base frequency shifts every harmonic and therefore every spectral spike.")
    if st.button("Load preset components", key="load_preset"):
        comps = PRESETS[preset_name]["generator"](base_freq)
        st.session_state.components = [
            SignalComponent(
                label=f"Component {i + 1}",
                amplitude=float(a),
                frequency=float(f),
                phase=float(phi),
                waveform=wave,
            )
            for i, (a, f, phi, wave) in enumerate(comps[:8])
        ]
        st.success("Preset applied—scroll down to fine-tune individual harmonics.")
        st.experimental_rerun()


def component_block(comp: SignalComponent, idx: int) -> None:
    comp.enabled = st.checkbox(
        "Include in sum",
        value=comp.enabled,
        key=f"enabled_{comp.uid}",
        help="Toggle components on/off to see their spectral peaks appear or vanish.",
    )
    st.caption("Use this checkbox to isolate terms and watch their fingerprint disappear when disabled.")

    comp.waveform = st.radio(
        "Waveform",
        options=["sine", "cosine"],
        index=0 if comp.waveform == "sine" else 1,
        key=f"wave_{comp.uid}",
        help="Sine and cosine differ by a 90° phase shift—great for highlighting timing vs. phase.",
    )
    st.caption("Switching between sine/cosine shows that phase shifts do not move frequency content.")

    comp.amplitude = st.slider(
        "Amplitude",
        min_value=0.0,
        max_value=5.0,
        value=float(comp.amplitude),
        step=0.1,
        key=f"amp_{comp.uid}",
        help="Amplitude sets the wave height and therefore the spectral spike height.",
    )
    st.caption("Bigger amplitude → taller spectral peaks. Use it to compare energy contributions.")

    comp.frequency = st.slider(
        "Frequency (Hz)",
        min_value=0.1,
        max_value=120.0,
        value=float(comp.frequency),
        step=0.1,
        key=f"freq_{comp.uid}",
        help="Frequency controls oscillations per second. Drag to move spikes along the frequency axis.",
    )
    st.caption("Slide the frequency to watch peaks travel left/right in the FFT plot.")

    comp.phase = st.slider(
        "Phase (rad)",
        min_value=-np.pi,
        max_value=np.pi,
        value=float(comp.phase),
        step=0.01,
        key=f"phase_{comp.uid}",
        help="Phase shifts move the waveform in time while keeping its spectrum magnitude unchanged.",
    )
    st.caption("Phase changes shift the waveform horizontally but only alter the phase plot, not |X(f)|.")

    cols = st.columns(2)
    with cols[0]:
        if st.button("Duplicate", key=f"dup_{comp.uid}"):
            clone = SignalComponent(
                label=f"Component {len(st.session_state.components) + 1}",
                amplitude=comp.amplitude,
                frequency=comp.frequency,
                phase=comp.phase,
                waveform=comp.waveform,
            )
            st.session_state.components.append(clone)
            st.experimental_rerun()
    with cols[1]:
        if st.button("Remove", key=f"remove_{comp.uid}", disabled=len(st.session_state.components) <= 1):
            st.session_state.components.pop(idx)
            st.experimental_rerun()


def sidebar_signal_builder() -> None:
    with st.sidebar:
        st.title("Signal Builder")
        st.write("Compose a custom signal as a sum of sinusoids. Every control explains its purpose so you can narrate the physics behind the math.")
        preset_controls()
        st.divider()
        st.subheader("Custom components")
        st.caption("Expand each card to configure amplitude, frequency, phase, and waveform type.")
        for idx, comp in enumerate(st.session_state.components):
            with st.expander(comp.label, expanded=(idx == 0)):
                component_block(comp, idx)
        if len(st.session_state.components) < 8 and st.button("Add new component", key="add_component"):
            st.session_state.components.append(
                SignalComponent(label=f"Component {len(st.session_state.components) + 1}", amplitude=0.5, frequency=20.0)
            )
            st.experimental_rerun()
        st.caption("Need more harmonics? Add up to eight components to build richer spectra.")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def render_time_freq_figures(
    t: np.ndarray,
    total: np.ndarray,
    pieces: List[np.ndarray],
    active_components: Sequence[SignalComponent],
    freqs: np.ndarray,
    mag: np.ndarray,
    phase: np.ndarray,
    highlight_freq: float | None,
    highlight_time: float | None,
) -> Tuple[go.Figure, go.Figure]:
    fig_time = go.Figure()
    fig_time.add_trace(
        go.Scatter(x=t, y=total, mode="lines", name="Sum", line=dict(color="#ff6600", width=3))
    )
    for idx, contrib in enumerate(pieces):
        color = "#1f77b4"
        opacity = 0.25
        if highlight_freq is not None and idx < len(active_components):
            if abs(active_components[idx].frequency - highlight_freq) < 0.5:
                color = "#d62728"
                opacity = 0.95
        fig_time.add_trace(
            go.Scatter(
                x=t,
                y=contrib,
                mode="lines",
                name=active_components[idx].label,
                line=dict(width=1, dash="dot", color=color),
                opacity=opacity,
            )
        )
    if highlight_time is not None:
        fig_time.add_vrect(
            x0=highlight_time - 0.02,
            x1=highlight_time + 0.02,
            fillcolor="#ffe6a7",
            opacity=0.5,
            line_width=0,
        )
    fig_time.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=360,
        hovermode="x unified",
        margin=dict(l=40, r=20, t=35, b=40),
    )

    fig_freq = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04)
    fig_freq.add_trace(
        go.Scatter(x=freqs, y=mag, mode="lines", name="|X(f)|", line=dict(color="#2ca02c")),
        row=1,
        col=1,
    )
    fig_freq.add_trace(
        go.Scatter(x=freqs, y=phase, mode="lines", name="∠X(f)", line=dict(color="#9467bd")),
        row=2,
        col=1,
    )
    if highlight_freq is not None:
        fig_freq.add_vline(x=highlight_freq, line=dict(color="#d62728", width=2, dash="dot"))
    if highlight_time is not None:
        y_top = float(np.max(mag) * 1.05) if len(mag) else 1.0
        fig_freq.add_shape(
            type="rect",
            x0=min(freqs),
            x1=max(freqs),
            y0=0,
            y1=y_top,
            fillcolor="#9be3ff",
            opacity=0.15,
            layer="below",
            line_width=0,
            row=1,
            col=1,
        )
    fig_freq.update_layout(
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        height=420,
        margin=dict(l=40, r=20, t=35, b=40),
        hovermode="x",
        dragmode="select",
    )
    fig_freq.update_yaxes(title_text="Phase (rad)", row=2, col=1)
    return fig_time, fig_freq


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------
def signal_builder_tab(
    t: np.ndarray,
    total: np.ndarray,
    pieces: List[np.ndarray],
    active_components: Sequence[SignalComponent],
    latex_expr: str,
    freqs_pos: np.ndarray,
    mag_pos: np.ndarray,
    phase_pos: np.ndarray,
    full_freqs: np.ndarray,
    full_spectrum: np.ndarray,
) -> None:
    st.info(
        "**Purpose**: Build signals, inspect both domains, and link every control to the resulting waveform and spectrum."
    )
    st.latex(latex_expr)
    st.caption("This expression mirrors your sidebar choices, showing exactly how each sinusoid adds to the sum.")

    highlight_freq = st.session_state.get("freq_hover")
    highlight_time = st.session_state.get("time_hover")
    fig_time, fig_freq = render_time_freq_figures(
        t, total, pieces, active_components, freqs_pos, mag_pos, phase_pos, highlight_freq, highlight_time
    )

    cols = st.columns(2)
    with cols[0]:
        time_events = plotly_events(fig_time, hover_event=True, key="time_plot")
        st.caption("Hover on time events to highlight their wide spectral support—sharp spikes need many frequencies.")
        if time_events:
            st.session_state.time_hover = time_events[-1]["x"]
    with cols[1]:
        freq_events = plotly_events(fig_freq, hover_event=True, select_event=True, key="freq_plot")
        st.caption("Hover on frequency spikes to highlight the matching sinusoid. Drag-select to define a reconstruction band.")
        if freq_events:
            last_event = freq_events[-1]
            if "x" in last_event:
                st.session_state.freq_hover = last_event["x"]
            xs = [ev["x"] for ev in freq_events if "x" in ev]
            if len(xs) >= 2:
                st.session_state.signal_band = (min(xs), max(xs))

    if "signal_band" in st.session_state:
        band = st.session_state.signal_band
        recon = reconstruct_band(full_spectrum, full_freqs, band)
        fig = go.Figure(
            data=[
                go.Scatter(x=t, y=total, name="Original"),
                go.Scatter(x=t, y=recon[: len(t)], name="Band-limited", line=dict(color="#d62728")),
            ]
        )
        fig.update_layout(title=f"Band Select & Reconstruct ({band[0]:.2f}-{band[1]:.2f} Hz)", hovermode="x")
        st.plotly_chart(fig, use_container_width=True)
        st.info("Band reconstruction behaves like an ideal filter—only the brushed frequencies survive the inverse transform.")
        if st.button("Clear band selection", key="clear_band_signal"):
            st.session_state.pop("signal_band")


def sampling_aliasing_tab(
    t_high_res: np.ndarray,
    x_high_res: np.ndarray,
    components: Sequence[SignalComponent],
) -> None:
    st.info(
        "**Purpose**: Explore how sampling rate, duration, and sample count control discrete representations, aliasing, and FFT resolution."
    )
    cols = st.columns(3)
    with cols[0]:
        fs_control = st.slider(
            "Sampling rate $f_s$ (Hz)",
            min_value=100.0,
            max_value=4000.0,
            value=800.0,
            step=50.0,
            help="Raise $f_s$ to capture faster oscillations. Nyquist says $f_s/2$ must exceed the highest signal frequency.",
        )
        st.caption("Use this slider to test Nyquist—when $f_s/2$ dips below a component, aliasing appears.")
    with cols[1]:
        duration = st.slider(
            "Duration $T$ (s)",
            min_value=0.25,
            max_value=5.0,
            value=1.0,
            step=0.25,
            help="Longer capture windows improve frequency resolution because Δf = 1/T.",
        )
        st.caption("Increasing $T$ narrows frequency bins, helping separate close peaks.")
    with cols[2]:
        n_samples = st.slider(
            "Samples $N$",
            min_value=256,
            max_value=8192,
            value=2048,
            step=256,
            help="N controls how many discrete points feed the FFT—more samples sharpen both domains.",
        )
        st.caption("Changing $N$ emulates denser acquisition without necessarily altering $f_s$ or $T$.")

    zero_pad = st.checkbox(
        "Zero padding",
        value=True,
        help="Padding interpolates the FFT display. It smooths curves but does not add new information.",
    )
    st.caption("Toggle padding to show students that smoother spectra still represent the same underlying samples.")
    pad_factor = st.slider(
        "Zero-pad factor",
        min_value=1.0,
        max_value=8.0,
        value=2.0,
        step=0.5,
        help="Controls how much longer the FFT should be relative to the data length.",
        disabled=not zero_pad,
    )
    st.caption("Higher pad factors give more interpolated frequency points for peak picking.")
    if not zero_pad:
        pad_factor = 1.0

    sample_t, fs_eff = generate_time_vector(fs_control, duration, n_samples)
    sampled_values = np.interp(sample_t, t_high_res, x_high_res)
    window = compute_window("hann", len(sampled_values), beta=8.6)
    freqs, mag, _, _ = fft_spectrum(sampled_values, fs_eff, window, pad_factor)
    pos_mask = freqs >= 0
    freqs_pos, mag_pos = freqs[pos_mask], mag[pos_mask]

    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(x=t_high_res, y=x_high_res, mode="lines", name="Continuous"))
    fig_time.add_trace(
        go.Scatter(
            x=sample_t,
            y=sampled_values,
            mode="markers",
            name="Samples",
            marker=dict(size=8, color="#d62728"),
        )
    )
    fig_time.update_layout(title="Sampling overlay", xaxis_title="Time (s)", yaxis_title="Amplitude")
    st.plotly_chart(fig_time, use_container_width=True)
    st.info("Markers show where the analog waveform is captured. Sparse sampling struggles to follow fast oscillations.")

    fig_fft = go.Figure(go.Scatter(x=freqs_pos, y=mag_pos, mode="lines"))
    fig_fft.update_layout(title="FFT of sampled data", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude")
    st.plotly_chart(fig_fft, use_container_width=True)
    st.caption("Doubling T halves the bin spacing, so peaks become easier to distinguish.")

    nyquist = fs_eff / 2
    alias_rows = []
    for comp in components:
        if not comp.enabled:
            continue
        alias = alias_frequency(comp.frequency, fs_eff)
        if alias != comp.frequency:
            alias_rows.append((comp.frequency, alias))
    if alias_rows:
        for orig, alias in alias_rows:
            st.warning(f"{orig:.1f} Hz exceeds Nyquist ({nyquist:.1f} Hz) and aliases to {alias:.1f} Hz.")
        st.caption("Aliasing folds energy back into the baseband—raise $f_s$ or pre-filter before sampling.")
    else:
        st.success("All components lie below Nyquist, so sampling is alias-free.")


def windowing_tab(
    t: np.ndarray,
    components: Sequence[SignalComponent],
    fs: float,
) -> None:
    st.info(
        "**Purpose**: Compare window shapes, observe spectral leakage, and discuss main-lobe/sidelobe trade-offs."
    )
    labels = [info["label"] for info in WINDOW_INFO.values()]
    key_by_label = {info["label"]: key for key, info in WINDOW_INFO.items()}

    cols = st.columns(3)
    with cols[0]:
        window_label = st.selectbox(
            "Window type",
            labels,
            help="Pick how aggressively to taper—rectangular keeps resolution, while Blackman/kaiser cut leakage.",
        )
        st.caption(WINDOW_INFO[key_by_label[window_label]]["description"])
    with cols[1]:
        beta = st.slider(
            "Kaiser β",
            min_value=0.0,
            max_value=14.0,
            value=8.6,
            help="Only affects the Kaiser window. Higher β lowers sidelobes but widens the main lobe.",
        )
        st.caption("Drag β to morph the Kaiser window between rectangular-like and Blackman-like behavior.")
    with cols[2]:
        misalign = st.checkbox(
            "Misalign frequency to bin",
            value=False,
            help="Forces the first component to complete a non-integer number of cycles, showing leakage streaks.",
        )
        st.caption("Misaligning a tone is the classic leakage demonstration—energy smears across bins.")

    window_key = key_by_label[window_label]
    window_vals = compute_window(window_key, len(t), beta)
    x_windowed, _, _, _ = build_signal(components, t, misalign=misalign)

    fig_window = go.Figure(go.Scatter(x=t, y=window_vals, mode="lines"))
    fig_window.update_layout(title="Window shape", xaxis_title="Time (s)", yaxis_title="Amplitude")
    st.plotly_chart(fig_window, use_container_width=True)
    st.caption("Windowing tapers the edges, reducing discontinuities before the FFT.")

    freqs, mag, _, _ = fft_spectrum(x_windowed, fs, window_vals, 1.0)
    freqs_pos, mag_pos, _ = positive_freq_slice(freqs, mag, mag)
    fig_fft = go.Figure(go.Scatter(x=freqs_pos, y=mag_pos, mode="lines"))
    fig_fft.update_layout(title="Windowed spectrum", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude")
    st.plotly_chart(fig_fft, use_container_width=True)
    st.caption("Compare leakage tails with different windows and with/without misalignment.")

    main_lobe, sidelobe = window_metrics(window_key, beta)
    st.metric("Main-lobe width (bins)", f"{main_lobe:.2f}")
    st.metric("Peak sidelobe (dB)", f"{sidelobe:.1f}")
    st.caption("Narrow lobes resolve close tones; low sidelobes prevent weak features from being masked.")


def band_reconstruction_tab(
    t: np.ndarray,
    total: np.ndarray,
    freqs_pos: np.ndarray,
    mag_pos: np.ndarray,
    full_freqs: np.ndarray,
    full_spectrum: np.ndarray,
) -> None:
    st.info(
        "**Purpose**: Treat frequency selection as an interactive filter design exercise. Brush a band, reconstruct it, and interpret the resulting waveform."
    )
    fig = go.Figure(go.Scatter(x=freqs_pos, y=mag_pos, mode="lines"))
    fig.update_layout(title="Brush to select a band", dragmode="select", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude")
    events = plotly_events(fig, select_event=True, key="band_tab_plot")
    if events:
        xs = [ev.get("x") for ev in events if ev.get("x") is not None]
        if len(xs) >= 2:
            st.session_state.band_tab = (min(xs), max(xs))
    band = st.session_state.get("band_tab")
    if band:
        recon = reconstruct_band(full_spectrum, full_freqs, band)
        fig_recon = go.Figure(
            data=[
                go.Scatter(x=t, y=total, name="Original"),
                go.Scatter(x=t, y=recon[: len(t)], name="Reconstructed", line=dict(color="#ff7f0e")),
            ]
        )
        fig_recon.update_layout(title=f"Reconstructed band {band[0]:.2f}-{band[1]:.2f} Hz", hovermode="x")
        st.plotly_chart(fig_recon, use_container_width=True)
        st.caption("Overlay the reconstructed trace to see which rhythms survive your frequency filter.")
    else:
        st.warning("Use the selection box on the spectrum to pick a band for reconstruction.")
    if st.button("Reset band selection", key="reset_band_tab"):
        st.session_state.pop("band_tab", None)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Fourier Transform Lab", layout="wide")
    init_components()
    sidebar_signal_builder()

    fs_ref = 800.0
    duration_ref = 1.0
    n_ref = 2048
    t, fs_effective = generate_time_vector(fs_ref, duration_ref, n_ref)
    total, pieces, latex_expr, active_components = build_signal(st.session_state.components, t)

    window_ref = np.ones_like(t)
    freqs, mag, phase, spectrum = fft_spectrum(total, fs_effective, window_ref, 1.0)
    freqs_pos, mag_pos, phase_pos = positive_freq_slice(freqs, mag, phase)

    tabs = st.tabs([
        "Signal Builder",
        "Sampling & Aliasing",
        "Windowing & Leakage",
        "Band Reconstruction",
    ])

    with tabs[0]:
        signal_builder_tab(
            t,
            total,
            pieces,
            active_components,
            latex_expr,
            freqs_pos,
            mag_pos,
            phase_pos,
            freqs,
            spectrum,
        )
    with tabs[1]:
        sampling_aliasing_tab(t, total, st.session_state.components)
    with tabs[2]:
        windowing_tab(t, st.session_state.components, fs_effective)
    with tabs[3]:
        band_reconstruction_tab(t, total, freqs_pos, mag_pos, freqs, spectrum)


if __name__ == "__main__":
    main()
