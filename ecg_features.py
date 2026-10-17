"""ECG feature extraction for the PTB-XL normal-vs-abnormal project.

One function turns a (samples x leads) ECG array into a flat dict of features.
The same code is used for training (notebook) and for inference (Streamlit app),
so the prototype can never drift away from the training pipeline.
"""
import numpy as np
from scipy import signal, stats

FS = 100                      # all signals are processed at 100 Hz
BANDS = {                     # frequency bands (Hz) for Welch band power
    "baseline": (0.5, 5),     # P/T waves, slow components
    "qrs": (5, 15),           # QRS complex
    "high": (15, 40),         # fast components / noise
}


def bandpass(x, fs=FS, low=0.5, high=40.0, order=3):
    """Zero-phase band-pass filter: removes baseline wander and high-frequency noise."""
    sos = signal.butter(order, [low, high], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def statistical_features(x, prefix):
    """Time-domain statistics. x stays in mV so amplitude keeps its clinical meaning."""
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_range": float(np.ptp(x)),
        f"{prefix}_rms": float(np.sqrt(np.mean(x ** 2))),
        f"{prefix}_energy": float(np.sum(x ** 2) / len(x)),
        f"{prefix}_skew": float(stats.skew(x)),
        f"{prefix}_kurtosis": float(stats.kurtosis(x)),
    }


def frequency_features(x, prefix, fs=FS):
    """Welch power spectral density -> band powers, dominant frequency, centroid, entropy."""
    freqs, psd = signal.welch(x, fs=fs, nperseg=min(256, len(x)))
    total = psd.sum() + 1e-12
    feats = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        feats[f"{prefix}_bp_{name}"] = float(psd[mask].sum() / total)   # relative band power
    p = psd / total
    feats[f"{prefix}_dom_freq"] = float(freqs[np.argmax(psd)])
    feats[f"{prefix}_spec_centroid"] = float(np.sum(freqs * p))
    feats[f"{prefix}_spec_entropy"] = float(-np.sum(p * np.log2(p + 1e-12)))
    return feats


def beat_features(x, prefix, fs=FS):
    """R-peak based features: heart rate and heart-rate variability. NaN if not reliable."""
    names = ["n_beats", "hr_mean", "rr_mean", "rr_std", "rmssd", "pnn50", "rr_cv",
             "r_amp_mean", "r_amp_std"]
    nan = {f"{prefix}_{n}": np.nan for n in names}
    # R peaks are usually positive; if the negative deflection dominates, flip the polarity
    xp = x if abs(x.max()) >= abs(x.min()) else -x
    if xp.max() <= 0:
        return nan
    peaks, _ = signal.find_peaks(xp, distance=int(0.25 * fs), height=0.5 * xp.max())
    if len(peaks) < 3:
        return nan
    rr = np.diff(peaks) / fs                       # seconds between beats
    drr = np.diff(rr)
    amps = xp[peaks]
    return {
        f"{prefix}_n_beats": float(len(peaks)),
        f"{prefix}_hr_mean": float(60.0 / rr.mean()),
        f"{prefix}_rr_mean": float(rr.mean()),
        f"{prefix}_rr_std": float(rr.std()),                              # SDNN
        f"{prefix}_rmssd": float(np.sqrt(np.mean(drr ** 2))) if len(drr) else np.nan,
        f"{prefix}_pnn50": float(np.mean(np.abs(drr) > 0.05)) if len(drr) else np.nan,
        f"{prefix}_rr_cv": float(rr.std() / rr.mean()),
        f"{prefix}_r_amp_mean": float(amps.mean()),
        f"{prefix}_r_amp_std": float(amps.std()),
    }


def extract_features(sig, sig_names, fs, leads=("II",), full_set_leads=("II",)):
    """sig: (n_samples, n_leads) array in mV. Returns a dict of features.

    leads          -> leads that get basic statistical + frequency features
    full_set_leads -> leads that additionally get R-peak / HRV features
    """
    sig = np.asarray(sig, dtype=float)
    if fs != FS:                                   # e.g. 500 Hz records -> 100 Hz
        n_new = int(round(sig.shape[0] * FS / fs))
        sig = signal.resample(sig, n_new, axis=0)
    feats = {}
    for lead in leads:
        if lead not in sig_names:
            raise ValueError(f"Lead {lead} not found in {sig_names}")
        x = bandpass(np.nan_to_num(sig[:, sig_names.index(lead)]))
        feats.update(statistical_features(x, lead))
        feats.update(frequency_features(x, lead))
        if lead in full_set_leads:
            feats.update(beat_features(x, lead))
    return feats
