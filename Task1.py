"""
Task 1: Impulsive Noise Mixing & Spectrogram Visualization Pipeline
=====================================================================

Folder structure expected:
    Dataset/Clean/*.wav      -> clean speech files
    Dataset/Noise/*.wav      -> impulsive noise files (gunshots, clanks, explosions)

What this script does:
    1. Loads a clean speech file + an impulsive noise file.
    2. Mixes them at a target SNR (dB) -> "noisy" file.
    3. Runs the noisy file through your enhancement model (DeepFilterNet
       hook provided below - currently a passthrough stub so the script
       runs end-to-end before you wire up Task 2's fine-tuned model).
    4. Plots Clean / Noisy / Enhanced spectrograms side by side.
    5. Exports clean.wav, noisy.wav, enhanced.wav so you can listen.

Usage:
    python task1_mix_visualize.py --clean_dir Dataset/Clean --noise_dir Dataset/Noise \
        --snr_db 5 --out_dir outputs/task1

    Batch-mix an entire dataset at multiple SNR levels (no plotting, just export):
    python task1_mix_visualize.py --batch --clean_dir Dataset/Clean --noise_dir Dataset/Noise \
        --snr_levels -5 0 5 10 15 --out_dir outputs/mixed_dataset
"""

import argparse
import random
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Core audio utilities
# --------------------------------------------------------------------------- #

def load_audio(path: Path, sr: int = 16000) -> np.ndarray:
    """Load a mono audio file resampled to `sr`."""
    audio, _ = librosa.load(str(path), sr=sr, mono=True)
    return audio


def rms(x: np.ndarray) -> float:
    return np.sqrt(np.mean(x ** 2) + 1e-12)


def fit_noise_to_length(noise: np.ndarray, target_len: int, place: str = "random") -> np.ndarray:
    """
    Impulsive noise clips are often much shorter than the speech clip
    (a gunshot might be 200ms vs 4s of speech). Instead of tiling/looping
    the impulse (which would fabricate multiple repeated bangs and bias
    training), we place ONE impulse at a random (or specified) position
    inside a zero buffer the length of the speech.

    If the noise clip is longer than the speech, it's randomly cropped.
    """
    out = np.zeros(target_len, dtype=np.float32)

    if len(noise) >= target_len:
        # Crop a random window of noise to fit
        start = random.randint(0, len(noise) - target_len) if place == "random" else 0
        out[:] = noise[start:start + target_len]
        return out

    # Noise shorter than speech -> place it at a random offset (or centered)
    if place == "random":
        start = random.randint(0, target_len - len(noise))
    elif place == "start":
        start = 0
    elif place == "center":
        start = (target_len - len(noise)) // 2
    else:
        start = 0

    out[start:start + len(noise)] = noise
    return out


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale `noise` so that mixing with `clean` achieves the target SNR (dB),
    computed over the full clean signal's RMS. Returns (mixed, scaled_noise).

    SNR(dB) = 20 * log10(rms_clean / rms_noise)
    => rms_noise_target = rms_clean / (10^(SNR/20))
    """
    clean_rms = rms(clean)
    noise_rms = rms(noise) if rms(noise) > 1e-8 else 1e-8

    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    scale = target_noise_rms / noise_rms
    scaled_noise = noise * scale

    mixed = clean + scaled_noise

    # Prevent clipping - normalize if needed, apply same gain to keep SNR ratio intact
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        gain = 0.99 / peak
        mixed = mixed * gain
        # NOTE: we intentionally do NOT rescale clean/scaled_noise references used
        # for metric computation elsewhere; if you need bit-exact post-clip SNR,
        # recompute rms(mixed - clean) downstream.

    return mixed.astype(np.float32), scaled_noise.astype(np.float32)


def measure_snr_db(clean: np.ndarray, mixed: np.ndarray) -> float:
    """Empirical SNR of a mixed signal relative to a clean reference."""
    noise_est = mixed[:len(clean)] - clean
    c = rms(clean)
    n = rms(noise_est) if rms(noise_est) > 1e-8 else 1e-8
    return 20 * np.log10(c / n)


# --------------------------------------------------------------------------- #
# Enhancement hook (plug in DeepFilterNet here)
# --------------------------------------------------------------------------- #

def enhance_audio(noisy: np.ndarray, sr: int = 16000, model=None) -> np.ndarray:
    """
    Hook for your enhancement model. Replace the passthrough body below with
    a real DeepFilterNet call once Task 2's fine-tuned checkpoint is ready.

    Example (DeepFilterNet Python API, once installed via `pip install deepfilternet`):

        from df.enhance import enhance, init_df, load_audio, save_audio
        model, df_state, _ = init_df(model_base_dir="path/to/checkpoint")
        audio_tensor = torch.from_numpy(noisy).unsqueeze(0)
        enhanced_tensor = enhance(model, df_state, audio_tensor)
        return enhanced_tensor.squeeze(0).numpy()

    For now this is a passthrough stub (returns noisy unchanged) so the full
    pipeline -- mixing, plotting, export -- is runnable and testable today.
    """
    if model is not None:
        # model is expected to expose a `.enhance(np.ndarray, sr) -> np.ndarray` method
        return model.enhance(noisy, sr)

    print("[enhance_audio] WARNING: no model provided - returning noisy audio unchanged "
          "(passthrough stub). Wire up DeepFilterNet here once Task 2 is done.")
    return noisy.copy()


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #

def plot_spectrogram_comparison(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    sr: int,
    out_path: Path,
    n_fft: int = 1024,
    hop_length: int = 256,
    title_suffix: str = "",
):
    """Side-by-side dB-scaled STFT spectrograms for Clean / Noisy / Enhanced."""
    signals = {"Clean": clean, "Noisy (Mixed)": noisy, "Enhanced": enhanced}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    # Compute a shared color scale across all three for fair visual comparison
    specs = {}
    vmax = -np.inf
    vmin = np.inf
    for name, sig in signals.items():
        S = librosa.stft(sig, n_fft=n_fft, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        specs[name] = S_db
        vmax = max(vmax, S_db.max())
        vmin = min(vmin, S_db.min())

    for ax, (name, S_db) in zip(axes, specs.items()):
        img = librosa.display.specshow(
            S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz",
            ax=ax, vmin=vmin, vmax=vmax, cmap="magma"
        )
        ax.set_title(f"{name} {title_suffix}".strip())

    fig.colorbar(img, ax=axes, format="%+2.0f dB", fraction=0.02, pad=0.02)
    fig.suptitle("Spectrogram Comparison: Clean vs Noisy vs Enhanced", fontsize=14)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved spectrogram comparison -> {out_path}")


def plot_waveform_with_impulse_highlight(
    clean: np.ndarray, noisy: np.ndarray, enhanced: np.ndarray, sr: int, out_path: Path
):
    """
    Extra diagnostic: overlays waveforms so you can visually confirm the
    impulsive spike is present in 'noisy' and suppressed in 'enhanced'.
    """
    t = np.arange(len(noisy)) / sr
    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True, sharey=True)
    for ax, sig, name, color in zip(
        axes, [clean, noisy, enhanced], ["Clean", "Noisy", "Enhanced"], ["tab:blue", "tab:red", "tab:green"]
    ):
        ax.plot(t[: len(sig)], sig[: len(t)], color=color, linewidth=0.7)
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Waveform View (impulse spikes should shrink Noisy -> Enhanced)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved waveform comparison -> {out_path}")


# --------------------------------------------------------------------------- #
# Single-sample demo pipeline (Task 1's main deliverable)
# --------------------------------------------------------------------------- #

def run_single_sample(
    clean_path: Path,
    noise_path: Path,
    snr_db: float,
    out_dir: Path,
    sr: int = 16000,
    model=None,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    clean = load_audio(clean_path, sr=sr)
    noise = load_audio(noise_path, sr=sr)
    noise_fit = fit_noise_to_length(noise, len(clean), place="random")

    mixed, scaled_noise = mix_at_snr(clean, noise_fit, snr_db)
    actual_snr = measure_snr_db(clean, mixed)
    print(f"[mix] Target SNR: {snr_db:+.1f} dB | Achieved SNR: {actual_snr:+.2f} dB")

    enhanced = enhance_audio(mixed, sr=sr, model=model)
    # Enhanced output must match clean length for fair comparison/metrics
    min_len = min(len(clean), len(mixed), len(enhanced))
    clean, mixed, enhanced = clean[:min_len], mixed[:min_len], enhanced[:min_len]

    # Export audio
    sf.write(out_dir / "clean.wav", clean, sr)
    sf.write(out_dir / "noisy.wav", mixed, sr)
    sf.write(out_dir / "enhanced.wav", enhanced, sr)
    print(f"[export] Saved clean.wav / noisy.wav / enhanced.wav -> {out_dir}")

    # Visualize
    plot_spectrogram_comparison(
        clean, mixed, enhanced, sr,
        out_dir / "spectrogram_comparison.png",
        title_suffix=f"(SNR={snr_db:+.0f}dB)"
    )
    plot_waveform_with_impulse_highlight(
        clean, mixed, enhanced, sr, out_dir / "waveform_comparison.png"
    )


# --------------------------------------------------------------------------- #
# Batch mixing (build a full noisy training/eval set across SNR levels)
# --------------------------------------------------------------------------- #

def run_batch_mix(
    clean_dir: Path,
    noise_dir: Path,
    snr_levels: list,
    out_dir: Path,
    sr: int = 16000,
    seed: int = 42,
):
    random.seed(seed)
    clean_files = sorted(clean_dir.glob("*.wav"))
    noise_files = sorted(noise_dir.glob("*.wav"))

    if not clean_files:
        raise FileNotFoundError(f"No .wav files found in {clean_dir}")
    if not noise_files:
        raise FileNotFoundError(f"No .wav files found in {noise_dir}")

    manifest = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for clean_path in clean_files:
        clean = load_audio(clean_path, sr=sr)
        for snr_db in snr_levels:
            noise_path = random.choice(noise_files)
            noise = load_audio(noise_path, sr=sr)
            noise_fit = fit_noise_to_length(noise, len(clean), place="random")
            mixed, _ = mix_at_snr(clean, noise_fit, snr_db)

            stem = f"{clean_path.stem}__{noise_path.stem}__snr{snr_db:+d}dB"
            mixed_path = out_dir / f"{stem}.wav"
            sf.write(mixed_path, mixed, sr)

            manifest.append({
                "clean": str(clean_path),
                "noise": str(noise_path),
                "mixed": str(mixed_path),
                "snr_db": snr_db,
            })

    # Write manifest CSV for downstream training/eval scripts
    import csv
    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clean", "noise", "mixed", "snr_db"])
        writer.writeheader()
        writer.writerows(manifest)

    print(f"[batch] Mixed {len(manifest)} files across {len(snr_levels)} SNR levels -> {out_dir}")
    print(f"[batch] Manifest written -> {manifest_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Impulsive noise mixing & spectrogram visualization")
    parser.add_argument("--clean_dir", type=str, default="Dataset/Clean")
    parser.add_argument("--noise_dir", type=str, default="Dataset/Noise")
    parser.add_argument("--out_dir", type=str, default="outputs/task1")
    parser.add_argument("--sr", type=int, default=16000)

    # Single-sample mode
    parser.add_argument("--clean_file", type=str, default=None,
                         help="Specific clean file to use (default: random pick from clean_dir)")
    parser.add_argument("--noise_file", type=str, default=None,
                         help="Specific noise file to use (default: random pick from noise_dir)")
    parser.add_argument("--snr_db", type=float, default=5.0)

    # Batch mode
    parser.add_argument("--batch", action="store_true", help="Run batch mixing instead of single-sample demo")
    parser.add_argument("--snr_levels", type=int, nargs="+", default=[-5, 0, 5, 10, 15])

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    clean_dir = Path(args.clean_dir)
    noise_dir = Path(args.noise_dir)
    out_dir = Path(args.out_dir)

    if args.batch:
        run_batch_mix(clean_dir, noise_dir, args.snr_levels, out_dir, sr=args.sr, seed=args.seed)
        return

    clean_files = sorted(clean_dir.glob("*.wav"))
    noise_files = sorted(noise_dir.glob("*.wav"))
    if not clean_files or not noise_files:
        raise FileNotFoundError(
            f"Need .wav files in both {clean_dir} and {noise_dir}. "
            f"Found {len(clean_files)} clean, {len(noise_files)} noise."
        )

    clean_path = Path(args.clean_file) if args.clean_file else random.choice(clean_files)
    noise_path = Path(args.noise_file) if args.noise_file else random.choice(noise_files)

    print(f"[select] Clean:  {clean_path}")
    print(f"[select] Noise:  {noise_path}")

    run_single_sample(clean_path, noise_path, args.snr_db, out_dir, sr=args.sr, model=None)


if __name__ == "__main__":
    main()
    