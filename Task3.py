"""
Task 3: Objective Validation Script
=====================================

Runs a trained enhancement model over a test set of (clean, noisy) pairs and
reports objective quality metrics, plus flags files where an impulsive spike
was NOT fully removed (which averaged metrics like PESQ/STOI can hide, since
a single missed spike barely moves a whole-file average).

Metrics reported per file:
    - Noisy SNR        : SNR of the noisy input vs clean reference
    - Enhanced SNR      : SNR of the model's output vs clean reference
    - SNR Improvement  : Enhanced SNR - Noisy SNR  (higher = better)
    - PESQ             : Perceptual quality, range ~[-0.5, 4.5] (higher = better)
    - STOI             : Intelligibility, range [0, 1] (higher = better)
    - Spike Flag       : "OK" or "RESIDUAL SPIKE" (see detection logic below)

Requirements:
    pip install torch soundfile librosa numpy pandas pesq pystoi tabulate

Usage:
    # Using the Task 2 lightweight checkpoint:
    python task3_validate.py --manifest outputs/mixed_dataset/manifest.csv \
        --mode lightweight --checkpoint checkpoints/lightweight_best.pt \
        --out_csv outputs/validation_results.csv

    # Passthrough / no model (e.g. to benchmark the raw noisy files as a baseline):
    python task3_validate.py --manifest outputs/mixed_dataset/manifest.csv --mode passthrough

Notes on the manifest:
    Expects the same manifest.csv schema produced by Task 1's --batch mode:
        clean, noise, mixed, snr_db
    "mixed" is treated as the noisy test input; "clean" is the reference.
    If you have a separate held-out test manifest, just point --manifest at it --
    it only needs "clean" and "mixed" columns.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F

try:
    from pesq import pesq as pesq_fn
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False

try:
    from pystoi import stoi as stoi_fn
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Audio + SNR utilities (consistent with Task 1 / Task 2)
# --------------------------------------------------------------------------- #

def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    audio, file_sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    return audio


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def measure_snr_db(clean: np.ndarray, other: np.ndarray) -> float:
    """SNR of `other` (noisy or enhanced) relative to `clean`, in dB."""
    n = min(len(clean), len(other))
    noise_est = other[:n] - clean[:n]
    c = rms(clean[:n])
    ne = rms(noise_est) if rms(noise_est) > 1e-8 else 1e-8
    return 20 * np.log10(c / ne)


def align_lengths(*signals: np.ndarray):
    n = min(len(s) for s in signals)
    return [s[:n] for s in signals]


# --------------------------------------------------------------------------- #
# Residual impulsive-spike detection
# --------------------------------------------------------------------------- #

def detect_residual_spike(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    sr: int = 16000,
    frame_ms: float = 10.0,
    spike_db_threshold: float = 10.0,
) -> dict:
    """
    Flags whether an impulsive spike present in the noisy input survived into
    the enhanced output, at the SAME time location. This catches the exact
    failure mode a whole-file PESQ/STOI average can hide: one un-removed
    50ms gunshot barely dents a 5-second file's average score.

    Method:
        1. Frame all three signals into `frame_ms` windows.
        2. Locate impulse frames: where noisy energy exceeds clean energy by
           `spike_db_threshold` dB (i.e. where the injected noise clearly dominates).
        3. At those same frames, compare enhanced energy to clean energy.
           If enhanced is STILL `spike_db_threshold` dB above clean there,
           the spike leaked through -> flag it.

    Returns a dict with:
        flagged (bool), num_impulse_frames (int), num_residual_frames (int),
        worst_residual_db (float) -- the highest enhanced-vs-clean excess found
    """
    clean, noisy, enhanced = align_lengths(clean, noisy, enhanced)
    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = len(clean) // frame_len
    if n_frames == 0:
        return {"flagged": False, "num_impulse_frames": 0, "num_residual_frames": 0,
                "worst_residual_db": 0.0}

    def frame_energy_db(x):
        trimmed = x[: n_frames * frame_len].reshape(n_frames, frame_len)
        energy = np.mean(trimmed ** 2, axis=-1) + 1e-10
        return 10 * np.log10(energy)

    clean_db = frame_energy_db(clean)
    noisy_db = frame_energy_db(noisy)
    enhanced_db = frame_energy_db(enhanced)

    impulse_frames = np.where((noisy_db - clean_db) > spike_db_threshold)[0]
    if len(impulse_frames) == 0:
        return {"flagged": False, "num_impulse_frames": 0, "num_residual_frames": 0,
                "worst_residual_db": 0.0}

    residual_excess = enhanced_db[impulse_frames] - clean_db[impulse_frames]
    residual_frames = impulse_frames[residual_excess > spike_db_threshold]
    worst = float(residual_excess.max()) if len(residual_excess) else 0.0

    return {
        "flagged": len(residual_frames) > 0,
        "num_impulse_frames": int(len(impulse_frames)),
        "num_residual_frames": int(len(residual_frames)),
        "worst_residual_db": round(worst, 2),
    }


# --------------------------------------------------------------------------- #
# Model loading (mirrors Task 2's architectures)
# --------------------------------------------------------------------------- #

def load_enhancer(mode: str, checkpoint: str = None, device: str = "cpu"):
    """
    Returns a callable enhance_fn(noisy_np: np.ndarray, sr: int) -> np.ndarray

    mode="passthrough" : returns noisy unchanged (baseline / sanity check)
    mode="lightweight"  : loads the Task 2 ImpulseSuppressorNet checkpoint
    mode="full_dfn"      : loads a fine-tuned DeepFilterNet checkpoint
    """
    if mode == "passthrough":
        return lambda noisy, sr: noisy.copy()

    if mode == "lightweight":
        from task2_finetune import ImpulseSuppressorNet  # reuse Task 2's architecture

        model = ImpulseSuppressorNet().to(device)
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location=device)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state)
            print(f"[model] Loaded lightweight checkpoint: {checkpoint} "
                  f"(val_loss={ckpt.get('val_loss', 'n/a')})")
        else:
            print("[model] WARNING: no checkpoint provided, using randomly initialized "
                  "lightweight model (results will be meaningless).")
        model.eval()

        @torch.no_grad()
        def enhance_fn(noisy, sr):
            x = torch.from_numpy(noisy).unsqueeze(0).to(device)
            out = model(x)
            return out.squeeze(0).cpu().numpy()

        return enhance_fn

    if mode == "full_dfn":
        try:
            from df.enhance import init_df, enhance
        except ImportError as e:
            raise RuntimeError(
                "deepfilternet not installed. Run `pip install deepfilternet` "
                "or use --mode lightweight / passthrough instead."
            ) from e

        model, df_state, _ = init_df(model_base_dir=checkpoint) if checkpoint else init_df()
        model.eval()

        def enhance_fn(noisy, sr):
            x = torch.from_numpy(noisy).unsqueeze(0)
            out = enhance(model, df_state, x)
            return out.squeeze(0).cpu().numpy()

        return enhance_fn

    raise ValueError(f"Unknown mode: {mode}")


# --------------------------------------------------------------------------- #
# Metric computation
# --------------------------------------------------------------------------- #

def compute_pesq(clean: np.ndarray, enhanced: np.ndarray, sr: int) -> float:
    if not PESQ_AVAILABLE:
        return float("nan")
    clean, enhanced = align_lengths(clean, enhanced)
    pesq_mode = "wb" if sr == 16000 else "nb"  # PESQ only supports 16k (wb) or 8k (nb)
    try:
        return float(pesq_fn(sr, clean, enhanced, pesq_mode))
    except Exception as e:
        print(f"  [pesq] failed: {e}")
        return float("nan")


def compute_stoi(clean: np.ndarray, enhanced: np.ndarray, sr: int) -> float:
    if not STOI_AVAILABLE:
        return float("nan")
    clean, enhanced = align_lengths(clean, enhanced)
    try:
        return float(stoi_fn(clean, enhanced, sr, extended=False))
    except Exception as e:
        print(f"  [stoi] failed: {e}")
        return float("nan")


# --------------------------------------------------------------------------- #
# Main validation loop
# --------------------------------------------------------------------------- #

def run_validation(args):
    df = pd.read_csv(args.manifest)
    required_cols = {"clean", "mixed"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Manifest must contain columns {required_cols}, found {list(df.columns)}")

    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    enhance_fn = load_enhancer(args.mode, args.checkpoint, device=device)

    results = []
    print(f"[validate] Running {len(df)} test files through mode='{args.mode}'...")

    for idx, row in df.iterrows():
        clean_path, noisy_path = row["clean"], row["mixed"]
        try:
            clean = load_audio(clean_path, sr=args.sr)
            noisy = load_audio(noisy_path, sr=args.sr)
            clean, noisy = align_lengths(clean, noisy)

            enhanced = enhance_fn(noisy, args.sr)
            clean_a, noisy_a, enhanced_a = align_lengths(clean, noisy, enhanced)

            noisy_snr = measure_snr_db(clean_a, noisy_a)
            enhanced_snr = measure_snr_db(clean_a, enhanced_a)
            snr_improvement = enhanced_snr - noisy_snr

            pesq_score = compute_pesq(clean_a, enhanced_a, args.sr)
            stoi_score = compute_stoi(clean_a, enhanced_a, args.sr)

            spike_result = detect_residual_spike(
                clean_a, noisy_a, enhanced_a, sr=args.sr,
                spike_db_threshold=args.spike_db_threshold,
            )

            results.append({
                "file": Path(noisy_path).name,
                "snr_db_target": row.get("snr_db", float("nan")),
                "noisy_snr": round(noisy_snr, 2),
                "enhanced_snr": round(enhanced_snr, 2),
                "snr_improvement": round(snr_improvement, 2),
                "pesq": round(pesq_score, 3) if not np.isnan(pesq_score) else float("nan"),
                "stoi": round(stoi_score, 3) if not np.isnan(stoi_score) else float("nan"),
                "spike_flag": "RESIDUAL SPIKE" if spike_result["flagged"] else "OK",
                "num_impulse_frames": spike_result["num_impulse_frames"],
                "num_residual_frames": spike_result["num_residual_frames"],
                "worst_residual_db": spike_result["worst_residual_db"],
            })

        except Exception as e:
            print(f"[error] Failed on {noisy_path}: {e}")
            results.append({
                "file": Path(noisy_path).name, "snr_db_target": row.get("snr_db", float("nan")),
                "noisy_snr": float("nan"), "enhanced_snr": float("nan"),
                "snr_improvement": float("nan"), "pesq": float("nan"), "stoi": float("nan"),
                "spike_flag": "ERROR", "num_impulse_frames": 0, "num_residual_frames": 0,
                "worst_residual_db": float("nan"),
            })

    results_df = pd.DataFrame(results)

    # --- Print summary table ---
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", None)
    print("\n" + "=" * 100)
    print("VALIDATION RESULTS")
    print("=" * 100)
    print(results_df.to_string(index=False))

    # --- Aggregate stats ---
    numeric_cols = ["noisy_snr", "enhanced_snr", "snr_improvement", "pesq", "stoi"]
    valid = results_df[results_df["spike_flag"] != "ERROR"]
    print("\n" + "-" * 100)
    print("SUMMARY (mean over valid files)")
    print("-" * 100)
    for col in numeric_cols:
        vals = valid[col].dropna()
        if len(vals):
            print(f"  {col:20s}: mean={vals.mean():.3f}  std={vals.std():.3f}  "
                  f"min={vals.min():.3f}  max={vals.max():.3f}")

    n_flagged = (results_df["spike_flag"] == "RESIDUAL SPIKE").sum()
    n_errors = (results_df["spike_flag"] == "ERROR").sum()
    print(f"\n  Files with residual (un-removed) spikes: {n_flagged} / {len(results_df)}")
    if n_errors:
        print(f"  Files that errored during processing:     {n_errors} / {len(results_df)}")

    if n_flagged:
        print("\n  Flagged files (spike survived enhancement):")
        for _, r in results_df[results_df["spike_flag"] == "RESIDUAL SPIKE"].iterrows():
            print(f"    - {r['file']}  (worst residual: {r['worst_residual_db']} dB above clean floor)")

    if not PESQ_AVAILABLE:
        print("\n  [note] PESQ not computed - install with `pip install pesq`")
    if not STOI_AVAILABLE:
        print("  [note] STOI not computed - install with `pip install pystoi`")

    # --- Save CSV ---
    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_path, index=False)
        print(f"\n[saved] Full results -> {out_path}")

    return results_df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Task 3: Objective validation (SNR / PESQ / STOI + spike flagging)")
    parser.add_argument("--manifest", type=str, required=True,
                         help="CSV with at least 'clean' and 'mixed' columns")
    parser.add_argument("--mode", type=str, choices=["passthrough", "lightweight", "full_dfn"],
                         default="lightweight")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to model checkpoint (required for lightweight/full_dfn unless "
                              "you want to sanity-check with random/default weights)")
    parser.add_argument("--sr", type=int, default=16000, help="PESQ requires 16000 (wb) or 8000 (nb)")
    parser.add_argument("--spike_db_threshold", type=float, default=10.0,
                         help="dB excess above clean-frame energy to count a frame as an impulse / residual")
    parser.add_argument("--out_csv", type=str, default="outputs/validation_results.csv")
    parser.add_argument("--force_cpu", action="store_true")
    args = parser.parse_args()

    run_validation(args)


if __name__ == "__main__":
    main()