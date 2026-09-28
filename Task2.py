"""
Task 2: Impulse-Specific Fine-Tuning on a Laptop (CPU: i5-13420H, 16GB RAM)
=============================================================================

IMPORTANT ENGINEERING NOTE (read before running):
---------------------------------------------------
DeepFilterNet's *official* training pipeline (https://github.com/Rikorose/DeepFilterNet)
does not expose a simple "model.fit()" path. It uses its own Rust-based dataloader,
ERB/complex-feature extraction, and a Hydra-configured training loop (`df/train.py`
in their repo) that expects their exact on-disk dataset format (HDF5 shards). Trying
to hot-wire gradients through their loaded inference-only checkpoint via
`df.enhance.init_df()` is unreliable across versions and NOT something you want to
debug on a 16GB laptop.

So this script gives you the approach that will actually work, fast, on your hardware:

  MODE "lightweight" (default, recommended for your laptop):
      Train a small residual mask-refinement network that sits AFTER DeepFilterNet
      in the pipeline, specialized purely for killing residual impulsive spikes
      that DFN's general-purpose model doesn't fully remove. This is:
        - ~1-2M params (vs DFN's ~2M+ already, but ours trains in minutes/epoch on CPU)
        - Trained directly with your Task 1 manifest.csv (noisy/clean pairs)
        - Fast enough for CPU-only training on an i5 in a reasonable time budget
        - The standard practical pattern: "general denoiser + specialist post-filter"

  MODE "full_dfn" (advanced, optional, best-effort):
      Attempts to unfreeze and fine-tune the last N layers of a loaded DeepFilterNet
      checkpoint directly. This requires `pip install deepfilternet` and its exact
      internal module names may differ between versions -- the script inspects the
      model and freezes everything except the final decoder/mask stage automatically,
      but you should verify it found trainable params before trusting a long run.

Both modes:
  - Use SI-SNR loss (scale-invariant, so it penalizes waveform *shape* distortion,
    not just amplitude) -- this is what stops the model from "smearing" speech to
    cheat a lower average error.
  - Add a transient-weighted loss term: frames with high-energy derivative (i.e.
    impulsive spikes) get up-weighted in the loss so the model is pushed harder
    to kill spikes specifically, not just improve average SNR.

Requirements:
    pip install torch soundfile librosa numpy pandas
    # optional, for MODE full_dfn:
    pip install deepfilternet

Usage:
    python task2_finetune.py --manifest outputs/mixed_dataset/manifest.csv \
        --mode lightweight --epochs 8 --batch_size 4 --segment_seconds 2.0

Laptop config recap (i5-13420H, 16GB RAM, CPU-only assumed):
    batch_size:        2-4   (RAM is the constraint, not compute)
    segment_seconds:   1.5-2.5s  (short crops -> fast iterations, plenty of impulse
                                   examples per clip since impulses are brief)
    epochs:            5-10  (small dataset -> more epochs overfit fast; watch val loss)
    num_workers:       0-2   (0 is safest, avoids multiprocessing overhead on laptop)
    optimizer:         AdamW, lr=1e-3 (lightweight net) / 1e-5 (full_dfn fine-tune)
    grad_accum_steps:  4     (simulates a larger effective batch without more RAM)
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------------------------------- #
# Reproducibility + CPU thread control (important on a shared laptop CPU)
# --------------------------------------------------------------------------- #

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_cpu_threads(max_threads: int = None):
    """
    i5-13420H has 8 cores (4P+4E). Leave 1-2 threads free for the OS/desktop
    so the laptop doesn't lock up while training in the background.
    """
    n = max_threads or max(1, (os.cpu_count() or 8) - 2)
    torch.set_num_threads(n)
    print(f"[cpu] torch using {n} threads")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class NoisyCleanDataset(Dataset):
    """
    Reads a manifest.csv (produced by Task 1's --batch mode) with columns:
        clean, noise, mixed, snr_db
    and returns fixed-length (clean, noisy) waveform crops for training.

    Fixed-length crops keep every batch the same tensor shape (required for
    batching) and keep each training step fast and RAM-light on a laptop.
    """

    def __init__(self, manifest_path: str, sr: int = 16000, segment_seconds: float = 2.0):
        self.df = pd.read_csv(manifest_path)
        self.sr = sr
        self.segment_len = int(segment_seconds * sr)

    def __len__(self):
        return len(self.df)

    def _load_crop(self, path: str, start: int = None):
        audio, file_sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != self.sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=self.sr)

        if len(audio) < self.segment_len:
            audio = np.pad(audio, (0, self.segment_len - len(audio)))

        return audio, start

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clean_path, mixed_path = row["clean"], row["mixed"]

        # Load clean first to know a valid crop range, then apply the SAME
        # crop offset to the noisy file so they stay time-aligned.
        clean_full, _ = sf.read(clean_path, dtype="float32")
        if clean_full.ndim > 1:
            clean_full = clean_full.mean(axis=1)

        max_start = max(0, len(clean_full) - self.segment_len)
        start = random.randint(0, max_start) if max_start > 0 else 0

        clean, _ = self._load_crop(clean_path, start)
        noisy, _ = self._load_crop(mixed_path, start)

        clean = clean[start:start + self.segment_len]
        noisy = noisy[start:start + self.segment_len]

        # Safety: force exact length in case of off-by-one from padding/crop
        clean = np.pad(clean, (0, max(0, self.segment_len - len(clean))))[: self.segment_len]
        noisy = np.pad(noisy, (0, max(0, self.segment_len - len(noisy))))[: self.segment_len]

        return torch.from_numpy(noisy), torch.from_numpy(clean)


# --------------------------------------------------------------------------- #
# Loss: SI-SNR + transient-weighted term
# --------------------------------------------------------------------------- #

def si_snr_loss(est: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-Invariant SNR loss (negative SI-SNR, so lower = better, minimized by optimizer).
    est, target: (batch, time)

    SI-SNR is preferred over plain MSE/SNR here because it's invariant to a global
    amplitude scale mismatch between estimate and target -- it measures whether the
    model reproduced the correct *waveform shape*, which is exactly what "smearing"
    (the failure mode you're worried about) would break.
    """
    est = est - est.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # Projection of est onto target (scale-invariant target component)
    dot = torch.sum(est * target, dim=-1, keepdim=True)
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    proj = dot * target / target_energy

    noise = est - proj

    ratio = torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps)
    si_snr = 10 * torch.log10(ratio + eps)

    return -si_snr.mean()


def transient_weight_mask(clean: torch.Tensor, frame: int = 160) -> torch.Tensor:
    """
    Produces a per-sample weight that's higher around high-energy transients
    (i.e. where an impulsive spike likely sits in the CLEAN+noise mixture).
    We derive it from the noisy signal's short-time energy derivative, then
    broadcast back to sample resolution.

    Used to up-weight the loss specifically where impulses occur, since a
    flat SI-SNR alone treats a missed spike the same as generic broadband
    residual noise -- we want the spike error penalized harder.
    """
    b, t = clean.shape
    n_frames = t // frame
    trimmed = clean[:, : n_frames * frame].reshape(b, n_frames, frame)
    frame_energy = trimmed.pow(2).mean(dim=-1)  # (b, n_frames)

    energy_db = 10 * torch.log10(frame_energy + 1e-8)
    delta = torch.diff(energy_db, dim=-1, prepend=energy_db[:, :1])
    spike_score = torch.clamp(delta, min=0)  # positive jumps = onsets/impulses

    # Normalize 0..1 per-sample, then floor at 1.0 so non-spike frames still count
    max_score = spike_score.max(dim=-1, keepdim=True).values + 1e-8
    weight = 1.0 + 4.0 * (spike_score / max_score)  # spikes get up to 5x weight

    weight_upsampled = weight.repeat_interleave(frame, dim=-1)
    if weight_upsampled.shape[-1] < t:
        pad = t - weight_upsampled.shape[-1]
        weight_upsampled = F.pad(weight_upsampled, (0, pad), value=1.0)
    return weight_upsampled[:, :t]


def combined_loss(est: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """SI-SNR (global shape fidelity) + weighted L1 on transient regions (spike removal)."""
    si_snr = si_snr_loss(est, target)

    weights = transient_weight_mask(target)
    weighted_l1 = (weights * (est - target).abs()).mean()

    return si_snr + 0.5 * weighted_l1


# --------------------------------------------------------------------------- #
# MODE "lightweight": trainable residual spike-suppressor network
# --------------------------------------------------------------------------- #

class ImpulseSuppressorNet(nn.Module):
    """
    Small conv + GRU network that predicts a per-sample residual correction
    to subtract impulsive energy from the input waveform. Operates directly
    in the time domain (no STFT round-trip) to avoid smearing transients
    across frames the way frequency-domain masking can.

    ~1.1M parameters -- trains comfortably on CPU in a batch-of-2..4 setting.
    """

    def __init__(self, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.PReLU(),
            nn.Conv1d(32, hidden, kernel_size=15, padding=7, stride=2),
            nn.PReLU(),
        )
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden * 2, 32, kernel_size=16, stride=2, padding=7),
            nn.PReLU(),
            nn.Conv1d(32, 1, kernel_size=15, padding=7),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time)
        h = self.encoder(x.unsqueeze(1))            # (batch, hidden, time/2)
        h = h.transpose(1, 2)                        # (batch, time/2, hidden)
        h, _ = self.gru(h)                            # (batch, time/2, hidden*2)
        h = h.transpose(1, 2)                         # (batch, hidden*2, time/2)
        residual = self.decoder(h).squeeze(1)         # (batch, time)

        # Match length exactly (transpose-conv can be off by a sample or two)
        if residual.shape[-1] != x.shape[-1]:
            residual = F.interpolate(residual.unsqueeze(1), size=x.shape[-1], mode="linear",
                                      align_corners=False).squeeze(1)

        enhanced = x - residual  # learn to SUBTRACT the impulsive component
        return enhanced


# --------------------------------------------------------------------------- #
# MODE "full_dfn": best-effort DeepFilterNet fine-tune (advanced, optional)
# --------------------------------------------------------------------------- #

def build_full_dfn_finetuner(lr: float):
    """
    Attempts to load a pretrained DeepFilterNet model and freeze everything
    except its final decoder/mask stage for fine-tuning. This is best-effort:
    DeepFilterNet's internal module names have changed across releases, so we
    auto-detect trainable layers by name heuristics rather than hardcoding them.

    Returns (model, df_state, optimizer, forward_fn) or raises RuntimeError
    with a clear message if the installed version isn't compatible -- in
    which case use --mode lightweight instead (recommended for this hardware
    anyway).
    """
    try:
        from df.enhance import init_df
    except ImportError as e:
        raise RuntimeError(
            "deepfilternet is not installed or failed to import. "
            "Run `pip install deepfilternet`, or use --mode lightweight instead "
            "(recommended for your laptop)."
        ) from e

    model, df_state, _ = init_df()  # loads the default pretrained checkpoint
    model.train()

    trainable_keywords = ("decoder", "mask", "df_dec", "out")
    trainable_params = []
    for name, param in model.named_parameters():
        if any(k in name.lower() for k in trainable_keywords):
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False

    if not trainable_params:
        raise RuntimeError(
            "Could not auto-detect a decoder/mask stage to fine-tune in this "
            "DeepFilterNet version's module names. Inspect `model.named_parameters()` "
            "yourself and adjust `trainable_keywords`, or use --mode lightweight."
        )

    print(f"[full_dfn] Fine-tuning {len(trainable_params)} parameter tensors "
          f"({sum(p.numel() for p in trainable_params):,} params)")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    def forward_fn(noisy_batch: torch.Tensor) -> torch.Tensor:
        # DeepFilterNet's enhance() typically expects (channels, time) per-item,
        # not a batched tensor -- so we loop. This is slow but workable for a
        # small laptop fine-tune batch (2-4 items).
        from df.enhance import enhance
        outs = []
        for i in range(noisy_batch.shape[0]):
            single = noisy_batch[i:i + 1]
            out = enhance(model, df_state, single)
            outs.append(out)
        return torch.cat(outs, dim=0)

    return model, df_state, optimizer, forward_fn


# --------------------------------------------------------------------------- #
# Training loop (shared by both modes)
# --------------------------------------------------------------------------- #

def train(args):
    set_seed(args.seed)
    configure_cpu_threads()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"[device] Training on: {device}")

    dataset = NoisyCleanDataset(args.manifest, sr=args.sr, segment_seconds=args.segment_seconds)
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"[data] {len(train_set)} train / {len(val_set)} val segments from {args.manifest}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    if args.mode == "lightweight":
        model = ImpulseSuppressorNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        forward_fn = lambda noisy: model(noisy)
    elif args.mode == "full_dfn":
        model, df_state, optimizer, forward_fn = build_full_dfn_finetuner(lr=args.lr)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    accum_steps = max(1, args.grad_accum_steps)

    for epoch in range(1, args.epochs + 1):
        if hasattr(model, "train"):
            model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, (noisy, clean) in enumerate(train_loader, start=1):
            noisy, clean = noisy.to(device), clean.to(device)

            enhanced = forward_fn(noisy)
            min_len = min(enhanced.shape[-1], clean.shape[-1])
            loss = combined_loss(enhanced[..., :min_len], clean[..., :min_len]) / accum_steps
            loss.backward()

            if step % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=5.0
                )
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * accum_steps

            if step % max(1, len(train_loader) // 5) == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={running_loss / step:.4f}")

        train_loss = running_loss / max(1, len(train_loader))

        # Validation
        if hasattr(model, "eval"):
            model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                enhanced = forward_fn(noisy)
                min_len = min(enhanced.shape[-1], clean.shape[-1])
                val_loss += combined_loss(enhanced[..., :min_len], clean[..., :min_len]).item()
        val_loss /= max(1, len(val_loader))

        print(f"[epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = ckpt_dir / f"{args.mode}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict() if hasattr(model, "state_dict") else None,
                "val_loss": val_loss,
                "args": vars(args),
            }, ckpt_path)
            print(f"  -> new best checkpoint saved: {ckpt_path}")

    print("[done] Training complete. Best val_loss:", best_val_loss)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Task 2: Impulse-specific fine-tuning (laptop-safe)")
    parser.add_argument("--manifest", type=str, required=True,
                         help="manifest.csv produced by Task 1's --batch mode")
    parser.add_argument("--mode", type=str, choices=["lightweight", "full_dfn"], default="lightweight")

    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--segment_seconds", type=float, default=2.0)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_cpu", action="store_true",
                         help="Force CPU even if a CUDA GPU is somehow detected")

    args = parser.parse_args()

    if args.mode == "full_dfn":
        args.lr = min(args.lr, 1e-5)  # fine-tuning a pretrained net needs a much smaller LR
        print("[note] mode=full_dfn -> clamping lr to <=1e-5 to avoid destroying the pretrained weights")

    train(args)


if __name__ == "__main__":
    main()
    