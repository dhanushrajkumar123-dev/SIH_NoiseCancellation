# SIH_NoiseCancellation

# Impulsive Noise Suppression

A laptop-friendly prototype for removing **impulsive noise** (gunshots, explosions, clanks) from speech recordings. It includes a mixing/visualization pipeline, a training script with SI-SNR loss, an objective validation tool, and an offline batch cleaner.

Developed and tested for CPU-only training on an Intel i5-13420H with 16 GB RAM.

---

## How it works

The core model is `ImpulseSuppressorNet`, a small (~1.1M parameter) time-domain network: a strided Conv1d encoder, a 2-layer bidirectional GRU, and a transpose-conv decoder. It predicts a **residual** (the impulsive component) and subtracts it from the input waveform. Working directly in the time domain avoids the frame-level smearing that STFT masking can introduce on short transients.

**Training loss** is SI-SNR (scale-invariant, so it penalizes waveform-shape distortion) plus a transient-weighted L1 term that up-weights regions around energy onsets, up to 5x, so the model is pushed specifically toward removing spikes.

> **Note:** `ImpulseSuppressorNet` is trained from scratch as a specialist spike-suppressor. It is not a fine-tuned DeepFilterNet. DeepFilterNet's official training pipeline uses its own dataloader and Hydra configs, so a `full_dfn` mode exists in the scripts as a best-effort, version-dependent option only. `lightweight` mode is the recommended path.

---

## Project structure

```
.
├── Dataset/
│   ├── Clean/                  # clean speech .wav files
│   └── Noise/                  # impulsive noise .wav files
├── task1_mix_visualize.py      # mixing + spectrogram comparison + audio export
├── task2_finetune.py           # training (SI-SNR + transient-weighted loss)
├── task3_validate.py           # SNR improvement / PESQ / STOI + spike flagging
├── task4_realtime.py           # optional: live mic -> headphones pipeline
├── checkpoints/                # trained weights (created by task2)
└── outputs/                    # generated audio, plots, CSVs
```

---

## Installation

```bash
pip install torch soundfile librosa numpy pandas matplotlib pesq pystoi tqdm
# only needed for the optional live pipeline:
pip install sounddevice
# only needed for the optional full_dfn mode:
pip install deepfilternet
```

Python 3.9+ is recommended.

---

## Workflow

### 1. Mix clean speech with impulsive noise (Task 1)

Single demo sample with spectrogram + waveform plots and exported `.wav` files:

```bash
python task1_mix_visualize.py --clean_dir Dataset/Clean --noise_dir Dataset/Noise \
    --snr_db 5 --out_dir outputs/demo
```

Batch-mix the whole dataset across SNR levels (produces `manifest.csv` used by training and validation):

```bash
python task1_mix_visualize.py --batch --clean_dir Dataset/Clean --noise_dir Dataset/Noise \
    --snr_levels -5 0 5 10 15 --out_dir outputs/mixed_dataset
```

A single impulse is placed at a random offset within a zero buffer rather than tiled, so each mixture has one realistic bang instead of repeated artificial ones.

### 2. Train the model (Task 2)

```bash
python task2_finetune.py --manifest outputs/mixed_dataset/manifest.csv \
    --mode lightweight --epochs 8 --batch_size 4 --segment_seconds 2.0
```

Laptop-safe defaults:

| Setting | Value |
|---|---|
| Batch size | 4 (with 4 gradient-accumulation steps, effective batch of 16) |
| Segment length | 2.0 s |
| Epochs | 8 (small dataset, so watch validation loss for overfitting) |
| Learning rate | 1e-3 (AdamW) |
| DataLoader workers | 0 |

The best checkpoint (by validation loss) is saved to `checkpoints/lightweight_best.pt`.

### 3. Validate objectively (Task 3)

```bash
python task3_validate.py --manifest outputs/mixed_dataset/manifest.csv \
    --mode lightweight --checkpoint checkpoints/lightweight_best.pt \
    --out_csv outputs/validation_results.csv
```

Reports per file: noisy SNR, enhanced SNR, **SNR improvement**, **PESQ**, **STOI**, and a **spike flag**. Run once with `--mode passthrough` first to get a baseline on the raw noisy files.

The spike flag works at the frame level (10 ms). It finds frames where the noisy input exceeds the clean reference by more than 10 dB, then checks whether the enhanced output is still more than 10 dB above clean at those same frames. This catches a missed gunshot that whole-file PESQ/STOI averages would hide. Adjust with `--spike_db_threshold`.

### 4. Clean pre-recorded audio (batch_clean.py)

```bash
python batch_clean.py --input_dir inputs/noisy_files --output_dir outputs/cleaned_files \
    --checkpoint checkpoints/lightweight_best.pt
```

Scans the input folder for `.wav` files, converts to mono, resamples, runs the model on CPU, clips output to `[-1.0, 1.0]`, and saves with the same filename. Corrupted files are logged and skipped without stopping the batch.

> **Sample rate warning:** the model was trained on **16 kHz** audio, but `batch_clean.py` is configured for 48 kHz per its spec. Feeding 48 kHz audio to a 16 kHz-trained model degrades results. If output sounds worse than expected, set `MODEL_SR = 16000` near the top of `batch_clean.py`. The model then runs at its native rate and the output is resampled back up to 48 kHz for saving.

---

## Optional scripts

**`task5_offline_enhance.py`** processes a file or folder with the option to save spectrogram comparison plots (`--plot`). Short files run in one pass; long files are split into overlapping chunks and crossfaded with a Hann window.

**`task4_realtime.py`** runs a live microphone-to-headphones pipeline using a two-thread producer/consumer design and a rolling context buffer. List devices with `--list_devices`, then pass `--input_device` and `--output_device`. If you hear underruns, increase `--block_ms`.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Cleaned audio sounds muffled or wrong | Sample-rate mismatch. Run the model at 16 kHz (see the warning above). |
| `ModuleNotFoundError: task2_finetune` | Keep all scripts in the same directory. Several import the model class from `task2_finetune.py`. |
| PESQ or STOI shows `nan` | Install `pesq` and `pystoi`. PESQ requires 16 kHz (wideband) or 8 kHz (narrowband). |
| Out of memory during training | Lower `--batch_size` or `--segment_seconds`. |
| Many files flagged `RESIDUAL SPIKE` | Train longer, add more noise variety, or raise `--spike_db_threshold` if the check is too strict for your data. |
| Loading a checkpoint fails | Confirm it was saved by `task2_finetune.py` in `lightweight` mode. |

---

## Limitations

- The dataset is small by design (prototype), so expect overfitting to the specific noise types in `Dataset/Noise`. Evaluate on held-out clean speakers and noise clips where possible.
- The residual-subtraction approach targets brief, high-energy transients. It is not intended for stationary or broadband noise.
- PESQ and STOI are computed against the clean reference, so they are only meaningful on synthetic mixtures where that reference exists.
