"""
Task 4: Real-Time Impulsive Noise Suppression (Mic -> Model -> Headphones)
=============================================================================

Live pipeline: microphone -> rolling-context buffer -> enhancement model ->
headphone output. Designed to run continuously on a laptop CPU (i5-13420H)
without underruns, using a producer/consumer thread split so audio I/O
timing is never blocked by model compute.

ARCHITECTURE
------------
    [mic] --InputStream callback--> [input_queue] --processing thread-->
        model.enhance(rolling_buffer) --> [output_queue] --OutputStream callback--> [headphones]

Why a rolling-context buffer instead of feeding raw blocks straight to the model:
    ImpulseSuppressorNet (Task 2) was trained on 2-second crops. Feeding it a
    bare 20ms block gives it almost no temporal context to distinguish a
    transient spike from normal speech onset, and its GRU has no persisted
    hidden state between isolated calls. Instead, we keep a rolling window of
    the last `context_ms` of already-captured audio, append each new block,
    run the model over the whole window, and only emit the newest slice of
    the output. This is still fully causal (only past audio, zero added
    look-ahead delay) -- it just gives the model recent history to reason
    over, which is exactly what it saw during training.

Why two threads (not just one audio callback that also runs the model):
    sounddevice's audio callbacks run on a real-time-priority thread and MUST
    return almost instantly, or you get audible clicks/underruns. A neural
    net forward pass (even a small one) can take a few ms to tens of ms on
    a CPU -- too slow/variable to trust inside the callback. So callbacks
    only do queue.put()/queue.get() (near-instant), and all compute happens
    in a normal Python thread that can take its time.

LATENCY BUDGET (approximate, tune with --block_ms):
    Algorithmic latency  = block_ms  (you must wait for a full block before
                            it can be processed and played back)
    + compute latency     = however long one forward pass takes on your CPU
                            (printed live in the console so you can tune it)
    Total should stay comfortably under ~150-200ms for a real-time feel.
    If you see "UNDERRUN" warnings, increase --block_ms or use a smaller
    --context_ms, or reduce model size.

Requirements:
    pip install sounddevice torch numpy

Usage:
    # First, list audio devices to find your mic / headphone indices:
    python task4_realtime.py --list_devices

    # Run live enhancement:
    python task4_realtime.py --mode lightweight --checkpoint checkpoints/lightweight_best.pt \
        --input_device 1 --output_device 3

    # Bypass mode (mic straight to headphones, no model) -- useful to sanity-check
    # your audio routing/latency before trusting the model is doing anything:
    python task4_realtime.py --mode passthrough --input_device 1 --output_device 3

    # Record the live session (both raw input and enhanced output) to disk for review:
    python task4_realtime.py --mode lightweight --checkpoint checkpoints/lightweight_best.pt \
        --record_dir outputs/live_session
"""

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice is required. Install with: pip install sounddevice")
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Device listing helper
# --------------------------------------------------------------------------- #

def list_devices():
    print(sd.query_devices())
    print("\nDefault input device :", sd.default.device[0])
    print("Default output device:", sd.default.device[1])
    print("\nTip: pass the index number shown above to --input_device / --output_device")


# --------------------------------------------------------------------------- #
# Model loading (reuses Task 2's architecture / Task 3's loader logic)
# --------------------------------------------------------------------------- #

def load_enhancer(mode: str, checkpoint: str, device: str):
    """
    Returns a callable enhance_block(buffer: np.ndarray) -> np.ndarray
    that takes the FULL rolling-context buffer and returns the enhanced
    version of that same buffer (caller extracts the newest slice).
    """
    if mode == "passthrough":
        return lambda buf: buf.copy()

    if mode == "lightweight":
        from task2_finetune import ImpulseSuppressorNet

        model = ImpulseSuppressorNet().to(device)
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location=device)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state)
            print(f"[model] Loaded checkpoint: {checkpoint}")
        else:
            print("[model] WARNING: no checkpoint given -- using randomly initialized "
                  "weights (output will just be noise, use only to test plumbing).")
        model.eval()

        @torch.no_grad()
        def enhance_block(buf: np.ndarray) -> np.ndarray:
            x = torch.from_numpy(buf).unsqueeze(0).to(device)
            out = model(x)
            out = out.squeeze(0).cpu().numpy()
            if len(out) != len(buf):  # safety, in case of conv length rounding
                out = np.resize(out, len(buf))
            return out

        return enhance_block

    if mode == "full_dfn":
        try:
            from df.enhance import init_df, enhance
        except ImportError as e:
            raise RuntimeError(
                "deepfilternet not installed. `pip install deepfilternet` or use "
                "--mode lightweight / passthrough."
            ) from e

        model, df_state, _ = init_df(model_base_dir=checkpoint) if checkpoint else init_df()
        model.eval()

        def enhance_block(buf: np.ndarray) -> np.ndarray:
            x = torch.from_numpy(buf).unsqueeze(0)
            out = enhance(model, df_state, x)
            out = out.squeeze(0).cpu().numpy()
            if len(out) != len(buf):
                out = np.resize(out, len(buf))
            return out

        return enhance_block

    raise ValueError(f"Unknown mode: {mode}")


# --------------------------------------------------------------------------- #
# Simple console VU meter (visual feedback, no extra deps)
# --------------------------------------------------------------------------- #

def db_of(x: np.ndarray) -> float:
    r = np.sqrt(np.mean(x ** 2) + 1e-12)
    return 20 * np.log10(r + 1e-12)


def vu_bar(db: float, floor=-60.0, ceil=0.0, width=30) -> str:
    frac = np.clip((db - floor) / (ceil - floor), 0.0, 1.0)
    filled = int(frac * width)
    return "#" * filled + "-" * (width - filled)


# --------------------------------------------------------------------------- #
# Real-time engine
# --------------------------------------------------------------------------- #

class RealtimeEnhancer:
    def __init__(self, args):
        self.args = args
        self.sr = args.sr
        self.block_size = int(args.block_ms * self.sr / 1000)
        self.context_size = int(args.context_ms * self.sr / 1000)

        self.device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
        print(f"[device] Running model on: {self.device}")
        self.enhance_block = load_enhancer(args.mode, args.checkpoint, self.device)

        # Rolling context buffer: [ ...past audio (context_size)... | newest block ]
        self.buffer = np.zeros(self.context_size + self.block_size, dtype=np.float32)

        self.input_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=50)
        self.output_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=50)

        self.running = threading.Event()
        self.running.set()

        self.spike_threshold_db = args.spike_alert_db
        self.last_compute_ms = 0.0

        # Optional recording buffers
        self.record = args.record_dir is not None
        self.rec_input = [] if self.record else None
        self.rec_output = [] if self.record else None

    # ---- sounddevice callbacks (must be fast, no compute here) ----

    def input_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[input status] {status}")
        block = indata[:, 0].copy()  # mono
        try:
            self.input_q.put_nowait(block)
        except queue.Full:
            print("[warn] input queue full -- dropping a block (processing thread too slow)")

    def output_callback(self, outdata, frames, time_info, status):
        if status:
            print(f"[output status] {status}")
        try:
            block = self.output_q.get_nowait()
        except queue.Empty:
            block = np.zeros(frames, dtype=np.float32)
            print("[warn] UNDERRUN -- output queue empty, playing silence "
                  "(try a larger --block_ms)")
        outdata[:, 0] = block

    # ---- processing thread (all compute happens here) ----

    def processing_loop(self):
        while self.running.is_set():
            try:
                block = self.input_q.get(timeout=0.5)
            except queue.Empty:
                continue

            in_db = db_of(block)
            spike_detected = in_db > self.spike_threshold_db

            # Slide the rolling buffer and append the new block
            self.buffer = np.roll(self.buffer, -len(block))
            self.buffer[-len(block):] = block

            t0 = time.perf_counter()
            enhanced_full = self.enhance_block(self.buffer)
            self.last_compute_ms = (time.perf_counter() - t0) * 1000

            enhanced_block = enhanced_full[-len(block):]
            # Safety clip to avoid blasting the user's headphones on model glitches
            enhanced_block = np.clip(enhanced_block, -1.0, 1.0).astype(np.float32)

            out_db = db_of(enhanced_block)

            try:
                self.output_q.put_nowait(enhanced_block)
            except queue.Full:
                print("[warn] output queue full -- dropping a block")

            if self.record:
                self.rec_input.append(block.copy())
                self.rec_output.append(enhanced_block.copy())

            self._print_status(in_db, out_db, spike_detected)

    def _print_status(self, in_db, out_db, spike_detected):
        flag = " <-- SPIKE DETECTED" if spike_detected else ""
        sys.stdout.write(
            f"\rIN  [{vu_bar(in_db)}] {in_db:6.1f} dB   "
            f"OUT [{vu_bar(out_db)}] {out_db:6.1f} dB   "
            f"compute={self.last_compute_ms:5.1f}ms{flag}   "
        )
        sys.stdout.flush()

    # ---- lifecycle ----

    def run(self):
        proc_thread = threading.Thread(target=self.processing_loop, daemon=True)
        proc_thread.start()

        # Pre-fill output queue with a bit of silence so playback doesn't
        # underrun the instant the stream starts, before the first block
        # has been processed.
        for _ in range(3):
            self.output_q.put(np.zeros(self.block_size, dtype=np.float32))

        print(f"\n[start] block={self.args.block_ms}ms  context={self.args.context_ms}ms  "
              f"mode={self.args.mode}  device={self.device}")
        print("[start] Speak / make noise near the mic. Press Ctrl+C to stop.\n")

        with sd.InputStream(
            samplerate=self.sr, blocksize=self.block_size, channels=1,
            dtype="float32", device=self.args.input_device, callback=self.input_callback,
        ), sd.OutputStream(
            samplerate=self.sr, blocksize=self.block_size, channels=1,
            dtype="float32", device=self.args.output_device, callback=self.output_callback,
        ):
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n[stop] Ctrl+C received, shutting down...")

        self.running.clear()
        proc_thread.join(timeout=2.0)

        if self.record:
            self._save_recording()

    def _save_recording(self):
        import soundfile as sf
        out_dir = Path(self.args.record_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        raw = np.concatenate(self.rec_input) if self.rec_input else np.zeros(0, dtype=np.float32)
        enh = np.concatenate(self.rec_output) if self.rec_output else np.zeros(0, dtype=np.float32)

        sf.write(out_dir / "session_input_raw.wav", raw, self.sr)
        sf.write(out_dir / "session_output_enhanced.wav", enh, self.sr)
        print(f"[saved] Session recording -> {out_dir}/session_input_raw.wav, "
              f"{out_dir}/session_output_enhanced.wav")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Task 4: Real-time impulsive noise suppression")
    parser.add_argument("--list_devices", action="store_true", help="List audio devices and exit")

    parser.add_argument("--mode", type=str, choices=["passthrough", "lightweight", "full_dfn"],
                         default="lightweight")
    parser.add_argument("--checkpoint", type=str, default=None)

    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--block_ms", type=float, default=32.0,
                         help="Audio block size in ms. Larger = more stable, more latency.")
    parser.add_argument("--context_ms", type=float, default=480.0,
                         help="Rolling history window fed to the model alongside each new block.")

    parser.add_argument("--input_device", type=int, default=None, help="Mic device index (see --list_devices)")
    parser.add_argument("--output_device", type=int, default=None, help="Headphone device index")

    parser.add_argument("--spike_alert_db", type=float, default=-10.0,
                         help="Console flags input blocks louder than this as a likely impulse")

    parser.add_argument("--record_dir", type=str, default=None,
                         help="If set, saves the raw and enhanced session audio here on exit")
    parser.add_argument("--force_cpu", action="store_true")

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    engine = RealtimeEnhancer(args)
    engine.run()


if __name__ == "__main__":
    main()