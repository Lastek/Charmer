"""
stream_overlay.py
-----------------
Reads your webcam feed, applies a rotating UAP overlay, and outputs it
as a virtual camera that OBS (or any app) can capture.

The UAP rotates every N seconds from a pregenerated bank in ./uap_bank/.

Usage:
    python stream_overlay.py \
        --uap-dir ./uap_bank \
        --rotate-every 8 \
        --alpha 0.85 \
        --camera 0

Requirements: See requirements.txt

How it works:
    1. Reads webcam frames via OpenCV
    2. Resizes UAP to match current frame size (handles non-fixed resolutions)
    3. Blends UAP onto frame with configurable alpha
    4. Writes blended frame to a virtual camera device
    5. OBS adds that virtual camera as a Video Capture Device source

OBS Setup:
    - Add a "Video Capture Device" source
    - Select "OBS-Camera" or "v4l2loopback" (Linux) / "OBS Virtual Camera" (Win/Mac)
    - This script feeds into that device
"""

import argparse
import time
import sys
import signal
import threading
from pathlib import Path

import cv2
import numpy as np

try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat
    HAS_VIRTUALCAM = True
except ImportError:
    HAS_VIRTUALCAM = False
    print("[!] pyvirtualcam not installed — preview-only mode (no virtual camera output).")
    print("    Install with: pip install pyvirtualcam")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Apply rotating UAP overlay to webcam feed")
    p.add_argument("--uap-dir",       type=str, default="./uap_bank",
                   help="Directory of .npy UAP files (default: ./uap_bank)")
    p.add_argument("--rotate-every",  type=float, default=8.0,
                   help="Seconds between UAP rotations (default: 8)")
    p.add_argument("--alpha",         type=float, default=0.85,
                   help="Frame blend factor 0.0=all UAP, 1.0=all frame (default: 0.85)")
    p.add_argument("--camera",        type=int, default=0,
                   help="Webcam device index (default: 0)")
    p.add_argument("--fps",           type=int, default=30,
                   help="Output FPS (default: 30)")
    p.add_argument("--width",         type=int, default=None,
                   help="Force capture width (default: camera native)")
    p.add_argument("--height",        type=int, default=None,
                   help="Force capture height (default: camera native)")
    p.add_argument("--preview",       action="store_true",
                   help="Show local OpenCV preview window")
    p.add_argument("--no-virtualcam", action="store_true",
                   help="Disable virtual camera output (preview only)")
    p.add_argument("--jitter",        action="store_true", default=True,
                   help="Add per-frame micro-jitter to UAP (default: True)")
    p.add_argument("--jitter-strength", type=float, default=2.0,
                   help="Pixel std of per-frame jitter (default: 2.0)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# UAP bank
# ---------------------------------------------------------------------------

class UAPBank:
    """
    Loads a directory of .npy UAP files and serves them in rotation.
    Handles resizing UAPs to match actual frame dimensions.
    """

    def __init__(self, uap_dir: str):
        self.uap_dir = Path(uap_dir)
        self.uaps_raw = []      # original numpy arrays (H, W, 3) float32
        self.uaps_resized = {}  # cache: (H,W) → resized array
        self._load()

    def _load(self):
        paths = sorted(self.uap_dir.glob("*.npy"))
        if not paths:
            raise FileNotFoundError(
                f"No .npy files found in {self.uap_dir}\n"
                "Run generate_uaps.py first."
            )
        for p in paths:
            arr = np.load(p).astype(np.float32)  # shape (H, W, 3)
            self.uaps_raw.append(arr)
        print(f"[+] Loaded {len(self.uaps_raw)} UAPs from {self.uap_dir}")

    def get(self, index: int, frame_h: int, frame_w: int) -> np.ndarray:
        """Return UAP at index, resized to (frame_h, frame_w, 3)."""
        idx = index % len(self.uaps_raw)
        cache_key = (idx, frame_h, frame_w)

        if cache_key not in self.uaps_resized:
            raw = self.uaps_raw[idx]
            resized = cv2.resize(raw, (frame_w, frame_h),
                                 interpolation=cv2.INTER_LINEAR)
            self.uaps_resized[cache_key] = resized

        return self.uaps_resized[cache_key]

    def __len__(self):
        return len(self.uaps_raw)


# ---------------------------------------------------------------------------
# Frame blending
# ---------------------------------------------------------------------------

def apply_uap(
    frame: np.ndarray,        # uint8 (H, W, 3) BGR
    uap: np.ndarray,          # float32 (H, W, 3) pixel-space delta
    alpha: float,             # 1.0 = invisible, 0.0 = fully replaced
    jitter: bool = True,
    jitter_strength: float = 2.0,
) -> np.ndarray:
    """
    Blend UAP perturbation onto frame.

    The UAP is an additive delta: output = clamp(frame + uap_scaled, 0, 255)
    alpha controls the UAP contribution strength:
        uap_contribution = uap * (1.0 - alpha)

    Per-frame jitter adds a tiny random shift each frame to break
    temporal averaging attacks (someone recording and averaging frames
    to cancel the static UAP).
    """
    frame_f = frame.astype(np.float32)

    # OpenCV is BGR; UAP is stored as RGB — convert
    uap_bgr = uap[:, :, ::-1]

    if jitter:
        noise = np.random.normal(0, jitter_strength, uap_bgr.shape).astype(np.float32)
        uap_bgr = uap_bgr + noise

    # Scale by (1 - alpha): higher alpha = weaker perturbation
    uap_contribution = uap_bgr * (1.0 - alpha)

    blended = frame_f + uap_contribution
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended


# ---------------------------------------------------------------------------
# Stats overlay (optional HUD)
# ---------------------------------------------------------------------------

def draw_hud(frame: np.ndarray, uap_index: int, uap_count: int,
             rotate_in: float, fps: float) -> np.ndarray:
    """Draw a small semi-transparent HUD for monitoring. Toggle off for production."""
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Background box
    cv2.rectangle(overlay, (0, 0), (280, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(frame, f"UAP {uap_index+1}/{uap_count}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1)
    cv2.putText(frame, f"Next rotate: {rotate_in:.1f}s", (8, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (8, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return frame


# ---------------------------------------------------------------------------
# Main streaming loop
# ---------------------------------------------------------------------------

def run(args):
    # Load UAP bank
    bank = UAPBank(args.uap_dir)

    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[!] Cannot open camera {args.camera}")
        sys.exit(1)

    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[+] Camera: {cam_w}x{cam_h} @ {args.fps}fps")

    use_virtualcam = HAS_VIRTUALCAM and not args.no_virtualcam

    uap_index = 0
    last_rotate = time.time()
    frame_count = 0
    fps_timer = time.time()
    current_fps = 0.0

    # Graceful shutdown
    running = True
    def _stop(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    def stream_loop(vcam=None):
        nonlocal uap_index, last_rotate, frame_count, current_fps, running

        while running:
            ret, frame = cap.read()
            if not ret:
                print("[!] Failed to read frame — retrying...")
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]
            now = time.time()

            # Rotate UAP
            elapsed = now - last_rotate
            if elapsed >= args.rotate_every:
                uap_index = (uap_index + 1) % len(bank)
                last_rotate = now
                elapsed = 0.0

            # Get UAP for current frame size
            uap = bank.get(uap_index, h, w)

            # Apply perturbation
            perturbed = apply_uap(
                frame, uap,
                alpha=args.alpha,
                jitter=args.jitter,
                jitter_strength=args.jitter_strength,
            )

            # FPS counter
            frame_count += 1
            if frame_count % 30 == 0:
                current_fps = 30.0 / (now - fps_timer)
                fps_timer = now

            # Preview window
            if args.preview:
                preview = draw_hud(perturbed.copy(), uap_index, len(bank),
                                   args.rotate_every - elapsed, current_fps)
                cv2.imshow("UAP Stream Preview (press Q to quit)", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    running = False
                    break

            # Virtual camera output
            if vcam is not None:
                # pyvirtualcam expects RGB
                rgb = cv2.cvtColor(perturbed, cv2.COLOR_BGR2RGB)
                vcam.send(rgb)
                vcam.sleep_until_next_frame()

    print("[+] Starting stream loop. Ctrl+C to stop.")
    print(f"    UAP rotates every {args.rotate_every}s | alpha={args.alpha} | "
          f"jitter={'on' if args.jitter else 'off'}")

    if use_virtualcam:
        with pyvirtualcam.Camera(
            width=cam_w, height=cam_h, fps=args.fps,
            fmt=PixelFormat.RGB,
            print_fps=False,
        ) as vcam:
            print(f"[+] Virtual camera active: {vcam.device}")
            print("     → Add this as a 'Video Capture Device' in OBS")
            stream_loop(vcam)
    else:
        if not args.preview:
            print("[!] No virtual camera and no --preview. Adding --preview automatically.")
            args.preview = True
        stream_loop(vcam=None)

    cap.release()
    cv2.destroyAllWindows()
    print("\n[✓] Stream stopped cleanly.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run(args)
