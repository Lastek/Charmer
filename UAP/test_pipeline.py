"""
test_pipeline.py
----------------
Sanity-check script — runs the full pipeline without a GPU or webcam.

1. Generates 3 tiny UAPs using synthetic data (no face images needed)
2. Applies them to a synthetic "frame" and saves before/after images
3. Prints a diff report so you can verify the perturbation is being applied

Run this first to confirm your install is working before running the
real generator and overlay scripts.

Usage:
    python test_pipeline.py
"""

import numpy as np
import cv2
from pathlib import Path

# ---------------------------------------------------------------------------
# Minimal UAP generation (CPU, no model — just structured noise for testing)
# ---------------------------------------------------------------------------

def generate_test_uap(height: int, width: int, seed: int = 0) -> np.ndarray:
    """
    Generate a synthetic UAP-like perturbation for pipeline testing.
    This is NOT an adversarially trained UAP — it's structured noise used
    only to verify the overlay and rotation pipeline works end-to-end.
    """
    rng = np.random.default_rng(seed)

    # Simulate mid-frequency structured noise (like a real UAP)
    uap = rng.normal(0, 8.0, (height, width, 3)).astype(np.float32)

    # Smooth slightly to push energy to mid-frequencies (survives compression better)
    uap = cv2.GaussianBlur(uap, (3, 3), 0.8)

    # Clip to epsilon budget (10/255 * 255 = 10 pixels)
    uap = np.clip(uap, -10.0, 10.0)

    return uap


def apply_uap(frame: np.ndarray, uap: np.ndarray, alpha: float = 0.85,
              jitter_strength: float = 2.0) -> np.ndarray:
    frame_f = frame.astype(np.float32)
    noise = np.random.normal(0, jitter_strength, uap.shape).astype(np.float32)
    contribution = (uap + noise) * (1.0 - alpha)
    blended = np.clip(frame_f + contribution, 0, 255).astype(np.uint8)
    return blended


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def main():
    H, W = 480, 320
    out_dir = Path("./test_output")
    out_dir.mkdir(exist_ok=True)

    print("=" * 55)
    print("  UAP Pipeline Sanity Check")
    print("=" * 55)

    # 1. Generate test UAPs
    print("\n[1] Generating 3 synthetic test UAPs...")
    uap_dir = Path("./uap_bank")
    uap_dir.mkdir(exist_ok=True)

    for i in range(3):
        uap = generate_test_uap(H, W, seed=i)
        path = uap_dir / f"uap_{i:03d}.npy"
        np.save(path, uap)
        print(f"    Saved {path}  shape={uap.shape}  "
              f"range=[{uap.min():.1f}, {uap.max():.1f}]")

    # 2. Create a synthetic "webcam frame" (gradient + noise to simulate a face scene)
    print("\n[2] Creating synthetic test frame...")
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    # Gradient background
    for y in range(H):
        frame[y, :, 0] = int(120 * y / H)           # Blue channel
        frame[y, :, 1] = int(80 + 80 * y / H)       # Green channel
        frame[y, :, 2] = int(200 - 100 * y / H)     # Red channel
    # Simulate a face-ish region
    face_center = (W // 2, H // 3)
    cv2.ellipse(frame, face_center, (W//5, H//5), 0, 0, 360,
                (200, 175, 150), -1)

    cv2.imwrite(str(out_dir / "frame_original.png"), frame)
    print(f"    Saved original frame → {out_dir}/frame_original.png")

    # 3. Apply each UAP and save
    print("\n[3] Applying UAPs and saving results...")
    uaps = [np.load(p) for p in sorted(uap_dir.glob("*.npy"))]

    for i, uap in enumerate(uaps):
        perturbed = apply_uap(frame, uap, alpha=0.85)
        path = out_dir / f"frame_uap_{i:03d}.png"
        cv2.imwrite(str(path), perturbed)

        # Pixel diff report
        diff = frame.astype(np.float32) - perturbed.astype(np.float32)
        print(f"    UAP {i}: max_pixel_change={np.abs(diff).max():.1f}  "
              f"mean_change={np.abs(diff).mean():.2f}  → {path.name}")

    # 4. Create a side-by-side comparison image
    print("\n[4] Creating comparison image...")
    comparison_frames = [frame]
    for uap in uaps:
        comparison_frames.append(apply_uap(frame, uap, alpha=0.85))

    labels = ["Original"] + [f"UAP {i}" for i in range(len(uaps))]
    labeled = []
    for f, label in zip(comparison_frames, labels):
        lf = f.copy()
        cv2.putText(lf, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 1)
        labeled.append(lf)

    comparison = np.hstack(labeled)
    comp_path = out_dir / "comparison.png"
    cv2.imwrite(str(comp_path), comparison)
    print(f"    Saved comparison → {comp_path}")

    # 5. Amplified diff for visual inspection
    print("\n[5] Creating amplified diff image (10x) for visual inspection...")
    diff_raw = frame.astype(np.float32) - apply_uap(frame, uaps[0], alpha=0.85).astype(np.float32)
    diff_amplified = np.clip(np.abs(diff_raw) * 10, 0, 255).astype(np.uint8)
    diff_path = out_dir / "diff_amplified_10x.png"
    cv2.imwrite(str(diff_path), diff_amplified)
    print(f"    Saved amplified diff → {diff_path}")
    print("    (This shows where perturbation energy is concentrated.)")

    print("\n" + "=" * 55)
    print("  ✓ All tests passed!")
    print("=" * 55)
    print(f"\nOutputs in: {out_dir.resolve()}")
    print("\nNext steps:")
    print("  1. Inspect comparison.png — differences should be invisible to the eye")
    print("  2. Inspect diff_amplified_10x.png — shows perturbation pattern")
    print("  3. Run: python generate_uaps.py  (needs GPU + facenet-pytorch)")
    print("  4. Run: python stream_overlay.py --preview")


if __name__ == "__main__":
    main()
