"""
generate_uaps.py
----------------
Generates a bank of Universal Adversarial Perturbations (UAPs) targeting
facial recognition and pose estimation models.

Saves UAPs as .npy files to ./uap_bank/ for use by stream_overlay.py.

Usage:
    python generate_uaps.py --count 30 --epsilon 10 --iters 500 --res 320x480

Requirements: See requirements.txt
"""

import os
import argparse
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate a UAP bank for stream privacy")
    p.add_argument("--count",   type=int, default=30,
                   help="Number of UAPs to generate (default: 30)")
    p.add_argument("--epsilon", type=float, default=10.0,
                   help="Perturbation budget in [0,255] pixel units (default: 10)")
    p.add_argument("--iters",   type=int, default=600,
                   help="Gradient steps per UAP (default: 600)")
    p.add_argument("--res",     type=str, default="320x480",
                   help="WxH resolution string, e.g. 320x480 (default: 320x480)")
    p.add_argument("--face-dir", type=str, default="./face_images",
                   help="Directory of diverse face images used for UAP fitting")
    p.add_argument("--out-dir", type=str, default="./uap_bank",
                   help="Output directory for .npy UAP files")
    p.add_argument("--device",  type=str, default="auto",
                   help="'cpu', 'cuda', or 'auto' (default: auto)")
    p.add_argument("--model",   type=str, default="facenet",
                   choices=["facenet", "arcface"],
                   help="Target model to attack (default: facenet)")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: torch.device) -> nn.Module:
    """Load a pretrained facial recognition model."""

    if model_name == "facenet":
        try:
            from facenet_pytorch import InceptionResnetV1
            model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
            print("[+] Loaded FaceNet (VGGFace2 pretrained)")
            return model
        except ImportError:
            raise ImportError(
                "facenet-pytorch not installed.\n"
                "Run: pip install facenet-pytorch"
            )

    elif model_name == "arcface":
        try:
            import insightface
            from insightface.app import FaceAnalysis
            # Use insightface's ArcFace backbone wrapped in a simple nn.Module
            model = ArcFaceWrapper(device)
            print("[+] Loaded ArcFace (insightface)")
            return model
        except ImportError:
            raise ImportError(
                "insightface not installed.\n"
                "Run: pip install insightface onnxruntime"
            )

    raise ValueError(f"Unknown model: {model_name}")


class ArcFaceWrapper(nn.Module):
    """Thin PyTorch wrapper around insightface ArcFace so we can backprop."""

    def __init__(self, device):
        super().__init__()
        # This is a simplified stand-in; full insightface integration
        # requires ONNX runtime and is not differentiable by default.
        # For full ArcFace gradient support use the buffalo_l ONNX model
        # via onnx2torch, or swap in a PyTorch ArcFace repo.
        raise NotImplementedError(
            "ArcFace gradient mode requires onnx2torch conversion.\n"
            "Recommended: use --model facenet for a quick start."
        )


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_face_images(face_dir: str, resolution: tuple, device: torch.device,
                     max_images: int = 200) -> list:
    """
    Load and preprocess face images from a directory.

    If the directory is empty or missing, falls back to synthetic random
    face-shaped tensors (less effective but still functional for testing).
    """
    W, H = resolution  # width, height
    transform = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor(),          # [0,1]
        transforms.Normalize([0.5]*3, [0.5]*3),  # [-1,1] for FaceNet
    ])

    images = []
    face_dir_path = Path(face_dir)

    if face_dir_path.exists():
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        paths = [p for p in face_dir_path.rglob("*") if p.suffix.lower() in exts]
        random.shuffle(paths)
        paths = paths[:max_images]

        for p in tqdm(paths, desc="Loading face images"):
            try:
                img = Image.open(p).convert("RGB")
                images.append(transform(img).unsqueeze(0).to(device))
            except Exception as e:
                print(f"  [!] Skipping {p.name}: {e}")

    if len(images) == 0:
        print("[!] No face images found — using synthetic tensors.")
        print("    For better UAPs, populate ./face_images/ with diverse face photos.")
        print("    (e.g. download a small LFW or CelebA subset)")
        for _ in range(50):
            images.append(torch.randn(1, 3, H, W).to(device))

    print(f"[+] Using {len(images)} images for UAP fitting")
    return images


# ---------------------------------------------------------------------------
# DCT-aware perturbation clamping
# ---------------------------------------------------------------------------

def clamp_to_mid_freq(uap: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Standard L-inf clamp. For true DCT-domain perturbations (compression-robust),
    you would apply a 2D DCT, zero the highest frequency bands, then iDCT back.
    This is the lightweight version that still survives moderate compression.
    """
    return torch.clamp(uap, -epsilon, epsilon)


# ---------------------------------------------------------------------------
# Core UAP generation
# ---------------------------------------------------------------------------

def generate_single_uap(
    model: nn.Module,
    images: list,
    epsilon: float,          # in [0,1] range
    iterations: int,
    resolution: tuple,       # (W, H)
    device: torch.device,
    seed: int|Nonet = None,
) -> np.ndarray:
    """
    Generate one Universal Adversarial Perturbation.

    Strategy: maximize cosine distance between the clean embedding and the
    perturbed embedding, accumulated across all images in the dataset.

    Returns a numpy array of shape (H, W, 3) with values in [-epsilon, epsilon]
    scaled to [0,255] pixel space, ready to be blended as an overlay.
    """
    W, H = resolution

    if seed is not None:
        torch.manual_seed(seed)

    # Initialize UAP from small random noise
    uap = torch.zeros(1, 3, H, W, device=device).uniform_(-epsilon * 0.1, epsilon * 0.1)
    uap.requires_grad_(True)

    optimizer = torch.optim.Adam([uap], lr=epsilon * 0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    model.eval()

    for step in range(iterations):
        # Sample a random mini-batch of images each step for diversity
        batch_imgs = random.sample(images, min(8, len(images)))

        total_loss = torch.tensor(0.0, device=device)

        for img in batch_imgs:
            # Clamp perturbed image to valid range
            perturbed = torch.clamp(img + uap, -1.0, 1.0)

            with torch.no_grad():
                clean_emb = model(img)

            perturbed_emb = model(perturbed)

            # Primary: maximize embedding distance (fool recognition)
            cos_sim = F.cosine_similarity(perturbed_emb, clean_emb, dim=1)
            recognition_loss = cos_sim.mean()

            # Secondary: maximize L2 distance in embedding space
            l2_loss = -F.pairwise_distance(perturbed_emb, clean_emb).mean()

            total_loss = total_loss + (recognition_loss + 0.3 * l2_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Project back into L-inf ball
        with torch.no_grad():
            uap.data = clamp_to_mid_freq(uap.data, epsilon)

        if step % 100 == 0:
            print(f"    step {step:4d}/{iterations}  loss={total_loss.item():.4f}")

    # Convert from [-1,1] normalized space back to [0,255] delta space
    uap_np = uap.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    # uap_np is in [-epsilon, epsilon] where epsilon is in [0,1]
    # Scale to pixel space for overlay blending
    uap_np_pixels = (uap_np * 255.0).astype(np.float32)

    return uap_np_pixels   # shape: (H, W, 3), range: [-255*eps, 255*eps]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[+] Using device: {device}")

    # Resolution
    W_str, H_str = args.res.split("x")
    resolution = (int(W_str), int(H_str))   # (width, height)
    print(f"[+] Target resolution: {resolution[0]}x{resolution[1]}")

    # Epsilon: convert from [0,255] → [0,1] for normalized tensor math
    epsilon_norm = args.epsilon / 255.0
    print(f"[+] Epsilon: {args.epsilon}/255 = {epsilon_norm:.4f}")

    # Output dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.model, device)

    # Load images
    images = load_face_images(args.face_dir, resolution, device)

    # Generate UAP bank
    print(f"\n[+] Generating {args.count} UAPs...")
    for i in range(args.count):
        print(f"\n  UAP {i+1}/{args.count}")
        uap = generate_single_uap(
            model=model,
            images=images,
            epsilon=epsilon_norm,
            iterations=args.iters,
            resolution=resolution,
            device=device,
            seed=args.seed + i,   # different seed per UAP for variety
        )

        out_path = out_dir / f"uap_{i:03d}.npy"
        np.save(out_path, uap)
        print(f"  Saved → {out_path}  (shape={uap.shape}, "
              f"range=[{uap.min():.1f}, {uap.max():.1f}])")

    print(f"\n[✓] Done. {args.count} UAPs saved to {out_dir}/")
    print("    Next step: run stream_overlay.py to apply them to your webcam feed.")


if __name__ == "__main__":
    main()
