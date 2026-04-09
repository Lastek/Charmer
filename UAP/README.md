# Stream UAP Privacy Shield

Applies rotating Universal Adversarial Perturbations (UAPs) to your webcam feed
in real-time via a virtual camera, making facial recognition and pose estimation
significantly harder on your stream content.

---

## How It Works

```
Webcam → [stream_overlay.py: UAP blend] → Virtual Camera → OBS → Twitch
```

- A UAP is an imperceptible noise pattern (≤10 pixel values) computed to
  maximally confuse facial recognition model embeddings
- A bank of 20-50 UAPs is pregenerated, then rotated every N seconds
- Per-frame micro-jitter breaks temporal averaging attacks
- The virtual camera output feeds directly into OBS

---

## Files

| File | Purpose |
|---|---|
| `generate_uaps.py` | Generate a bank of UAPs targeting FaceNet/ArcFace |
| `stream_overlay.py` | Real-time overlay + virtual camera output |
| `test_pipeline.py` | Sanity check — no GPU or webcam needed |
| `requirements.txt` | Python dependencies |

---

## Quick Start

### Step 0 — Test your install (no GPU needed)
```bash
pip install -r requirements.txt
python test_pipeline.py
# Check ./test_output/comparison.png — perturbation should be invisible
```

### Step 1 — Get face images for UAP fitting

You need ~50-200 diverse face photos. Options:

**Option A (easiest):** Download a small LFW subset
```bash
# LFW (Labeled Faces in the Wild) - publicly available
wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
tar -xzf lfw.tgz
# Copy ~200 images to ./face_images/
find lfw/ -name "*.jpg" | head -200 | xargs -I{} cp {} ./face_images/
```

**Option B:** Use any 50+ diverse face photos you have access to.
Put them all flat in `./face_images/`.

> **Note:** These images are used *only* to compute gradients for UAP fitting.
> They are never uploaded anywhere.

### Step 2 — Generate UAP bank
```bash
# GPU (recommended, ~10-20 min for 30 UAPs)
python generate_uaps.py \
    --count 30 \
    --epsilon 10 \
    --iters 600 \
    --res 320x480 \
    --face-dir ./face_images

# CPU fallback (~1-3 hours)
python generate_uaps.py --count 10 --iters 200 --device cpu
```

UAPs are saved to `./uap_bank/*.npy`.

### Step 3 — Start the overlay stream
```bash
python stream_overlay.py \
    --uap-dir ./uap_bank \
    --rotate-every 8 \
    --alpha 0.85 \
    --camera 0 \
    --preview
```

### Step 4 — Connect OBS

1. Open OBS → Add Source → **Video Capture Device**
2. Select **OBS Virtual Camera** (Windows/macOS) or **v4l2loopback** (Linux)
3. That's it — OBS now receives your perturbed feed

---

## Key Parameters

### generate_uaps.py

| Arg | Default | Notes |
|---|---|---|
| `--count` | 30 | More UAPs = more rotation variety |
| `--epsilon` | 10 | Pixel budget [0-255]. 8-12 is the sweet spot |
| `--iters` | 600 | More = stronger UAP, slower generation |
| `--res` | 320x480 | Match your webcam resolution |
| `--model` | facenet | Target model. `arcface` requires extra setup |

### stream_overlay.py

| Arg | Default | Notes |
|---|---|---|
| `--rotate-every` | 8 | Seconds between UAP switches |
| `--alpha` | 0.85 | 0.8-0.9 is invisible, <0.7 becomes faintly visible |
| `--jitter` | True | Per-frame micro-noise, breaks temporal attacks |
| `--jitter-strength` | 2.0 | Std of per-frame noise in pixels |
| `--preview` | False | Show local OpenCV window |

---

## Platform-Specific Notes

### Windows
- Virtual camera: install **OBS** and enable OBS Virtual Camera (comes bundled)
- `pyvirtualcam` uses OBS's built-in virtual camera backend

### macOS
- Virtual camera: install **OBS** (includes virtual camera support from v27+)
- May need to grant camera permissions in System Preferences

### Linux
- Virtual camera: install `v4l2loopback`
  ```bash
  sudo apt install v4l2loopback-dkms
  sudo modprobe v4l2loopback
  ```
- `pyvirtualcam` detects `/dev/video*` loopback devices automatically

---

## Tuning for Compression Survival

Twitch's H.264 encoding destroys very high-frequency perturbations.
Mitigations already built in:

1. **Mid-frequency smoothing** in `generate_test_uap()` — energy is in bands
   that survive DCT quantization
2. **Boost epsilon** — use `--epsilon 12` to compensate for ~2-3 point
   compression attenuation
3. **Jitter** — since static UAPs are most vulnerable to compression averaging,
   the per-frame jitter keeps the perturbation from becoming a predictable pattern

---

## Extending This

**Multi-model targeting:** Run `generate_uaps.py` twice with `--model facenet`
and `--model arcface`, then mix UAPs from both banks. This gives you perturbations
that confuse a broader range of deployed models.

**Pose estimation disruption:** Tools like OpenPose and MediaPipe use separate
heatmap regression networks. To target these, swap the loss function in
`generate_uaps.py` to maximize keypoint prediction error. A starting point:
use a pretrained MoveNet/Lightning model from TensorFlow Hub as the target.

**Physical patch combo:** Print an adversarial pattern (search "adversarial
t-shirt" or AdvPatch) and wear it on stream. Combined with this software UAP,
you have defense in depth across both software and physical vectors.

---

## Limitations

- UAPs degrade over time as models are retrained — regenerate your bank every
  few months or if you notice degraded protection
- Very high compression (low Twitch bitrate settings) will weaken the perturbation;
  use 4500+ kbps for best results
- This is a probabilistic defense, not a guarantee — it raises the cost and
  effort of AI training on your content significantly, but is not absolute

---

## References

- Moosavi-Dezfooli et al., *Universal adversarial perturbations* (CVPR 2017)
- Fawkes: https://sandlab.cs.uchicago.edu/fawkes/
- LowKey: https://lowkey.umiacs.umd.edu/
- Wu et al., *Making an Invisibility Cloak* (ECCV 2020) — adversarial patches
- pyvirtualcam: https://github.com/letmaik/pyvirtualcam
