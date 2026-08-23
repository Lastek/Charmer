# Plan: MatrixStabilizer — stabilize facial_transformation_matrixes before bone tree

## Approved scope
- Smooth `results.facial_transformation_matrixes[0]` ONLY; skeleton stays a pure transform consumer.
- Techniques: OneEuro matrix filtering + spike gate + deadband + dropout decay.
- Tuning: stillness-first. Buffers preallocated; minimal per-frame allocs.

## Data flow
```
results.facial_transformation_matrixes[0]
  -> face_matrix = np.array(...).reshape(4,4)
  -> smoothed = stabilizer.stabilize(face_matrix, timestamp_ms)
  -> head_bone.local = smoothed          # skeleton untouched by smoothing
```

## Change 1: add MatrixStabilizer class (before GLB Renderer section)

Requires: `from scipy.spatial.transform import Rotation` added to imports
(scipy 1.17.1 already in media env; used in bone_tree_debug).

```python
# ── Matrix Stabilizer ─────────────────────────────────────────────
class MatrixStabilizer:
    """Stabilizes the facial transformation matrix before it reaches the
    bone tree. All state lives in preallocated buffers (no per-frame allocs
    beyond scipy's fixed-size euler/matrix conversions).

    Pipeline: spike gate -> OneEuro filter -> deadband -> dropout decay.
    Tuned stillness-first: heavy smoothing when slow, hard clamps on spikes.
    """

    MIN_CUTOFF = 0.35     # OneEuro: lower = smoother when still (more lag)
    BETA = 0.05           # OneEuro: speed coefficient
    DEADBAND_CM = 0.15    # translation delta below this -> output frozen
    DEADBAND_DEG = 0.4    # rotation delta below this -> output frozen
    GATE_CM = 8.0         # single-frame jump above this = spike -> reject
    GATE_DEG = 25.0
    HOLD_FRAMES = 5       # reject spikes for this many frames, then snap
    STALE_MS = 500        # no fresh matrix after this -> decay to neutral
    DECAY_RATE = 0.1      # fraction lerped toward neutral per stale frame

    def __init__(self):
        self.trans_filter = OneEuroFilter(self.MIN_CUTOFF, self.BETA)
        self.rot_filter = OneEuroFilter(self.MIN_CUTOFF, self.BETA)
        self.t_state = np.empty(3, dtype="f8")
        self.r_state = np.empty(3, dtype="f8")
        self.out_matrix = np.eye(4, dtype="f4")
        self.hold_count = 0
        self.last_seen_ms = None

    def stabilize(self, matrix, now_ms):
        """matrix: raw 4x4 facial transformation matrix.
        Returns shared out_matrix buffer (do not hold references)."""
        t_raw = matrix[:3, 3].astype("f8")
        rpy_raw = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=True)

        if self.last_seen_ms is None:
            np.copyto(self.t_state, t_raw)
            np.copyto(self.r_state, rpy_raw)
        elif now_ms - self.last_seen_ms > self.STALE_MS:
            decay = self.DECAY_RATE
            self.t_state *= (1 - decay)
            self.r_state *= (1 - decay)
        else:
            jump_t = float(np.linalg.norm(t_raw - self.t_state))
            jump_r = float(np.abs(rpy_raw - self.r_state).max())
            if (jump_t > self.GATE_CM or jump_r > self.GATE_DEG) \
                    and self.hold_count < self.HOLD_FRAMES:
                self.hold_count += 1          # reject spike, keep last stable
            elif self.hold_count:
                np.copyto(self.t_state, t_raw)  # snap after sustained jump
                np.copyto(self.r_state, rpy_raw)
                self.hold_count = 0
            else:
                self.trans_filter.filter(t_raw, now_ms / 1000.0, out=self.t_state)
                self.rot_filter.filter(rpy_raw, now_ms / 1000.0, out=self.r_state)

        self.last_seen_ms = now_ms
        self.out_matrix[:3, 3] = self.t_state
        self.out_matrix[:3, :3] = Rotation.from_euler(
            "xyz", self.r_state, degrees=True).as_matrix()
        return self.out_matrix
```

NOTE on `filter(..., out=...)`: existing OneEuroFilter.filter returns x_hat;
implementation will add optional `out=None` param writing result into caller
buffer when provided (small change to OneEuroFilter.filter, backward
compatible). If adding `out` feels intrusive, fallback is
`np.copyto(self.t_state, self.trans_filter.filter(t_raw, ...))`.

Deadband note: with MIN_CUTOFF=0.35 the OneEuro output moves < ~0.15cm when
the raw jitter is small, so the DEADBAND_* constants act on filtered deltas:
freeze output if |new_filtered - current_output| below threshold. Implement
as: after filtering, if delta < deadband -> keep previous out_matrix values.

## Change 2: wire into main loop

Replace (mshow3.py ~757):
```python
smoother = LandmarkSmoother(min_cutoff=0.2, beta=0.70)
```
with:
```python
stabilizer = MatrixStabilizer()
```

Replace (~789-791):
```python
smoother.smooth(results.face_landmarks[0], (h, w))
face_matrix = np.array(results.facial_transformation_matrixes[0]).reshape(4, 4)
head_bone.local = face_matrix.astype("f4")
```
with:
```python
face_matrix = np.array(results.facial_transformation_matrixes[0]).reshape(4, 4)
head_bone.local = stabilizer.stabilize(face_matrix, timestamp_ms)
```

LandmarkSmoother class stays (unused here; kept for facemesh debug later).

## Change 3: self-checks appended to test_rig.py

1. Noise rejection: constant pose + N(0, 1cm) noise over 300 frames ->
   std(output_translation_tail) < 0.5 * std(input noise).
2. Spike gate: steady sequence + one-frame 50cm jump -> output displacement <= GATE_CM that frame.
3. Convergence: true step of 10cm -> |output - target| < 0.5cm within 200 frames.
4. Dropout: feed matrices then advance now_ms by > STALE_MS repeatedly -> out_matrix[:3,3] shrinks toward zero.
5. Deadband: tiny (< DEADBAND_CM) oscillation around rest -> output exactly unchanged.

Run: `& C:\Dev\Anaconda3\envs\media\python.exe test_rig.py` plus pyflakes.

## Risks / notes
- Euler branch flips near upside-down poses: gated (discontinuous frame rejected as spike); ponytail comment included.
- Returned buffer is shared/reused — main loop must not retain references across frames (documented in docstring).
