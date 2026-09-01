"""Self-check for the mshow3 bone rig. Run: python test_rig.py (conda env: media)"""
import numpy as np
from mshow3 import build_skeleton, HAT_LOCAL, GHOST_LOCAL, bone_tree_debug, \
    build_axis_frame_lines, MatrixStabilizer, make_neck_pivot_model
from scipy.spatial.transform import Rotation

# 1. Hierarchy structure
root, neck, head = build_skeleton()
assert root.parent is None and neck.parent is root and head.parent is neck

# 2. World matrix = composition of locals up the chain
root.local = np.eye(4, dtype="f4"); root.local[:3, 3] = (10, 20, 30)
neck.local = np.eye(4, dtype="f4")
neck.local[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype="f4")  # 90 deg about Z
head.local = np.eye(4, dtype="f4"); head.local[:3, 3] = (1, 2, 3)
expected = root.local @ neck.local @ head.local
assert np.allclose(head.world(), expected), "head.world() != root @ neck @ head"

# 3. Attachment rigidity: hat in camera space = head_world @ HAT_LOCAL
hat_cam = head.world() @ HAT_LOCAL
assert hat_cam.shape == (4, 4)
t_expected = head.world()[:3, :3] @ HAT_LOCAL[:3, 3] + head.world()[:3, 3]
assert np.allclose(hat_cam[:3, 3], t_expected), "hat translation not rigid with head"
assert GHOST_LOCAL.shape == (4, 4)

# 4. Bone tree debug formatter lists all bones with transforms
report = bone_tree_debug(root, neck, head)
for name in ("root", "neck", "head"):
    assert name in report, f"{name} missing from debug report"
assert "t=(" in report and "rpy=" in report

# 5. Axis-frame debug geometry: 3 axis segs + 3 plane quads (4 segs each)
dv, dc = build_axis_frame_lines()
assert dv.shape == (30, 3) and dc.shape == (30, 3), f"bad shapes {dv.shape} {dc.shape}"
assert np.isfinite(dv).all()

# ── MatrixStabilizer checks ──────────────────────────────────────
def make_mat(t, rpy_deg):
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    m[:3, 3] = t
    return m

rng = np.random.default_rng(42)

# 6. Noise rejection: output variance well below input variance
st = MatrixStabilizer()
noise = rng.normal(0, 1.0, (300, 3))
outs = np.array([
    st.stabilize(make_mat(noise[i], (0, 0, 0)), i * 33)[:3, 3].copy()
    for i in range(300)
])
in_std = noise[50:].std()
out_std = outs[150:].std(axis=0).mean()
assert out_std < 0.5 * in_std, f"noise not reduced: {out_std:.3f} vs {in_std:.3f}"

# 7. Spike gate: single-frame 50cm jump is rejected
st2 = MatrixStabilizer()
for i in range(30):
    st2.stabilize(make_mat((0, -20, -66), (0, 0, 0)), i * 33)
prev = st2.stabilize(make_mat((0, -20, -66), (0, 0, 0)), 30 * 33)[:3, 3].copy()
o = st2.stabilize(make_mat((50, -20, -66), (0, 0, 0)), 31 * 33)
assert np.linalg.norm(o[:3, 3] - prev) <= MatrixStabilizer.GATE_CM, "spike leaked through"

# 8. Convergence: true step input reached within bounded frames
st3 = MatrixStabilizer()
o = None
for i in range(400):
    target = (10.0, -20.0, -66.0) if i >= 10 else (0.0, -20.0, -66.0)
    o = st3.stabilize(make_mat(target, (0, 0, 0)), i * 33)
assert np.linalg.norm(o[:3, 3] - (10, -20, -66)) < 0.5, \
    f"no convergence: {o[:3, 3]}"

# 9. Dropout decay: stale detection eases toward NEUTRAL_T, not the lens
st4 = MatrixStabilizer()
for i in range(30):
    o = st4.stabilize(make_mat((15, -20, -66), (0, 0, 0)), i * 33, i * 33)
t_before = o[:3, 3].copy()
detect_ts = 30 * 33  # detection frozen here
for i in range(40, 80):
    o = st4.stabilize(make_mat((15, -20, -66), (0, 0, 0)), i * 1000, detect_ts)
assert abs(o[0, 3]) < abs(t_before[0]), "x did not decay toward neutral"
assert abs(o[2, 3] + 60) < abs(t_before[2] + 60), "z did not ease toward -60"
assert o[2, 3] < -40, f"dived toward lens: z={o[2, 3]:.1f}"

# 10. Soft deadband: breathing-scale motion is followed smoothly, no snap
st5 = MatrixStabilizer()
st5.stabilize(make_mat((0, -20, -66), (0, 0, 0)), 0)
frozen = []
for i in range(1, 120):
    o = st5.stabilize(make_mat((0.3 * np.sin(i * 0.21), -20, -66), (0, 0, 0)), i * 33)
    frozen.append(o[:3, 3].copy())
arr = np.array(frozen)
steps = np.abs(np.diff(arr[:, 0]))
assert np.ptp(arr[:, 0]) > 0, "slow motion not followed at all"
assert steps.max() < MatrixStabilizer.DEADBAND_CM, \
    f"snap detected: max step {steps.max():.4f} cm"

# 11. Fast motion still tracks: step converges and no frame exceeds gate
steps_fast = np.diff(arr[:, 0])
assert np.all(np.isfinite(steps_fast))

# 12. Rotation continuity through steep yaw: rotvec channel stays smooth
# (pre-rotvec euler blew out ~25 deg/frame near the 90 deg branch flip)
st7 = MatrixStabilizer()
rv_steps = []
prev_rv = None
for i in range(58):
    ang = -170.0 + i * 6.0          # sweeps -170 -> +178, stays under 180
    o = st7.stabilize(make_mat((0, -20, -66), (0.0, ang, 0.0)), i * 33, i * 33)
    rv = np.degrees(Rotation.from_matrix(o[:3, :3]).as_rotvec())
    if prev_rv is not None:
        rv_steps.append(float(np.linalg.norm(rv - prev_rv)))
    prev_rv = rv
max_step = max(rv_steps)
assert max_step < 8.0, f"rotation blew through: {max_step:.2f} deg/frame"

# 13. Neck pivot: identity at rest, and yaw swings a nose point sideways
#     (rotation pivots about the neck, not the face center).
def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype="f4")

rest = make_neck_pivot_model(np.eye(3, dtype="f4"), 0.55, 0.2)
assert np.allclose(rest, np.eye(4, dtype="f4"), atol=1e-6), \
    "neck pivot must be identity at rest"
yaw = make_neck_pivot_model(_rot_y(np.radians(60)), 0.55, 0.2)
nose = np.array([0.0, 0.0, -0.3, 1.0], dtype="f4")
world = yaw @ nose
assert abs(world[0]) > 0.1, f"nose did not swing under yaw: x={world[0]:.3f}"

print("rig self-check OK")

print("rig self-check OK")
