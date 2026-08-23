"""Self-check for the mshow3 bone rig. Run: python test_rig.py (conda env: media)"""
import numpy as np
from mshow3 import build_skeleton, HAT_LOCAL, GHOST_LOCAL, bone_tree_debug, build_axis_frame_lines

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

print("rig self-check OK")
