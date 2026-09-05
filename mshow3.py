import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import moderngl
import trimesh
import os
from scipy.spatial.transform import Rotation
from mediapipe.tasks.python.vision import ImageSegmenter, ImageSegmenterOptions
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    print("ONNXRT IS AVAILABLE")
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNXRT FAILED TO LOAD")

DEBUG = True
MODEL_PATH = "face_landmarker.task"
GLB_PATH = "Pirate hat.glb"  # your GLB file
HEAD_PATH = "head.glb"
SEGMENTATION_MODEL = "selfie_multiclass_256x256.tflite"
DENOISE_MODEL_PATH = "dncnn.onnx"
LANDMARKER_RESULT = None # Reserved variable for the landmarker async callback: landmarkerAsyncCallback
LANDMARKER_RESULT_TS = None  # detection timestamp from the same callback



def flatten_scene_meshes(scene):
    """Extract all meshes from a trimesh scene."""
    if isinstance(scene, trimesh.Scene):
        raw_meshes = []
        for node_name in scene.graph.nodes_geometry:
            transform, geom_name = scene.graph[node_name]
            geom = scene.geometry[geom_name]
            if isinstance(geom, trimesh.Trimesh):
                geom = geom.copy()
                geom.apply_transform(transform)
                raw_meshes.append(geom)
    else:
        raw_meshes = [scene]
    return raw_meshes if raw_meshes else [trimesh.Trimesh()]


def create_rotation_matrix(angle, axis='y'):
    """Create a rotation matrix (radians) around X, Y, or Z axis."""
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        return np.array([[1,0,0,0], [0,c,-s,0], [0,s,c,0], [0,0,0,1]], dtype='f4')
    elif axis == 'y':
        return np.array([[c,0,s,0], [0,1,0,0], [-s,0,c,0], [0,0,0,1]], dtype='f4')
    return np.array([[c,-s,0,0], [s,c,0,0], [0,0,1,0], [0,0,0,1]], dtype='f4')


def create_mediapipe_image(frame):
    """Convert numpy frame to MediaPipe Image."""
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)


def composite_frame(frame, rendered_img):
    """Composite rendered AR frame over original using alpha channel."""
    alpha = rendered_img[:, :, 3:4] / 255.0
    rgb_bgr = rendered_img[:, :, 2::-1]
    return (frame * (1 - alpha) + rgb_bgr * alpha).astype(np.uint8)


def create_scale_matrix(scale):
    """Create a uniform scaling matrix."""
    return np.diag([scale, scale, scale, 1]).astype('f4')


def create_translation_matrix(x, y, z):
    """Create a translation matrix."""
    return np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [x,y,z,1]], dtype='f4')


def make_neck_pivot_model(rotation, neck_y, neck_z):
    """Rotate about a point 'neck' below/behind the face center:
    T(neck) @ R @ T(-neck). The face stays centered at rest, but yaw/pitch
    now pivot about the neck (column-vector convention, like the bone rig)."""
    R4 = np.eye(4, dtype="f4")
    R4[:3, :3] = rotation
    n = np.array([0.0, -neck_y, neck_z], dtype="f4")
    Tp = create_translation_matrix(n[0], n[1], n[2]).T
    Tn = create_translation_matrix(-n[0], -n[1], -n[2]).T
    return Tp @ R4 @ Tn


# ── Skeleton (bone tree) ──────────────────────────────────────────
class Bone:
    """Minimal bone: local 4x4 transform parented to another bone.
    World matrix = parent world * local (column-vector convention)."""

    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.local = np.eye(4, dtype="f4")

    def world(self):
        if self.parent is None:
            return self.local.astype("f4")
        return self.parent.world() @ self.local


def build_skeleton():
    """root(torso) -> neck -> head."""
    root = Bone("root")
    neck = Bone("neck", root)
    head = Bone("head", neck)
    return root, neck, head


def build_axis_frame_lines(length=3.0, plane_half=1.5):
    """Line-segment vertices for a TNB axis triad + three orthogonal
    plane squares (YZ, XZ, XY), centered on a bone origin."""
    verts, colors = [], []

    def seg(a, b, c):
        verts.append(a); verts.append(b); colors.extend([c, c])

    def quad(pts, c):
        for i in range(4):
            seg(pts[i], pts[(i + 1) % 4], c)

    # TNB axes: T=+X red, N=+Y green, B=+Z blue
    seg((0, 0, 0), (length, 0, 0), (1.0, 0.2, 0.2))
    seg((0, 0, 0), (0, length, 0), (0.2, 1.0, 0.2))
    seg((0, 0, 0), (0, 0, length), (0.3, 0.5, 1.0))

    s = plane_half
    grey = (0.5, 0.5, 0.5)
    quad(((0, -s, -s), (0, s, -s), (0, s, s), (0, -s, s)), grey)    # YZ
    quad(((-s, 0, -s), (s, 0, -s), (s, 0, s), (-s, 0, s)), grey)    # XZ
    quad(((-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)), grey)    # XY

    return np.array(verts, dtype="f4"), np.array(colors, dtype="f4")


def bone_tree_debug(*bones):
    """Debug summary of the rig: one line per bone with world
    translation and rotation (roll/pitch/yaw in degrees)."""
    lines = []
    for b in bones:
        w = b.world()
        t = w[:3, 3]
        rpy = Rotation.from_matrix(w[:3, :3]).as_euler("xyz", degrees=True)
        lines.append(
            f"{b.name:>5}: t=({t[0]:7.2f},{t[1]:7.2f},{t[2]:7.2f})  "
            f"rpy=({rpy[0]:6.1f},{rpy[1]:6.1f},{rpy[2]:6.1f})"
        )
    return "\n".join(lines)


# Attachment locals: constant transforms in head-bone space
# (MediaPipe canonical face space), rendered against the head-driven view.
HAT_LOCAL = create_translation_matrix(0, -13.0, 4.0).T@create_scale_matrix(1.3)
HAT_LOCAL[1, 1] = -1  # flip Y axis
HAT_LOCAL = HAT_LOCAL @ create_rotation_matrix(np.radians(210), 'y')

GHOST_LOCAL = create_scale_matrix(4.2)
GHOST_LOCAL[1, 3] = -4.0
GHOST_LOCAL[2, 3] = 4.0

# Ghost head debug lighting rig (head-local space, canonical face ~10 units
# tall): warm white key, cool blue fill, amber rim.
GHOST_LIGHTS_POS = np.array([(7.0, 4.0, 9.0), (-7.0, 2.0, 9.0), (0.0, 9.0, -8.0)], dtype="f4")
GHOST_LIGHTS_COLOR = np.array([
    (1.00, 0.95, 0.85),  # key: warm white
    (0.35, 0.55, 1.00),  # fill: cool blue
    (1.00, 0.70, 0.30),  # rim: amber
], dtype="f4")
GHOST_ALPHA = 0.3

# ── Holographic portrait panel ──────────────────────────────────────
# Game-style comms portrait: the facemesh point cloud is rendered as 3D
# into an offscreen portrait FBO (head rotation drives the model matrix,
# a fixed portrait camera looks at the face), then the texture is drawn
# on a fixed screen-space panel quad with holo styling (scanlines, rim
# glow, flicker). Two spaces, never mixed: 3D model space inside the
# portrait pass, raw NDC in the panel pass.
PORTRAIT_RES = 384                          # offscreen FBO height (px)
PANEL_RECT = (0.35, -0.95, 0.95, 0.35)      # NDC: left, bottom, right, top
PORTRAIT_CAM_DIST = 3.0                     # portrait camera distance
PORTRAIT_FOV = 56.0
PORTRAIT_FIT = 2.3                          # extra margin around x/y only
PORTRAIT_DEPTH = 0.9                        # face depth (z span) relative to size
# Per-axis head-rotation gain. Yaw (turn left/right) barely moves the nose
# (it sits near the centroid), while pitch (nod) swings the chin/forehead
# through a lot of depth; tune these independently to balance the feel.
PORTRAIT_YAW_GAIN = 3.0
PORTRAIT_PITCH_GAIN = 0.5
PORTRAIT_ROLL_GAIN = 1.0
# Neck pivot: the face center sits this far above (y) and forward (z) of the
# neck. Rotation pivots about the neck, so yaw swings the head instead of
# foreshortening the face in place. Normalized portrait units (face ~1).
PORTRAIT_NECK_Y = 0.55
PORTRAIT_NECK_Z = 0 # 0.20
# Same pivot in the metric AR rig (face ~10 units); placeholder until real
# hat/ghost GLBs are loaded, so it only affects the (currently blank) GLBs.
NECK_TO_FACE_METRIC = (0.0, 1.5, -3.0)
HOLO_TINT = (0.30, 0.80, 1.00)
PANEL_BASE_ALPHA = 0.85
GLOW_STRENGTH = 4.7                      # bloom add-back strength (0 = off)
BLOOM_DIV = 6                            # bloom downsample factor (wider = softer)
BLOOM_RADIUS = 56.0                       # gaussian sigma in downsampled pixels
PORTRAIT_POINT_SIZE = 5.0
OVERLAY_POINT_SIZE = 3.0

# MP -> portrait basis change (y down/z in -> y up/z out). Ponytail:
# chirality verified by eye, not derived from a spec; if the portrait
# turns mirrored, flip signs on this diagonal only.
MP_TO_PORTRAIT = np.diag([1.0, -1.0, -1.0, 1.0]).astype("f4")


def draw_facemesh(frame, landmarks, color=(0, 255, 0), radius=1):
    """Draw face landmarks on frame."""
    h, w = frame.shape[:2]
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), radius, color, -1)
    return frame



# ── One Euro Filter ───────────────────────────────────────────────
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, t, out=None):
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            if out is not None:
                np.copyto(out, x)
                return out
            return x
        dt = t - self.t_prev
        if dt <= 0:
            if out is not None:
                np.copyto(out, self.x_prev)
                return out
            return self.x_prev
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        if out is not None:
            np.copyto(out, x_hat)
            return out
        return x_hat


class LandmarkSmoother:
    def __init__(self, min_cutoff=1.0, beta=0.01):
        self.filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)

    def smooth(self, landmarks, shape):
        h, w = shape
        points = np.array([[lm.x * w, lm.y * h] for lm in landmarks])
        t = cv2.getTickCount() / cv2.getTickFrequency()
        return self.filter.filter(points, t)


class HeadSegmenter:
    # Category indices for selfie_multiclass model:
    # 0=background, 1=hair, 2=body_skin, 3=face_skin, 4=clothes, 5=others
    FACE_CATS = {2, 3}  # face + body skin
    HAIR_CAT = 1

    def __init__(self, model_path):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=True,
            output_confidence_masks=False,
        )
        self.segmenter = ImageSegmenter.create_from_options(options)

    def get_head_mask(self, frame):
        """Returns a binary mask (0 or 255) of head region."""
        mp_image = create_mediapipe_image(frame)
        result = self.segmenter.segment(mp_image)
        category = result.category_mask.numpy_view()

        # Combine face skin + hair into one mask
        mask = np.isin(category, [1, 2, 3]).astype(np.uint8) * 255

        # Resize to frame size if needed
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(
                mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        return mask

    # def get_head_mask(self, frame):
    #     """Returns a binary mask of hair region only."""
    #     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    #     result   = self.segmenter.segment(mp_image)
    #     category = result.category_mask.numpy_view()

    #     # Only mask hair (category 1) — ghost mesh handles face occlusion via depth
    #     mask = (category == 1).astype(np.uint8) * 255

    #     if mask.shape[:2] != frame.shape[:2]:
    #         mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
    #                         interpolation=cv2.INTER_LINEAR)
    #     return mask


# ── Matrix Stabilizer ─────────────────────────────────────────────
class MatrixStabilizer:
    """Stabilizes the facial transformation matrix before it reaches the
    bone tree. All state lives in preallocated buffers (no per-frame allocs
    beyond scipy's fixed-size conversions).

    Rotation is tracked as a rotation vector (axis-angle, degrees): branch
    free up to 180 deg total rotation, unlike euler which flips its
    decomposition near +/-90 deg pitch/yaw and wraps at 180.

    Pipeline: spike gate -> OneEuro filter -> soft tracking -> dropout decay.
    Tuned stillness-first: heavy smoothing when slow, hard clamps on spikes.
    Ponytail (Gemini review, rejected): quaternion SLERP output easing —
    per-frame deltas here are sub-degree so curvature is invisible, and
    scipy Slerp allocates per frame; spherical deadband — max/norm metric
    anisotropy is bounded by ~1.7x at a 0.4 deg scale, imperceptible.
    """

    MIN_CUTOFF = 0.35     # OneEuro: lower = smoother when still (more lag)
    BETA = 0.05           # OneEuro: speed coefficient
    DEADBAND_CM = 0.15    # creep zone boundary (output eases slowly inside)
    DEADBAND_DEG = 0.4
    TRACK_RATE_SLOW = 4.0   # 1/s easing rate inside deadband (breathing follow)
    TRACK_RATE_FAST = 60.0  # 1/s easing rate outside deadband
    GATE_CM = 8.0         # single-frame jump above this = spike -> reject
    GATE_DEG = 25.0
    HOLD_FRAMES = 5       # reject spikes for this many frames, then snap
    STALE_MS = 500        # detection older than this -> decay to neutral
    DECAY_RATE = 0.1      # fraction eased toward neutral per stale frame
    NEUTRAL_T = np.array([0.0, 0.0, -60.0])  # decay target, NOT the lens (0,0,0)

    def __init__(self):
        self.trans_filter = OneEuroFilter(self.MIN_CUTOFF, self.BETA)
        self.rot_filter = OneEuroFilter(self.MIN_CUTOFF, self.BETA)
        self.t_state = np.empty(3, dtype="f8")
        self.r_state = np.empty(3, dtype="f8")
        self.t_out = np.zeros(3, dtype="f8")
        self.r_out = np.zeros(3, dtype="f8")
        self.out_matrix = np.eye(4, dtype="f4")
        self.hold_count = 0
        self.last_seen_ms = None

    def stabilize(self, matrix, now_ms, last_detect_ms=None):
        """matrix: raw 4x4 facial transformation matrix.
        last_detect_ms: timestamp of the detection the matrix came from
                        (None = treat as fresh).
        Returns the shared out_matrix buffer (do not hold references)."""
        t_raw = matrix[:3, 3].astype("f8")
        # Ponytail: scipy conversions allocate two small fixed-size arrays
        # per call; negligible at webcam rates.
        rv_raw = np.degrees(Rotation.from_matrix(matrix[:3, :3]).as_rotvec())
        prev_ms = self.last_seen_ms

        if prev_ms is None:
            np.copyto(self.t_state, t_raw)
            np.copyto(self.r_state, rv_raw)
            np.copyto(self.t_out, t_raw)
            np.copyto(self.r_out, rv_raw)
            self.out_matrix[:3, 3] = t_raw
            self.out_matrix[:3, :3] = matrix[:3, :3]
            self.last_seen_ms = now_ms
            return self.out_matrix

        is_stale = (last_detect_ms is not None
                    and now_ms - last_detect_ms > self.STALE_MS)
        if is_stale:
            # Dropout: ease toward neutral pose. NEUTRAL_T keeps Z at a sane
            # head distance; decaying to origin would dive into the lens.
            self.t_state -= (self.t_state - self.NEUTRAL_T) * self.DECAY_RATE
            self.r_state *= (1 - self.DECAY_RATE)
        else:
            jump_t = float(np.linalg.norm(t_raw - self.t_state))
            jump_r = float(np.linalg.norm(rv_raw - self.r_state))
            if (jump_t > self.GATE_CM or jump_r > self.GATE_DEG) \
                    and self.hold_count < self.HOLD_FRAMES:
                self.hold_count += 1          # reject spike, keep last stable
            elif self.hold_count:
                np.copyto(self.t_state, t_raw)  # snap after sustained jump
                np.copyto(self.r_state, rv_raw)
                self.hold_count = 0
            else:
                self.trans_filter.filter(t_raw, now_ms / 1000.0, out=self.t_state)
                self.rot_filter.filter(rv_raw, now_ms / 1000.0, out=self.r_state)

        # Soft tracking: output always eases toward the filtered state.
        # Inside the deadband it creeps slowly (breathing is followed instead
        # of frozen -> no snap on exit); outside it closes fast. The easing is
        # exponential so no transition ever pops. Frame-rate independent.
        dt_s = min(max(now_ms - prev_ms, 1.0), 200.0) / 1000.0
        d_t = float(np.linalg.norm(self.t_state - self.t_out))
        r_delta = self.r_state - self.r_out
        d_r = float(np.linalg.norm(r_delta))  # rotvec norm == angle in deg
        rate = self.TRACK_RATE_SLOW \
            if (d_t < self.DEADBAND_CM and d_r < self.DEADBAND_DEG) \
            else self.TRACK_RATE_FAST
        alpha = 1.0 - float(np.exp(-dt_s * rate))
        self.t_out += (self.t_state - self.t_out) * alpha
        self.r_out += r_delta * alpha
        self.out_matrix[:3, 3] = self.t_out
        self.out_matrix[:3, :3] = Rotation.from_rotvec(
            np.radians(self.r_out)).as_matrix()
        self.last_seen_ms = now_ms
        return self.out_matrix


# ── GLB Renderer ──────────────────────────────────────────────────
VERT_SHADER = """
#version 330
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat3 u_normal_mat;

in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;

out vec3 v_normal;
out vec3 v_color;
out vec3 v_frag_pos;

void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    v_frag_pos  = world_pos.xyz;
    v_normal    = normalize(u_normal_mat * in_normal);
    v_color     = in_color;
    gl_Position = u_proj * u_view * world_pos;
}
"""

FRAG_SHADER = """
#version 330
uniform vec3 u_light_dir;
uniform vec3 u_light_color;
uniform vec3 u_ambient;

in vec3 v_normal;
in vec3 v_color;
in vec3 v_frag_pos;

out vec4 fragColor;

void main() {
    float diff    = max(dot(normalize(v_normal), normalize(u_light_dir)), 0.0);
    vec3  diffuse = diff * u_light_color;
    vec3  result  = (u_ambient + diffuse) * v_color;
    fragColor     = vec4(result, 1.0);
}
"""
GHOST_VERT_SHADER = """
#version 330
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
in vec3 in_position;
in vec3 in_normal;
out vec3 v_normal;
out vec3 v_frag_pos;

void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    v_frag_pos = world_pos.xyz;
    v_normal = normalize(mat3(u_model) * in_normal);
    gl_Position = u_proj * u_view * world_pos;
}
"""

GHOST_FRAG_SHADER = """
#version 330
uniform vec3 u_light_pos[3];
uniform vec3 u_light_color[3];
uniform float u_alpha;
in vec3 v_normal;
in vec3 v_frag_pos;
out vec4 fragColor;

void main() {
    vec3 N = normalize(v_normal);
    vec3 total = vec3(0.18);  // ambient
    for (int i = 0; i < 3; i++) {
        vec3 L = u_light_pos[i] - v_frag_pos;
        float dist = length(L);
        float atten = 1.0 / (1.0 + 0.05 * dist);
        total += max(dot(N, L / dist), 0.0) * u_light_color[i] * atten;
    }
    fragColor = vec4(total, u_alpha);
}
"""

POINT_VERT_SHADER = """
#version 330
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat4 u_model;
uniform float u_point_size;

in vec3 in_position;
in vec3 in_color;

out vec3 v_color;

void main() {
    gl_Position = u_proj * u_view * u_model * vec4(in_position, 1.0);
    gl_PointSize = u_point_size;
    v_color = in_color;
}
"""

POINT_FRAG_SHADER = """
#version 330
uniform vec3 u_tint;
uniform float u_holo;   // 1.0 = holographic styling, 0.0 = flat 2D overlay

in vec3 v_color;
out vec4 fragColor;

void main() {
    // round, soft-edged point sprite
    float d = length(gl_PointCoord - 0.5) * 2.0;
    float a = smoothstep(1.0, 0.5, d);
    // Only the tint here: scanlines/flicker are applied AFTER bloom (see
    // HOLO_FINISH_FRAG_SHADER), so the bloom doesn't pulse with the flicker.
    vec3 col = mix(v_color, v_color * u_tint, u_holo);
    fragColor = vec4(col, a);
}
"""

HOLO_SCREEN_VERT_SHADER = """
#version 330
in vec2 in_position;   // fixed NDC quad corners (PANEL_RECT)
in vec2 in_uv;
out vec2 v_uv;

void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

HOLO_SCREEN_FRAG_SHADER = """
#version 330
uniform sampler2D u_holo_tex;
uniform vec3 u_tint;
uniform float u_base_alpha;

in vec2 v_uv;
out vec4 fragColor;

void main() {
    // The portrait texture already carries the holographic styling
    // (tint/scanlines/flicker); here we only add the panel frame.
    vec4 content = texture(u_holo_tex, v_uv);

    vec2  to_edge = min(v_uv, 1.0 - v_uv);
    float edge    = min(to_edge.x, to_edge.y);
    float border  = smoothstep(0.015, 0.0, edge);   // hard rim line
    float glow    = smoothstep(0.10, 0.0, edge);    // soft inner falloff

    vec3 col = content.rgb + u_tint * (border * 0.9 + glow * 0.20);
    float alpha = clamp(content.a + u_base_alpha * glow + border * 0.6, 0.0, 1.0);
    fragColor = vec4(col, alpha);
}
"""

LINE_VERT_SHADER = """
#version 330
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
in vec3 in_position;
in vec3 in_color;
out vec3 v_color;

void main() {
    gl_Position = u_proj * u_view * u_model * vec4(in_position, 1.0);
    v_color = in_color;
}
"""

LINE_FRAG_SHADER = """
#version 330
in vec3 v_color;
out vec4 fragColor;

void main() {
    fragColor = vec4(v_color, 1.0);
}
"""

POST_VERT_SHADER = """
#version 330
in vec2 in_position;   // fullscreen NDC quad (-1..1)
out vec2 v_uv;

void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

DOWNSAMPLE_FRAG_SHADER = """
#version 330
uniform sampler2D u_tex;
uniform vec2 u_res;   // source (full-res) size in pixels

in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec2 d = 1.0 / u_res;
    vec3 acc = vec3(0.0);
    for (int y = -1; y <= 2; y++)
        for (int x = -1; x <= 2; x++)
            acc += texture(u_tex, v_uv + vec2(float(x), float(y)) * d).rgb;
    fragColor = vec4(acc / 16.0, 1.0);
}
"""

BLUR_FRAG_SHADER = """
#version 330
uniform sampler2D u_tex;
uniform vec2 u_res;      // texture size in pixels
uniform float u_radius;  // gaussian sigma, in pixels

in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec2 step = 1.0 / u_res;
    vec3 acc = vec3(0.0);
    float wsum = 0.0;
    int R = int(min(u_radius * 2.0, 16.0) + 0.5);
    for (int y = -R; y <= R; y++) {
        for (int x = -R; x <= R; x++) {
            float w = exp(-0.5 * float(x * x + y * y) / (u_radius * u_radius));
            acc += texture(u_tex, v_uv + vec2(float(x), float(y)) * step).rgb * w;
            wsum += w;
        }
    }
    fragColor = vec4(acc / max(wsum, 0.0001), 1.0);
}
"""

BLOOM_ADD_FRAG_SHADER = """
#version 330
uniform sampler2D u_tex;
uniform float u_strength;

in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec3 c = texture(u_tex, v_uv).rgb * u_strength;
    float l = dot(c, vec3(0.299, 0.587, 0.114));
    fragColor = vec4(c, l);
}
"""

HOLO_FINISH_FRAG_SHADER = """
#version 330
uniform sampler2D u_tex;
uniform float u_time;

in vec2 v_uv;
out vec4 fragColor;

void main() {
    // Scanlines + flicker applied AFTER bloom, so the glow is steady and
    // only the finished image pulses.
    vec4 c = texture(u_tex, v_uv);
    float scan    = 0.85 + 0.15 * sin(gl_FragCoord.y * 0.8 - u_time * 6.0);
    float flicker = 0.92 + 0.08 * sin(u_time * 47.0) * sin(u_time * 13.7);
    fragColor = vec4(c.rgb * scan * flicker, c.a);
}
"""


class GLBRenderer:
    def __init__(self, glb_path, head_path, width, height):
        self.ctx = moderngl.create_standalone_context()
        self.width = width
        self.height = height
        self.aspect = width / height     # frame aspect (w/h)
        # Runtime view toggles (flip live from a UI without rebuilding)
        self.show_facemesh = True
        self.show_hologram = True       # screen-space holo portrait panel
        self.show_axes = DEBUG          # TNB axis/plane debug lines
        self.ghost_shaded = DEBUG       # shaded ghost rig vs depth-only occluder
        self.glow = True                 # bloom/glow post-process on the portrait
        self.last_fbo = None            # raw FBO RGB (pre-mask, pre-composite)
        self.meshes = []  # list of (vao, prog, index_count)
        self._setup_fbo(width, height)
        # self._load_glb(glb_path)
        self._load_glb('blank.glb')
        # self._load_ghost_head(head_path)
        self._load_ghost_head('blank.glb')
        self._setup_point_shader()
        self.face_overlay_vao = None
        self.face_portrait_vao = None
        self.facemesh_count = 0
        self.neutral_rot = None     # first-frame head rotation (front reference)

    def _setup_point_shader(self):
        # Required in core profile: without this the vertex shader's
        # gl_PointSize is ignored and points clip to 1px (invisible once
        # down-sampled through the portrait FBO).
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.point_prog = self.ctx.program(
            vertex_shader=POINT_VERT_SHADER, fragment_shader=POINT_FRAG_SHADER
        )
        # Bone debug frame (TNB axes + planes), drawn as lines
        self.debug_prog = self.ctx.program(
            vertex_shader=LINE_VERT_SHADER, fragment_shader=LINE_FRAG_SHADER
        )
        dv, dc = build_axis_frame_lines()
        self.debug_vbo = self.ctx.buffer(dv.tobytes())
        self.debug_cbo = self.ctx.buffer(dc.tobytes())
        self.debug_vao = self.ctx.vertex_array(
            self.debug_prog,
            [
                (self.debug_vbo, "3f", "in_position"),
                (self.debug_cbo, "3f", "in_color"),
            ],
        )
        self.debug_vert_count = len(dv)

        # Hologram panel: fixed NDC quad (PANEL_RECT) that samples the
        # portrait FBO texture through the holo screen shader.
        self.holo_prog = self.ctx.program(
            vertex_shader=HOLO_SCREEN_VERT_SHADER,
            fragment_shader=HOLO_SCREEN_FRAG_SHADER,
        )
        l, b, r, t = PANEL_RECT
        holo_pos = np.array([(l, b), (l, t), (r, b), (r, t)], dtype="f4")
        holo_uv = np.array([(0, 0), (0, 1), (1, 0), (1, 1)], dtype="f4")
        self.holo_pbo = self.ctx.buffer(holo_pos.tobytes())
        self.holo_ubo = self.ctx.buffer(holo_uv.tobytes())
        self.holo_vao = self.ctx.vertex_array(
            self.holo_prog,
            [(self.holo_pbo, "2f", "in_position"),
             (self.holo_ubo, "2f", "in_uv")],
        )
        self._setup_portrait_fbo()
        self._setup_bloom()

    def _setup_bloom(self):
        # Post-process programs + fullscreen quad for the glow/bloom passes.
        self.post_down_prog = self.ctx.program(
            vertex_shader=POST_VERT_SHADER, fragment_shader=DOWNSAMPLE_FRAG_SHADER
        )
        self.post_blur_prog = self.ctx.program(
            vertex_shader=POST_VERT_SHADER, fragment_shader=BLUR_FRAG_SHADER
        )
        self.post_add_prog = self.ctx.program(
            vertex_shader=POST_VERT_SHADER, fragment_shader=BLOOM_ADD_FRAG_SHADER
        )
        self.post_finish_prog = self.ctx.program(
            vertex_shader=POST_VERT_SHADER, fragment_shader=HOLO_FINISH_FRAG_SHADER
        )
        quad = np.array([(-1, -1), (-1, 1), (1, -1), (1, 1)], dtype="f4")
        self.post_vbo = self.ctx.buffer(quad.tobytes())
        self.post_down_vao = self.ctx.vertex_array(
            self.post_down_prog, [(self.post_vbo, "2f", "in_position")]
        )
        self.post_blur_vao = self.ctx.vertex_array(
            self.post_blur_prog, [(self.post_vbo, "2f", "in_position")]
        )
        self.post_add_vao = self.ctx.vertex_array(
            self.post_add_prog, [(self.post_vbo, "2f", "in_position")]
        )
        self.post_finish_vao = self.ctx.vertex_array(
            self.post_finish_prog, [(self.post_vbo, "2f", "in_position")]
        )

    def _render_bloom(self):
        """Downsample the portrait, run a wide gaussian at low res, then add
        it back additively (LINEAR upsample spreads light around the points).
        Reads portrait_tex, writes bloom_low/bloom_blur, lands on portrait_fbo."""
        pw, ph = self.portrait_size

        # downsample (box 4x4): portrait (full) -> bloom_low
        self.bloom_low_fbo.use()
        self.portrait_tex.use(0)
        d = self.post_down_prog
        d["u_tex"].value = 0
        d["u_res"].value = (pw, ph)
        self.post_down_vao.render(moderngl.TRIANGLE_STRIP)

        # 2D gaussian at low res: bloom_low -> bloom_blur
        self.bloom_blur_fbo.use()
        self.bloom_low_tex.use(0)
        b = self.post_blur_prog
        b["u_tex"].value = 0
        b["u_res"].value = self.bloom_size
        b["u_radius"].value = BLOOM_RADIUS
        self.post_blur_vao.render(moderngl.TRIANGLE_STRIP)

        # additive add-back (LINEAR upsample) onto the portrait
        self.portrait_fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.ONE, moderngl.ONE)
        self.bloom_blur_tex.use(0)
        a = self.post_add_prog
        a["u_tex"].value = 0
        a["u_strength"].value = GLOW_STRENGTH
        self.post_add_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.disable(moderngl.BLEND)

    def _render_finish(self):
        """Scanlines + flicker over the bloomed portrait -> final texture."""
        self.portrait_final_fbo.use()
        self.portrait_tex.use(0)
        f = self.post_finish_prog
        f["u_tex"].value = 0
        f["u_time"].value = time.monotonic() % 100.0
        self.post_finish_vao.render(moderngl.TRIANGLE_STRIP)

    def _setup_portrait_fbo(self):
        # Size the offscreen target to the panel's pixel aspect so the
        # content is never squashed horizontally/vertically when the
        # portrait texture is stretched onto the panel quad. Color-only:
        # points are alpha-blended, no depth needed.
        ndc_w = PANEL_RECT[2] - PANEL_RECT[0]
        ndc_h = PANEL_RECT[3] - PANEL_RECT[1]
        self.portrait_aspect = (ndc_w / ndc_h) * self.aspect
        ph = PORTRAIT_RES
        pw = max(int(round(ph * self.portrait_aspect)), 1)
        self.portrait_size = (pw, ph)
        self.portrait_tex = self.ctx.texture(self.portrait_size, 4)
        self.portrait_fbo = self.ctx.framebuffer(
            color_attachments=[self.portrait_tex]
        )
        # Bloom at reduced resolution: spreading light is cheap and soft
        # at low res, then upsamples (LINEAR) when added back.
        self.bloom_size = (max(pw // BLOOM_DIV, 1), max(ph // BLOOM_DIV, 1))
        self.bloom_low_tex = self.ctx.texture(self.bloom_size, 4)
        self.bloom_low_fbo = self.ctx.framebuffer(
            color_attachments=[self.bloom_low_tex]
        )
        self.bloom_blur_tex = self.ctx.texture(self.bloom_size, 4)
        self.bloom_blur_fbo = self.ctx.framebuffer(
            color_attachments=[self.bloom_blur_tex]
        )
        # Finished portrait: points + bloom, with scan/flicker applied last.
        self.portrait_final_tex = self.ctx.texture(self.portrait_size, 4)
        self.portrait_final_fbo = self.ctx.framebuffer(
            color_attachments=[self.portrait_final_tex]
        )

    def _setup_fbo(self, w, h):
        self.color_tex = self.ctx.texture((w, h), 4)
        self.depth_buf = self.ctx.depth_renderbuffer((w, h))
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.color_tex], depth_attachment=self.depth_buf
        )

    def _load_ghost_head(self, path):
        """
        Load an external head mesh: shaded with a 3-light debug rig when
        DEBUG, depth-only occluder in production (color mask off).
        Expects a GLB or OBJ file centered roughly at the origin.
        """
        scene = trimesh.load(path, force="scene")
        scene = trimesh.load(path, force="scene")
        raw_meshes = flatten_scene_meshes(scene)

        # Merge all meshes into one for simplicity
        merged = trimesh.util.concatenate(raw_meshes)

        # ── Normalize to MediaPipe face space ─────────────────────────
        # MediaPipe canonical face is roughly 10 units tall centered at origin
        # We center the mesh and scale it to match
        merged.vertices -= merged.centroid
        scale_factor = -10.0 / merged.scale  # tune this if head is too big/small
        merged.vertices *= scale_factor

        print(
            f"Ghost head loaded: {len(merged.vertices)} verts, {len(merged.faces)} faces"
        )
        print(f"Ghost head bounds after normalize: {merged.bounds}")

        verts = merged.vertices.astype("f4")
        faces = merged.faces.astype("i4")
        merged.fix_normals()
        normals = merged.vertex_normals.astype("f4")

        prog = self.ctx.program(
            vertex_shader=GHOST_VERT_SHADER, fragment_shader=GHOST_FRAG_SHADER
        )

        vbo = self.ctx.buffer(verts.tobytes())
        vbo_n = self.ctx.buffer(normals.tobytes())
        ibo = self.ctx.buffer(faces.tobytes())

        self.ghost_vao = self.ctx.vertex_array(
            prog,
            [
                (vbo, "3f", "in_position"),
                (vbo_n, "3f", "in_normal"),
            ],
            ibo,
        )
        self.ghost_prog = prog
        self.ghost_face_count = faces.size

        # Light rig: positions pre-transformed into ghost world space
        # (GHOST_LOCAL is constant, so this is done once).
        lights_h = np.hstack([GHOST_LIGHTS_POS, np.ones((3, 1), dtype="f4")])
        lights_w = (GHOST_LOCAL @ lights_h.T).T[:, :3].astype("f4")
        prog["u_light_pos"].write(lights_w.tobytes())
        prog["u_light_color"].write(GHOST_LIGHTS_COLOR.tobytes())
        prog["u_alpha"].value = GHOST_ALPHA

    def _load_glb(self, path):
        scene = trimesh.load(path, force="scene")
        raw_meshes = flatten_scene_meshes(scene)

        if DEBUG == True:
            for mesh in raw_meshes:
                print(f"Mesh bounds: {mesh.bounds}")
                print(f"Mesh centroid: {mesh.centroid}")
                print(f"Mesh scale: {mesh.scale}")

        # for mesh in raw_meshes:
        #     # Center and normalize to unit size
        #     mesh.vertices -= mesh.centroid
        #     mesh.vertices /= mesh.scale
        for mesh in raw_meshes:
            mesh.vertices *= 2.2

        prog = self.ctx.program(vertex_shader=VERT_SHADER, fragment_shader=FRAG_SHADER)

        for mesh in raw_meshes:
            mesh.fix_normals()
            verts = mesh.vertices.astype("f4")
            normals = mesh.vertex_normals.astype("f4")
            faces = mesh.faces.astype("i4")
            mesh.vertices -= mesh.centroid
            # Vertex colors — fall back to a neutral grey
            try:
                colors = (
                    mesh.visual.to_color().vertex_colors[:, :3].astype("f4") / 255.0
                )
            except Exception:
                colors = np.full((len(verts), 3), 0.8, dtype="f4")

            vbo_v = self.ctx.buffer(verts.tobytes())
            vbo_n = self.ctx.buffer(normals.tobytes())
            vbo_c = self.ctx.buffer(colors.tobytes())
            ibo = self.ctx.buffer(faces.tobytes())

            vao = self.ctx.vertex_array(
                prog,
                [
                    (vbo_v, "3f", "in_position"),
                    (vbo_n, "3f", "in_normal"),
                    (vbo_c, "3f", "in_color"),
                ],
                ibo,
            )

            self.meshes.append((vao, prog, faces.size))

    def _make_projection(self, fov_deg=60.0, aspect=None):
        """Build a perspective matrix matching the webcam FOV."""
        fov = np.radians(fov_deg)
        asp = aspect if aspect is not None else self.width / self.height
        near, far = 0.01, 200.0
        f = 1.0 / np.tan(fov / 2)
        return np.array(
            [
                [f / asp, 0, 0, 0],
                [0, f, 0, 0],
                [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
                [0, 0, -1, 0],
            ],
            dtype="f4",
        )

    def set_facemesh(self, landmarks):
        """Build two point-cloud VAOs from MediaPipe landmarks.

        overlay  — full-image NDC (y up): drawn over the face in the
                   camera view with identity view/proj, so the dots sit
                   exactly on the tracked face.
        portrait — face-normalized model: centered and scaled so the face
                   fills [-1,1], y up, Z kept as relative depth. Drawn into
                   the offscreen portrait with the head-relative rotation.
        Ponytail: lm.z is relative depth, not true geometry — fine for a
        point cloud, wrong for solid shading."""
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype='f4')

        overlay = pts.copy()
        overlay[:, 0] = overlay[:, 0] * 2 - 1
        overlay[:, 1] = -(overlay[:, 1] * 2 - 1)
        overlay[:, 2] = overlay[:, 2] * 2

        # Isotropic model units: MediaPipe normalizes x by width and y by
        # height, so y must be divided by the frame aspect to reach the same
        # units as x — otherwise the face is squashed from the sides.
        iso_x = overlay[:, 0]
        iso_y = overlay[:, 1] / self.aspect

        cx = (iso_x.min() + iso_x.max()) / 2.0
        cy = (iso_y.min() + iso_y.max()) / 2.0
        ext = max(iso_x.max() - iso_x.min(), iso_y.max() - iso_y.min())
        ext = ext if ext > 1e-6 else 1.0

        portrait = np.empty_like(overlay)
        portrait[:, 0] = (iso_x - cx) / ext
        portrait[:, 1] = (iso_y - cy) / ext
        # depth: MediaPipe lm.z is not in the same units as x/y, so scale it
        # into a fixed fraction of the face width, centered on the face and
        # negated so the nose points toward the portrait camera (-z = near).
        zraw = overlay[:, 2] - overlay[:, 2].mean()
        zamp = max(abs(zraw.min()), abs(zraw.max()), 1e-6)
        portrait[:, 2] = -zraw / zamp * PORTRAIT_DEPTH
        # Keep depth full-strength; only inset x/y (FIT). Shrinking z too
        # makes yaw read as pure horizontal foreshortening.
        portrait[:, 0] *= PORTRAIT_FIT
        portrait[:, 1] *= PORTRAIT_FIT
        self.portrait_points = portrait
        self.z_raw_span = (float(pts[:, 2].min()), float(pts[:, 2].max()))

        colors = np.ones((len(pts), 3), dtype='f4')  # white; panel tints
        self.face_overlay_vao = self._make_point_vao(overlay, colors)
        self.face_portrait_vao = self._make_point_vao(portrait, colors)
        self.facemesh_count = len(pts)

    def _make_point_vao(self, points, colors):
        vbo = self.ctx.buffer(points.astype("f4").tobytes())
        vbo_color = self.ctx.buffer(colors.astype("f4").tobytes())
        return self.ctx.vertex_array(
            self.point_prog,
            [(vbo, "3f", "in_position"), (vbo_color, "3f", "in_color")],
        )

    def render(self, frame, head_world, head_mask=None, face_landmarks=None):
        """
        head_world: 4x4 head-bone world matrix in MediaPipe camera space
                    (torso @ ... @ head, from facial_transformation_matrixes).
        face_landmarks: optional list of face landmarks for debug visualization
        Returns frame with 3D model composited on top.
        """
        # ── Pass 0: hologram portrait (offscreen) ──────────────────────
        # Facemesh point cloud in model space, rotated by the head pose,
        # viewed by a fixed portrait camera. Output: portrait_tex.
        show_portrait = (face_landmarks is not None
                         and self.show_facemesh and self.show_hologram)
        if show_portrait:
            self.set_facemesh(face_landmarks)
            self.portrait_fbo.use()
            self.ctx.clear(0.0, 0.0, 0.0, 0.0)
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)
            # Relative head rotation from the first frame as "forward", so
            # the portrait starts front-facing and only the delta turns it.
            # The MP->portrait basis change (y/z flip) is applied last.
            head_rot = head_world[:3, :3].astype("f4")
            if self.neutral_rot is None:
                self.neutral_rot = head_rot
            delta = head_rot @ self.neutral_rot.T
            f3 = MP_TO_PORTRAIT[:3, :3]
            R = (f3 @ delta @ f3).astype("f4")
            # Rebalance the rotation per axis: yaw reads too weak and pitch
            # too strong on a near-flat face cloud. Decompose -> scale ->
            # recompose (small angles, euler is branch-stable here).
            e = Rotation.from_matrix(R).as_euler("xyz")
            e[0] *= PORTRAIT_PITCH_GAIN
            e[1] *= PORTRAIT_YAW_GAIN
            e[2] *= PORTRAIT_ROLL_GAIN
            R = Rotation.from_euler("xyz", e).as_matrix().astype("f4")
            # Pivot the rotation about the neck (see make_neck_pivot_model),
            # not the nose bridge.
            model = make_neck_pivot_model(R, PORTRAIT_NECK_Y, PORTRAIT_NECK_Z)
            # create_translation_matrix writes a row-vector matrix (translation in
            # the last row); transpose it into the column-vector convention
            # used everywhere else (translation in the last column), or the
            # camera offset lands in W and distance only changes clipping.
            view_p = create_translation_matrix(0.0, 0.0, -PORTRAIT_CAM_DIST).T
            proj_p = self._make_projection(PORTRAIT_FOV, aspect=self.portrait_aspect)
            self.point_prog["u_model"].write(model.T.tobytes())
            self.point_prog["u_view"].write(view_p.T.tobytes())
            self.point_prog["u_proj"].write(proj_p.T.tobytes())
            self.point_prog["u_point_size"].value = PORTRAIT_POINT_SIZE
            self.point_prog["u_tint"].value = HOLO_TINT
            self.point_prog["u_holo"].value = 1.0
            self.face_portrait_vao.render(moderngl.POINTS)
            self.ctx.disable(moderngl.BLEND)
            if self.glow:
                self._render_bloom()
            # Scanlines + flicker last (after bloom), so the glow is steady.
            self._render_finish()
            if DEBUG:
                raw_p = self.portrait_final_fbo.read(components=3)
                pw, ph = self.portrait_size
                self.last_portrait = np.frombuffer(
                    raw_p, dtype=np.uint8).reshape(ph, pw, 3)
                self.last_portrait = np.flipud(self.last_portrait.copy())

        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)

        # ── Coordinate system conversion ──────────────────────────
        # MediaPipe: Y down, Z into scene (right-handed, camera looks +Z)
        # OpenGL:    Y up,   Z out of screen (right-handed, camera looks -Z)
        # model_view = flip @ mediapipe_matrix.astype('f4')

        # # Separate into view (camera) and model transforms
        # # MediaPipe matrix already encodes model→camera, use as view matrix

        # Convert MediaPipe → OpenGL coordinate system
        # MediaPipe: Y down, camera looks +Z
        # OpenGL:    Y up,   camera looks -Z
        # We only flip Y for rotation, but must preserve negative Z for depth

        conv = np.diag([-1, -1, -1, 1]).astype("f4")

        # Apply conversion to the rotation part only, keep translation intact
        mp_mat = head_world.astype("f4")
        view = conv @ mp_mat

        # Fix translation direction
        view[0, 3] = -view[0, 3]  # correct X
        view[1, 3] = -view[1, 3]  # correct Y
        view[2, 3] = -abs(view[2, 3])  # keep Z negative (in front of camera)
        view[2, 3] = view[2, 3] - 1  # Shift Z

        # Hat: constant local transform in head-bone space; the head-driven
        # view carries the motion.
        model = HAT_LOCAL

        proj = self._make_projection()
        normal_mat = np.linalg.inv(model[:3, :3]).T
        # ── Step 1: Render ghost head ──────────────────────────────────
        # Shaded 3-light debug rig when ghost_shaded; otherwise depth-only
        # occluder (color writes off, restored after the draw).
        ghost_model = GHOST_LOCAL
        if not self.ghost_shaded:
            self.ctx.color_mask = False, False, False, False

        self.ghost_prog["u_model"].write(ghost_model.T.tobytes())
        self.ghost_prog["u_view"].write(view.T.tobytes())
        self.ghost_prog["u_proj"].write(proj.T.tobytes())
        self.ghost_vao.render()
        self.ctx.color_mask = True, True, True, True

        # ── Step 2: Render hat (depth tested against ghost head) ───────
        for vao, prog, _ in self.meshes:
            prog["u_model"].write(model.T.tobytes())
            prog["u_view"].write(view.T.tobytes())
            prog["u_proj"].write(proj.T.tobytes())
            prog["u_normal_mat"].write(normal_mat.astype("f4").tobytes())
            prog["u_light_dir"].value = (1.0, 2.0, 3.0)
            prog["u_light_color"].value = (1.0, 1.0, 1.0)
            prog["u_ambient"].value = (0.3, 0.3, 0.3)
            vao.render()

        # ── Debug: TNB axes + planes at bone origins (no depth test) ──
        if self.show_axes:
            self.ctx.disable(moderngl.DEPTH_TEST)
            identity = np.eye(4, dtype="f4")
            self.debug_prog["u_model"].write(identity.T.tobytes())
            self.debug_prog["u_view"].write(view.T.tobytes())
            self.debug_prog["u_proj"].write(proj.T.tobytes())
            self.debug_vao.render(moderngl.LINES)

        # ── Overlay facemesh points on the face (screen space) ──────
        # NDC points drawn with identity view/proj so they sit exactly on
        # the tracked face; depth off so nothing occludes them.
        if face_landmarks is not None and self.show_facemesh:
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)
            identity = np.eye(4, dtype="f4")
            self.point_prog["u_model"].write(identity.T.tobytes())
            self.point_prog["u_view"].write(identity.T.tobytes())
            self.point_prog["u_proj"].write(identity.T.tobytes())
            self.point_prog["u_point_size"].value = OVERLAY_POINT_SIZE
            self.point_prog["u_holo"].value = 0.0
            self.face_overlay_vao.render(moderngl.POINTS)
            self.ctx.disable(moderngl.BLEND)
            self.ctx.enable(moderngl.DEPTH_TEST)
        elif self.show_axes:
            self.ctx.enable(moderngl.DEPTH_TEST)

        # ── Pass 1: hologram panel (screen space, over the scene) ──────
        if show_portrait:
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)
            self.portrait_final_tex.use(0)
            self.holo_prog["u_holo_tex"].value = 0
            self.holo_prog["u_tint"].value = HOLO_TINT
            self.holo_prog["u_base_alpha"].value = PANEL_BASE_ALPHA
            self.holo_vao.render(moderngl.TRIANGLE_STRIP)
            self.ctx.disable(moderngl.BLEND)
            self.ctx.enable(moderngl.DEPTH_TEST)

        # ── Read pixels and composite ─────────────────────────────
        raw = self.fbo.read(components=4)
        img = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 4)
        img = np.flipud(img).copy()  # OpenGL origin is bottom-left
        self.last_fbo = img[:, :, :3].copy()  # raw FBO RGB for the FBO Debug view

        # ── Step 4: Apply segmentation mask ──────────────────────────
        if head_mask is not None:
            head_region = (head_mask > 128).squeeze().astype(np.uint8)
            img[:, :, 3] = img[:, :, 3] * (1 - head_region)
        if DEBUG:
            # in render(), replace the composite block at the bottom with this temporarily:
            # raw = self.fbo.read(components=4)
            # img = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 4)
            # img = np.flipud(img)
            # fbo_debug = img[:, :, :3].copy()  # just RGB, no composite
            # Facemesh is now rendered via GL pipeline above
            # cv2.imshow("FBO Debug", fbo_debug)
            # DEBUG: Visualize all segmentation categories with different colors
            if head_mask is not None:
                mp_image = create_mediapipe_image(frame)
                _result = segmenter.segmenter.segment(mp_image)
                category = _result.category_mask.numpy_view().squeeze()
                cat_vis = np.zeros_like(frame)
                for cat_id, color in CATEGORY_COLORS.items():
                    cat_vis[category == cat_id] = color
                cat_vis = cv2.resize(cat_vis, (frame.shape[1], frame.shape[0]))
                # cv2.imshow("Category Debug", cat_vis)
                # return frame  # return frame unchanged

        result = composite_frame(frame, img)

        return result


# ── Main ──────────────────────────────────────────────────────────
def landmarkerAsyncCallback(result: vision.FaceLandmarkerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
    global LANDMARKER_RESULT, LANDMARKER_RESULT_TS
    LANDMARKER_RESULT = result
    LANDMARKER_RESULT_TS = timestamp_ms


def main():
    global segmenter
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    h, w = frame.shape[:2]

    renderer = GLBRenderer(GLB_PATH, HEAD_PATH, w, h)
    stabilizer = MatrixStabilizer()
    segmenter = HeadSegmenter(SEGMENTATION_MODEL)
    root, neck, head_bone = build_skeleton()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_facial_transformation_matrixes=True,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_faces=1,
        min_face_detection_confidence=0.7,
        min_face_presence_confidence=0.7,
        min_tracking_confidence=0.85,
        result_callback=landmarkerAsyncCallback
    )
    fps = 0
    frame_count = 0
    old_time = 0
    new_time = 0
    start_time = time.monotonic_ns()//1_000_000 
    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                original = frame.copy()
                mp_image = create_mediapipe_image(frame)
                timestamp_ms = time.monotonic_ns()//1_000_000
                new_time = timestamp_ms
                landmarker.detect_async(mp_image, timestamp_ms)
                if LANDMARKER_RESULT and LANDMARKER_RESULT.facial_transformation_matrixes:
                    results = LANDMARKER_RESULT
                    face_matrix = np.array(results.facial_transformation_matrixes[0]).reshape(4, 4)
                    smoothed = stabilizer.stabilize(face_matrix, timestamp_ms, LANDMARKER_RESULT_TS)
                    # Realize the rig: the neck is the pivot (holds the pose
                    # rotation + position), the head carries only the fixed
                    # neck->face offset. head_bone.world() still reconstructs
                    # `smoothed`, so the AR attach is unchanged.
                    mx, my, mz = NECK_TO_FACE_METRIC
                    head_bone.local = create_translation_matrix(mx, my, mz).T
                    neck.local = smoothed @ create_translation_matrix(-mx, -my, -mz).T
                    # Get segmentation mask
                    # head_mask = segmenter.get_head_mask(frame)
                    head_mask = None

                    frame = renderer.render(frame, head_bone.world(), head_mask, results.face_landmarks[0])
                    fbo_debug = renderer.last_fbo
                    frame_count += 1
                    if DEBUG and frame_count % 30 == 0:
                        print(bone_tree_debug(head_bone))
                        zr = getattr(renderer, "z_raw_span", None)
                        if zr is not None:
                            print(f"lm.z raw span: min={zr[0]:+.4f} "
                                  f"max={zr[1]:+.4f} (range={zr[1]-zr[0]:.4f})")

                    if DEBUG:
                        cv2.imshow("FBO", fbo_debug)
                        if getattr(renderer, "last_portrait", None) is not None:
                            cv2.imshow("Portrait FBO", renderer.last_portrait)
                
                fps = (new_time-old_time)
                frame = cv2.putText(frame, f"{fps}", (32, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow("AR 3D Model", frame)
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
                old_time = timestamp_ms 
                frame_count=0
            
    finally:
        cv2.destroyAllWindows()
        cap.release()


def _check_portrait_projection():
    """Self-check: the fixed portrait camera must place the model origin at
    panel center, keep +X mapping right, and swing the nose horizontally
    under head yaw — so the portrait stays centered and live without the
    webcam."""
    import numpy as _np
    aspect = 1.0
    fov = _np.radians(PORTRAIT_FOV)
    near, far = 0.01, 200.0  # must mirror _make_projection
    f = 1.0 / _np.tan(fov / 2)
    proj = _np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]], dtype="f4")
    view = create_translation_matrix(0.0, 0.0, -PORTRAIT_CAM_DIST).T

    def ndc(model, p):
        # Mirror the shader: gl_Position = proj * view * model * pos. The
        # matrices are the numpy column-vector matrices as-written (the
        # ".T" at upload only fixes byte order, introducing no transpose).
        clip = proj @ view @ model @ _np.array([*p, 1.0], dtype="f4")
        return clip[:2] / clip[3]

    ident = _np.eye(4, dtype="f4")
    assert _np.allclose(ndc(ident, (0.0, 0.0, 0.0)), (0.0, 0.0), atol=1e-6), \
        "model origin must project to portrait center"
    assert ndc(ident, (0.5, 0.0, 0.0))[0] > 0, "model +X must project right"
    assert ndc(ident, (0.0, 0.5, 0.0))[1] > 0, "model +Y must project up"

    head = create_rotation_matrix(_np.radians(90), "y")
    head[:3, 3] = (0.0, 0.0, 0.0)
    model = MP_TO_PORTRAIT @ head @ MP_TO_PORTRAIT
    nose = ndc(model, (0.0, 0.0, 1.0))
    assert abs(nose[0]) > 0.2 and abs(nose[1]) < 1e-3, \
        "head yaw must move the nose horizontally"
    print("portrait projection check: OK")


if __name__ == "__main__":
    _check_portrait_projection()
    main()
