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
PORTRAIT_RES = 384                          # square offscreen FBO size
PANEL_RECT = (0.35, -0.95, 0.95, 0.35)      # NDC: left, bottom, right, top
PORTRAIT_CAM_DIST = 3.5                     # portrait camera distance
PORTRAIT_FOV = 60.0
HOLO_TINT = (0.30, 0.80, 1.00)
PANEL_BASE_ALPHA = 0.08
PORTRAIT_POINT_SIZE = 4.0

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
in vec3 v_color;
out vec4 fragColor;

void main() {
    // round, soft-edged point sprite
    float d = length(gl_PointCoord - 0.5) * 2.0;
    float a = smoothstep(1.0, 0.5, d);
    fragColor = vec4(v_color, a);
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
uniform float u_time;
uniform vec3 u_tint;
uniform float u_base_alpha;

in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec4 content = texture(u_holo_tex, v_uv);
    float lum = dot(content.rgb, vec3(0.299, 0.587, 0.114));

    float scan    = 0.85 + 0.15 * sin(v_uv.y * 300.0 - u_time * 6.0);
    float flicker = 0.92 + 0.08 * sin(u_time * 47.0) * sin(u_time * 13.7);

    vec2  to_edge = min(v_uv, 1.0 - v_uv);
    float edge    = min(to_edge.x, to_edge.y);
    float border  = smoothstep(0.015, 0.0, edge);   // hard rim line
    float glow    = smoothstep(0.10, 0.0, edge);    // soft inner falloff

    vec3 col = u_tint * lum * scan * flicker
             + u_tint * (border * 0.9 + glow * 0.20);
    float alpha = clamp(lum * content.a * scan
                        + u_base_alpha * glow + border * 0.6, 0.0, 1.0);
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


class GLBRenderer:
    def __init__(self, glb_path, head_path, width, height):
        self.ctx = moderngl.create_standalone_context()
        self.width = width
        self.height = height
        # Runtime view toggles (flip live from a UI without rebuilding)
        self.show_facemesh = True
        self.show_hologram = True       # screen-space holo portrait panel
        self.show_axes = DEBUG          # TNB axis/plane debug lines
        self.ghost_shaded = DEBUG       # shaded ghost rig vs depth-only occluder
        self.last_fbo = None            # raw FBO RGB (pre-mask, pre-composite)
        self.meshes = []  # list of (vao, prog, index_count)
        self._setup_fbo(width, height)
        # self._load_glb(glb_path)
        self._load_glb('blank.glb')
        # self._load_ghost_head(head_path)
        self._load_ghost_head('blank.glb')
        self._setup_point_shader()
        self.facemesh_vbo = None
        self.facemesh_vao = None
        self.facemesh_count = 0

    def _setup_point_shader(self):
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

    def _setup_portrait_fbo(self):
        # Color-only offscreen target: points are alpha-blended, no depth
        # needed for a point cloud.
        self.portrait_tex = self.ctx.texture((PORTRAIT_RES, PORTRAIT_RES), 4)
        self.portrait_fbo = self.ctx.framebuffer(
            color_attachments=[self.portrait_tex]
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
        """Update the portrait point cloud from MediaPipe landmarks.
        Model space: image-normalized XY (y up), Z kept as MediaPipe
        relative depth. Ponytail: lm.z is relative depth, not true
        geometry — fine for a point cloud, wrong for solid shading."""
        points = np.array([[
            lm.x * 2 - 1,       # x, centered
            -(lm.y * 2 - 1),    # y up
            lm.z * 2,           # scaled for visible parallax
        ] for lm in landmarks], dtype='f4')
        colors = np.ones((len(points), 3), dtype='f4')  # white; panel tints
        vbo = self.ctx.buffer(points.tobytes())
        vbo_color = self.ctx.buffer(colors.tobytes())

        self.facemesh_vao = self.ctx.vertex_array(
            self.point_prog,
            [
                (vbo, "3f", "in_position"),
                (vbo_color, "3f", "in_color"),
            ],
        )
        self.facemesh_count = len(points)

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
            model = (MP_TO_PORTRAIT @ head_world @ MP_TO_PORTRAIT).astype("f4")
            model[:, 3] = (0, 0, 0, 1)  # drop head translation; portrait orbits
            view_p = create_translation_matrix(0.0, 0.0, -PORTRAIT_CAM_DIST)
            proj_p = self._make_projection(PORTRAIT_FOV, aspect=1.0)
            self.point_prog["u_model"].write(model.T.tobytes())
            self.point_prog["u_view"].write(view_p.T.tobytes())
            self.point_prog["u_proj"].write(proj_p.T.tobytes())
            self.point_prog["u_point_size"].value = PORTRAIT_POINT_SIZE
            self.facemesh_vao.render(moderngl.POINTS)
            self.ctx.disable(moderngl.BLEND)

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
            self.ctx.enable(moderngl.DEPTH_TEST)

        # ── Pass 1: hologram panel (screen space, over the scene) ──────
        if show_portrait:
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)
            self.portrait_tex.use(0)
            self.holo_prog["u_holo_tex"].value = 0
            self.holo_prog["u_time"].value = time.monotonic() % 100.0
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
    _, _, head_bone = build_skeleton()

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

    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                original = frame.copy()
                mp_image = create_mediapipe_image(frame)
                timestamp_ms = time.monotonic_ns()//1_000_000
                landmarker.detect_async(mp_image, timestamp_ms)
                if LANDMARKER_RESULT and LANDMARKER_RESULT.facial_transformation_matrixes:
                    results = LANDMARKER_RESULT
                    face_matrix = np.array(results.facial_transformation_matrixes[0]).reshape(4, 4)
                    head_bone.local = stabilizer.stabilize(face_matrix, timestamp_ms, LANDMARKER_RESULT_TS)
                    # Get segmentation mask
                    # head_mask = segmenter.get_head_mask(frame)
                    head_mask = None

                    frame = renderer.render(frame, head_bone.world(), head_mask, results.face_landmarks[0])
                    fbo_debug = renderer.last_fbo
                    frame_count += 1
                    if DEBUG and frame_count % 30 == 0:
                        print(bone_tree_debug(head_bone))

                    if DEBUG:
                        original_small = cv2.resize(original, (320, 240))
                        denoised_debug = cv2.resize(frame, (320, 240))
                        combined = np.hstack([original_small, denoised_debug])
                        cv2.imshow("Denoise Debug", combined)
                        cv2.imshow("FBO", fbo_debug)

                cv2.imshow("AR 3D Model", frame)
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
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
    view = create_translation_matrix(0.0, 0.0, -PORTRAIT_CAM_DIST)

    def ndc(model, p):
        # Mirror the shader: gl_Position = proj * view * model * pos, with
        # the numpy row-vector matrices uploaded transposed (column-major).
        clip = proj.T @ view.T @ model.T @ _np.array([*p, 1.0], dtype="f4")
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
