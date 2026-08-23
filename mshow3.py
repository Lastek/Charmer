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
CATEGORY_COLORS = {
    0: (0, 0, 0),  # background - black
    1: (0, 255, 0),  # hair       - green
    2: (255, 0, 0),  # body skin  - blue
    3: (0, 0, 255),  # face skin  - red
    4: (255, 255, 0),  # clothes    - cyan
    5: (255, 0, 255),  # others     - magenta
}

class VideoDenoiser:
    """Real-time video denoiser using ONNX Runtime or OpenCV fallback."""

    def __init__(self, model_path=None, use_onnx=False):
        self.model_path = model_path or DENOISE_MODEL_PATH
        self.use_onnx = use_onnx and ONNX_AVAILABLE
        self.session = None
        
        if self.use_onnx: self._init_onnx()
        else: print("Using OpenCV fastNlMeansDenoising (ONNX not available)")
    
    def _init_onnx(self):
        """Initialize ONNX Runtime session."""
        if not os.path.exists(self.model_path):
            print(f"ONNX model not found at {self.model_path}, falling back to OpenCV")
            self.use_onnx = False
            return
        
        try:
            providers = ['CPUExecutionProvider']
            if 'DmlExecutionProvider' in ort.get_available_providers():
                providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
            
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            print(f"ONNX denoiser loaded from {self.model_path}")
        except Exception as e:
            print(f"Failed to load ONNX model: {e}, falling back to OpenCV")
            self.use_onnx = False
    
    def denoise(self, frame):
        """Denoise a single frame."""
        if self.use_onnx and self.session is not None:
            return self._denoise_onnx(frame)
        return self._denoise_opencv(frame)
    
    def _denoise_onnx(self, frame):
        """Denoise using ONNX model."""
        h, w = frame.shape[:2]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0
        gray = np.expand_dims(gray, axis=(0, 1))
        
        noise_map = np.zeros((1, 1, h, w), dtype=np.float32)
        
        input_data = np.concatenate([gray, noise_map], axis=1)
        
        output = self.session.run(None, {self.input_name: input_data})[0]
        
        denoised = np.squeeze(output) * 255.0
        denoised = np.clip(denoised, 0, 255).astype(np.uint8)
        
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    
    def _denoise_opencv(self, frame):
        """Denoise using OpenCV fastNlMeansDenoising."""
        return cv2.fastNlMeansDenoisingColored(
            frame, None, h=10, hColor=10, 
            templateWindowSize=7, searchWindowSize=21
        ) # type: ignore



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
HAT_LOCAL = create_translation_matrix(0, -12.0, 6.0).T
HAT_LOCAL[1, 1] = -1  # flip Y axis
HAT_LOCAL = HAT_LOCAL @ create_rotation_matrix(np.radians(210), 'y')

GHOST_LOCAL = create_scale_matrix(4.4)
GHOST_LOCAL[1, 3] = -1.0
GHOST_LOCAL[2, 3] = 6.0


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
    beyond scipy's fixed-size euler/matrix conversions).

    Pipeline: spike gate -> OneEuro filter -> deadband -> dropout decay.
    Tuned stillness-first: heavy smoothing when slow, hard clamps on spikes.
    """

    MIN_CUTOFF = 0.35     # OneEuro: lower = smoother when still (more lag)
    BETA = 0.05           # OneEuro: speed coefficient
    DEADBAND_CM = 0.15    # output frozen below this filtered delta
    DEADBAND_DEG = 0.4
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
        self.t_out = np.zeros(3, dtype="f8")
        self.r_out = np.zeros(3, dtype="f8")
        self.out_matrix = np.eye(4, dtype="f4")
        self.hold_count = 0
        self.last_seen_ms = None

    def stabilize(self, matrix, now_ms):
        """matrix: raw 4x4 facial transformation matrix.
        Returns the shared out_matrix buffer (do not hold references)."""
        t_raw = matrix[:3, 3].astype("f8")
        # Ponytail: scipy euler/matrix conversions allocate two small
        # fixed-size arrays per call; unavoidable with this API and
        # negligible at webcam rates. Upgrade path: manual RPY extraction.
        rpy_raw = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=True)

        if self.last_seen_ms is None:
            np.copyto(self.t_state, t_raw)
            np.copyto(self.r_state, rpy_raw)
            np.copyto(self.t_out, t_raw)
            np.copyto(self.r_out, rpy_raw)
            self.out_matrix[:3, 3] = t_raw
            self.out_matrix[:3, :3] = Rotation.from_euler(
                "xyz", rpy_raw, degrees=True).as_matrix()
            self.last_seen_ms = now_ms
            return self.out_matrix

        if now_ms - self.last_seen_ms > self.STALE_MS:
            # Dropout decay toward neutral pose
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

        # Deadband: freeze output while the filtered delta stays tiny
        d_t = float(np.linalg.norm(self.t_state - self.t_out))
        d_r = float(np.abs(self.r_state - self.r_out).max())
        if d_t >= self.DEADBAND_CM or d_r >= self.DEADBAND_DEG:
            np.copyto(self.t_out, self.t_state)
            np.copyto(self.r_out, self.r_state)
            self.out_matrix[:3, 3] = self.t_state
            self.out_matrix[:3, :3] = Rotation.from_euler(
                "xyz", self.r_state, degrees=True).as_matrix()
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
void main() {
    gl_Position = u_proj * u_view * u_model * vec4(in_position, 1.0);
}
"""

GHOST_FRAG_SHADER = """
#version 330
out vec4 fragColor;
void main() {
    fragColor = vec4(0.8, 0.8, 0.8, 0.3);  // invisible but writes depth
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
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    gl_Position = u_proj * u_view * world_pos;
    gl_PointSize = u_point_size;
    v_color = in_color;
}
"""

POINT_FRAG_SHADER = """
#version 330
in vec3 v_color;
out vec4 fragColor;

void main() {
    fragColor = vec4(v_color, 1.0);
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
        self.meshes = []  # list of (vao, prog, index_count)
        self._setup_fbo(width, height)
        self._load_glb(glb_path)
        self._load_ghost_head(head_path)
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

    def _setup_fbo(self, w, h):
        self.color_tex = self.ctx.texture((w, h), 4)
        self.depth_buf = self.ctx.depth_renderbuffer((w, h))
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.color_tex], depth_attachment=self.depth_buf
        )

    def _load_ghost_head(self, path):
        """
        Load an external head mesh for depth-only occlusion.
        Expects a GLB or OBJ file centered roughly at the origin.
        """
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

        prog = self.ctx.program(
            vertex_shader=GHOST_VERT_SHADER, fragment_shader=GHOST_FRAG_SHADER
        )

        vbo = self.ctx.buffer(verts.tobytes())
        ibo = self.ctx.buffer(faces.tobytes())

        self.ghost_vao = self.ctx.vertex_array(prog, [(vbo, "3f", "in_position")], ibo)
        self.ghost_prog = prog
        self.ghost_face_count = faces.size

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

    def _make_projection(self, fov_deg=60.0):
        """Build a perspective matrix matching the webcam FOV."""
        fov = np.radians(fov_deg)
        asp = self.width / self.height
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

    def set_facemesh(self, landmarks, shape):
        """Update the facemesh vertices from MediaPipe landmarks."""
        h, w = shape
        # Convert from pixel space to NDC (-1 to 1)
        points = np.array([[
            (lm.x * w / w) * 2 - 1,  # NDC x
            -((lm.y * h / h) * 2 - 1),  # NDC y (flip Y)
            lm.z * 2  # scale Z for visibility
        ] for lm in landmarks], dtype='f4')
        colors = np.full((len(points), 3), 0.0, dtype='f4')
        colors[:, 1] = 1.0  # green

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
        # ── Step 1: Render ghost head (depth only, no color) ──────────
        ghost_model = GHOST_LOCAL
        if DEBUG:
            # Draw wireframe in color for FBO debug window
            pass
            # self.ctx.wireframe = True
            # self.ctx.color_mask = True, True, True, True
        else:
            # Depth only in production
            self.ctx.wireframe = False
            self.ctx.color_mask = False, False, False, False

        self.ghost_prog["u_model"].write(ghost_model.T.tobytes())
        self.ghost_prog["u_view"].write(view.T.tobytes())
        self.ghost_prog["u_proj"].write(proj.T.tobytes())
        self.ghost_vao.render()
        self.ctx.color_mask = True, True, True, True

        self.ctx.wireframe = False
        # ── Step 2: Render hat (depth tested against ghost head) ──────
        for vao, prog, _ in self.meshes:
            prog["u_model"].write(model.T.tobytes())
            prog["u_view"].write(view.T.tobytes())
            prog["u_proj"].write(proj.T.tobytes())
            prog["u_normal_mat"].write(normal_mat.astype("f4").tobytes())
            prog["u_light_dir"].value = (1.0, 2.0, 3.0)
            prog["u_light_color"].value = (1.0, 1.0, 1.0)
            prog["u_ambient"].value = (0.3, 0.3, 0.3)
            vao.render()

        # ── Debug: TNB axes + planes at bone origins (no depth test) ─
        if DEBUG:
            self.ctx.disable(moderngl.DEPTH_TEST)
            identity = np.eye(4, dtype="f4")
            self.debug_prog["u_model"].write(identity.T.tobytes())
            self.debug_prog["u_view"].write(view.T.tobytes())
            self.debug_prog["u_proj"].write(proj.T.tobytes())
            self.debug_vao.render(moderngl.LINES)
            self.ctx.enable(moderngl.DEPTH_TEST)

        # ── Step 3: Render facemesh points via GL ─────────────────────        if face_landmarks is not None:
            self.set_facemesh(face_landmarks, (self.height, self.width))
            # Points are already in NDC, use identity matrices
            identity = np.eye(4, dtype='f4')
            self.point_prog["u_model"].write(identity.T.tobytes())
            self.point_prog["u_view"].write(identity.T.tobytes())
            self.point_prog["u_proj"].write(identity.T.tobytes())
            self.point_prog["u_point_size"].value = 3.0
            self.facemesh_vao.render(moderngl.POINTS)

        # ── Read pixels and composite ─────────────────────────────
        raw = self.fbo.read(components=4)
        img = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 4)
        img = np.flipud(img).copy()  # OpenGL origin is bottom-left

        # ── Step 4: Apply segmentation mask ──────────────────────────
        if head_mask is not None:
            head_region = (head_mask > 128).squeeze().astype(np.uint8)
            img[:, :, 3] = img[:, :, 3] * (1 - head_region)

        result = composite_frame(frame, img)

        if DEBUG == True:
            # in render(), replace the composite block at the bottom with this temporarily:
            # raw = self.fbo.read(components=4)
            # img = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 4)
            # img = np.flipud(img)
            debug = img[:, :, :3].copy()  # just RGB, no composite
            # Facemesh is now rendered via GL pipeline above
            cv2.imshow("FBO Debug", debug)
            # DEBUG: Visualize all segmentation categories with different colors
            if head_mask is not None:
                mp_image = create_mediapipe_image(frame)
                _result = segmenter.segmenter.segment(mp_image)
                category = _result.category_mask.numpy_view().squeeze()
                cat_vis = np.zeros_like(frame)
                for cat_id, color in CATEGORY_COLORS.items():
                    cat_vis[category == cat_id] = color
                cat_vis = cv2.resize(cat_vis, (frame.shape[1], frame.shape[0]))
                cv2.imshow("Category Debug", cat_vis)
                # return frame  # return frame unchanged

        return result


# ── Main ──────────────────────────────────────────────────────────
def landmarkerAsyncCallback(result: vision.FaceLandmarkerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
    global LANDMARKER_RESULT
    LANDMARKER_RESULT = result


def main():
    global segmenter
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    h, w = frame.shape[:2]

    # denoiser = VideoDenoiser(use_onnx=False)
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
                # frame = denoiser.denoise(frame)

                mp_image = create_mediapipe_image(frame)
                timestamp_ms = time.monotonic_ns()//1_000_000
                landmarker.detect_async(mp_image, timestamp_ms)
                if LANDMARKER_RESULT and LANDMARKER_RESULT.facial_transformation_matrixes:
                    results = LANDMARKER_RESULT
                    face_matrix = np.array(results.facial_transformation_matrixes[0]).reshape(4, 4)
                    head_bone.local = stabilizer.stabilize(face_matrix, timestamp_ms)
                    # Get segmentation mask
                    # head_mask = segmenter.get_head_mask(frame)
                    head_mask = None

                    frame = renderer.render(frame, head_bone.world(), head_mask, results.face_landmarks[0])

                    frame_count += 1
                    if DEBUG and frame_count % 30 == 0:
                        print(bone_tree_debug(head_bone))

                    if DEBUG:
                        original_small = cv2.resize(original, (320, 240))
                        denoised_debug = cv2.resize(frame, (320, 240))
                        combined = np.hstack([original_small, denoised_debug])
                        cv2.imshow("Denoise Debug", combined)

                cv2.imshow("AR 3D Model", frame)
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
