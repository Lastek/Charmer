import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import moderngl
import trimesh
import os

from mediapipe.tasks.python.vision import ImageSegmenter, ImageSegmenterOptions
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

DEBUG = True
MODEL_PATH = "face_landmarker.task"
GLB_PATH = "Pirate hat.glb"  # your GLB file
DENOISE_MODEL_PATH = "dncnn.onnx"

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
            frame, None, h=10, hForColorComponents=10, 
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

    def filter(self, x, t):
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x
        dt = t - self.t_prev
        if dt <= 0:
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

    def render(self, frame, mediapipe_matrix, head_mask=None, face_landmarks=None):
        """
        mediapipe_matrix: 4x4 numpy array from facial_transformation_matrixes[0]
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
        mp_mat = mediapipe_matrix.astype("f4")
        view = conv @ mp_mat

        # Fix translation direction
        view[0, 3] = -view[0, 3]  # correct X
        view[1, 3] = -view[1, 3]  # correct Y
        view[2, 3] = -abs(view[2, 3])  # keep Z negative (in front of camera)
        view[2, 3] = view[2, 3] - 1  # Shift Z

        angle = np.radians(210)  # clockwise around Y
        rot_y = create_rotation_matrix(angle, 'y')

        y_off = -12.0
        z_off = 6.0  # adjust front/back (negative = push back into head)

        flip_y = create_translation_matrix(0, y_off, z_off).T
        flip_y[1, 1] = -1  # flip Y axis

        model = flip_y @ rot_y  # apply rotation first, then flip

        proj = self._make_projection()
        normal_mat = np.linalg.inv(model[:3, :3]).T
        # ── Step 1: Render ghost head (depth only, no color) ──────────
        # ── Ghost head alignment ──────────────────────────────────────
        # Tune these to align the head mesh with your real head
        ghost_scale = 4.4  # overall size
        ghost_y_offset = -1.0  # up/down
        ghost_z_offset = 6.0  # front/back
        ghost_x_rot = 0.0  # pitch (nod)

        rot_x = create_rotation_matrix(np.radians(ghost_x_rot), 'x')

        ghost_model = create_scale_matrix(ghost_scale)
        ghost_model[1, 3] = ghost_y_offset
        ghost_model[2, 3] = ghost_z_offset
        ghost_model = ghost_model @ rot_x
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

        # ── Step 3: Render facemesh points via GL ─────────────────────
        if face_landmarks is not None:
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
SEGMENTATION_MODEL = "selfie_multiclass_256x256.tflite"
HEAD_PATH = "head.glb"
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
h, w = frame.shape[:2]

# denoiser = VideoDenoiser(use_onnx=False)
renderer = GLBRenderer(GLB_PATH, HEAD_PATH, w, h)
smoother = LandmarkSmoother(min_cutoff=1.0, beta=0.01)
segmenter = HeadSegmenter(SEGMENTATION_MODEL)

# _CTYPE_VALUE_MAP = types.MappingProxyType({
#     'IMAGE': 1,
#     'VIDEO': 2,
#     'LIVE_STREAM': 3,
# })

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_facial_transformation_matrixes=True,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_faces=1,
    min_face_detection_confidence=0.7,
    min_face_presence_confidence=0.7,
    min_tracking_confidence=0.75,
)

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        original = frame.copy()
        # frame = denoiser.denoise(frame)

        mp_image = create_mediapipe_image(frame)
        results = landmarker.detect(mp_image)

        if results.face_landmarks and results.facial_transformation_matrixes:
            smoother.smooth(results.face_landmarks[0], (h, w))
            face_matrix = np.array(results.facial_transformation_matrixes[0]).reshape(
                4, 4
            )
            # Get segmentation mask
            # head_mask = segmenter.get_head_mask(frame)
            head_mask = None

            # print(face_matrix)  # add this
            frame = renderer.render(frame, face_matrix, head_mask, results.face_landmarks[0])

            if DEBUG:
                original_small = cv2.resize(original, (320, 240))
                denoised_debug = cv2.resize(frame, (320, 240))
                combined = np.hstack([original_small, denoised_debug])
                cv2.imshow("Denoise Debug", combined)

        cv2.imshow("AR 3D Model", frame)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
