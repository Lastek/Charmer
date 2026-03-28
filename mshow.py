import os
# os.environ['PYOPENGL_PLATFORM'] = 'pyglet'

import cv2
import mediapipe as mp
from mediapipe.tasks import python

from mediapipe.tasks.python import vision
import numpy as np
# import pyrender
import moderngl
import trimesh
import glm

MODEL_PATH = "face_landmarker.task"
HAT_PATH = "hat.png"
MODEL_3D_PATH = "Pirate hat.glb"  # GLTF/GLB/OBJ
# --- One Euro Filter ---
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        """
        min_cutoff: lower = smoother when still (but more lag)
        beta:       higher = more responsive during fast movement
        d_cutoff:   cutoff for the derivative filter, usually leave at 1.0
        """
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

        # Derivative estimate
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        # Dynamic cutoff based on speed
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)

        # Filter the signal
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
        t = cv2.getTickCount() / cv2.getTickFrequency()  # current time in seconds
        return self.filter.filter(points, t)

# ── 3D Model Renderer ─────────────────────────────────────────────

# class ModelRenderer:
#     def __init__(self, model_path, frame_shape):
#         h, w = frame_shape[:2]
#         self.w, self.h = w, h

#         # Standalone context — no window needed, works on AMD
#         self.ctx = moderngl.create_standalone_context()

#         # Framebuffer to render into
#         self.fbo = self.ctx.framebuffer(
#             color_attachments=[self.ctx.texture((w, h), 4)],
#             depth_attachment=self.ctx.depth_renderbuffer((w, h))
#         )

#         # Shaders — basic Phong
#         self.prog = self.ctx.program(
#             vertex_shader="""
#                 #version 330
#                 in vec3 in_position;
#                 in vec3 in_normal;

#                 uniform mat4 mvp;
#                 uniform mat4 model;

#                 out vec3 frag_normal;
#                 out vec3 frag_pos;

#                 void main() {
#                     gl_Position = mvp * vec4(in_position, 1.0);
#                     frag_pos    = vec3(model * vec4(in_position, 1.0));
#                     frag_normal = mat3(transpose(inverse(model))) * in_normal;
#                 }
#             """,
#             fragment_shader="""
#                 #version 330
#                 in vec3 frag_normal;
#                 in vec3 frag_pos;

#                 uniform vec3 light_dir;
#                 uniform vec3 color;

#                 out vec4 out_color;

#                 void main() {
#                     vec3 norm     = normalize(frag_normal);
#                     float diff    = max(dot(norm, normalize(light_dir)), 0.0);
#                     vec3 ambient  = 0.3 * color;
#                     vec3 diffuse  = diff * color;
#                     float alpha   = 1.0;
#                     out_color     = vec4(ambient + diffuse, alpha);
#                 }
#             """
#         )

#         # Load model with trimesh
#         scene = trimesh.load(model_path)
#         if isinstance(scene, trimesh.Scene):
#             mesh = trimesh.util.concatenate(scene.dump())
#         else:
#             mesh = scene

#         vertices = mesh.vertices.astype(np.float32)
#         normals  = mesh.vertex_normals.astype(np.float32)
#         indices  = mesh.faces.astype(np.uint32).flatten()

#         # Normalize model size and center it
#         center = (vertices.max(axis=0) + vertices.min(axis=0)) / 3.0
#         scale  = 1.0 / np.max(vertices.max(axis=0) - vertices.min(axis=0))
#         vertices = (vertices - center) * scale

#         # Upload to GPU
#         vbo_pos = self.ctx.buffer(vertices.tobytes())
#         vbo_nor = self.ctx.buffer(normals.tobytes())
#         ibo     = self.ctx.buffer(indices.tobytes())

#         self.vao = self.ctx.vertex_array(
#             self.prog,
#             [
#                 (vbo_pos, "3f", "in_position"),
#                 (vbo_nor, "3f", "in_normal"),
#             ],
#             ibo
#         )

#         # Camera intrinsics
#         self.focal = float(w)
#         self.cx    = w / 2.0
#         self.cy    = h / 2.0

#         self.ctx.enable(moderngl.DEPTH_TEST)
#         self.ctx.enable(moderngl.CULL_FACE)

#     def _mediapipe_to_glm(self, pose_matrix):
#         """Convert MediaPipe 4x4 pose to a GLM view matrix."""
#         m = pose_matrix.copy().astype(np.float64)
#         m[:, 1] *= 1
#         m[:, 2] *= 1
#         return glm.mat4(*m.T.flatten())

#     def render(self, frame, pose_matrix):
#         self.fbo.use()
#         self.ctx.clear(0.0, 0.0, 0.0, 0.0)

#         h, w = self.h, self.w

#         # Projection matrix from camera intrinsics
#         near, far = 0.01, 100.0
#         proj = glm.mat4(
#             2*self.focal/w,  0,                0,                          0,
#             0,               2*self.focal/h,   0,                          0,
#             1 - 2*self.cx/w, 2*self.cy/h - 1, (far+near)/(near-far),     -1,
#             0,               0,                2*far*near/(near-far),       0
#         )

#         model_mat = glm.mat4(1.0)  # identity — pose drives the view
#         view      = self._mediapipe_to_glm(pose_matrix)
#         mvp       = proj * view * model_mat

#         self.prog["mvp"].write(bytes(mvp))
#         self.prog["model"].write(bytes(model_mat))
#         self.prog["light_dir"].value = (1.0, 1.0, 1.0)
#         self.prog["color"].value     = (0.8, 0.7, 0.6)  # tweak for model tint

#         self.vao.render(moderngl.TRIANGLES)

#         # Read pixels back as RGBA
#         data  = self.fbo.read(components=4)
#         color = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 4))
#         color = np.flipud(color)  # OpenGL origin is bottom-left

#         # Alpha composite onto frame
#         alpha = color[:, :, 3:4] / 255.0
#         rgb   = color[:, :, :3]
#         frame = (frame * (1 - alpha) + rgb * alpha).astype(np.uint8)

#         return frame

class ModelRenderer:
    def __init__(self, model_path, frame_shape):
        h, w = frame_shape[:2]
        self.w, self.h = w, h

        self.ctx = moderngl.create_standalone_context()

        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((w, h), 4)],
            depth_attachment=self.ctx.depth_renderbuffer((w, h))
        )

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_position;
                in vec3 in_normal;

                uniform mat4 mvp;
                uniform mat4 model;

                out vec3 frag_normal;
                out vec3 frag_pos;

                void main() {
                    gl_Position = mvp * vec4(in_position, 1.0);
                    frag_pos    = vec3(model * vec4(in_position, 1.0));
                    frag_normal = mat3(model) * in_normal;
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 frag_normal;
                in vec3 frag_pos;

                uniform vec3 light_dir;
                uniform vec3 color;

                out vec4 out_color;

                void main() {
                    vec3 norm    = normalize(frag_normal);
                    float diff   = max(dot(norm, normalize(light_dir)), 0.0);
                    vec3 ambient = 0.3 * color;
                    vec3 diffuse = diff * color;
                    out_color    = vec4(ambient + diffuse, 1.0);
                }
            """
        )

        # Load + normalize model
        scene = trimesh.load(model_path)
        if isinstance(scene, trimesh.Scene):
            mesh = trimesh.util.concatenate(scene.dump())
        else:
            mesh = scene

        vertices = mesh.vertices.astype(np.float32)
        normals  = mesh.vertex_normals.astype(np.float32)
        indices  = mesh.faces.astype(np.uint32).flatten()

        center   = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
        scale    = 1.0 / np.max(vertices.max(axis=0) - vertices.min(axis=0))
        vertices = (vertices - center) * scale

        vbo_pos = self.ctx.buffer(vertices.tobytes())
        vbo_nor = self.ctx.buffer(normals.tobytes())
        ibo     = self.ctx.buffer(indices.tobytes())

        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (vbo_pos, "3f", "in_position"),
                (vbo_nor, "3f", "in_normal"),
            ],
            ibo
        )

        self.focal = float(w)
        self.cx    = w / 2.0
        self.cy    = h / 2.0
        self.near  = 0.01
        self.far   = 200.0

        self.ctx.enable(moderngl.DEPTH_TEST)

    def _build_projection(self):
        """Build OpenCV-compatible projection matrix as numpy, column-major for GL."""
        f, n  = self.far, self.near
        w, h  = self.w, self.h
        fx    = self.focal
        fy    = self.focal
        cx    = self.cx
        cy    = self.cy

        # Row-major first, then transpose for column-major GL
        proj = np.array([
            [2*fx/w,  0,         1 - 2*cx/w,          0           ],
            [0,       2*fy/h,    2*cy/h - 1,           0           ],
            [0,       0,        -(f+n)/(f-n),  -2*f*n/(f-n)        ],
            [0,       0,        -1,                     0           ],
        ], dtype=np.float32)

        return proj.T  # transpose to column-major

    def render(self, frame, pose_matrix, smoothed_points, landmarks_3d):
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)

        h, w = self.h, self.w

        # Extract only rotation from pose matrix, discard translation
        rot = pose_matrix[:3, :3].copy().astype(np.float32)
        rot[:, 1] *= 1
        rot[:, 2] *= 1
        rot[:, 0] *= 1

        # Use forehead landmark (10) to anchor position in image space
        anchor = smoothed_points[10]
        # Convert image coords to normalized device coords (-1 to 1)
        nx = (anchor[0] / w) * 2 - 1
        ny = ((anchor[1] / h) * 2 - 1)  # flip Y for GL
        tz = 0.0  # push model back into scene, tune this

        # Build model matrix from rotation + landmark-derived translation
        model_mat = np.eye(4, dtype=np.float32)
        model_mat[:3, :3] = rot
        model_mat[0, 3]   = nx
        model_mat[1, 3]   = ny + 0.3   # nudge upward to sit on head, tune this
        model_mat[2, 3]   = tz

        view = np.eye(4, dtype=np.float32)
        proj = self._build_projection()
        mvp  = model_mat

        self.prog["mvp"].write(mvp.tobytes())
        self.prog["model"].write(model_mat.tobytes())
        self.prog["light_dir"].value = (1.0, 1.0, 1.0)
        self.prog["color"].value     = (0.8, 0.7, 0.6)

        self.vao.render(moderngl.TRIANGLES)

        data  = self.fbo.read(components=4)
        color = np.frombuffer(data, dtype=np.uint8).reshape((self.h, self.w, 4))
        color = np.flipud(color)

        alpha = color[:, :, 3:4] / 255.0
        rgb   = color[:, :, :3]
        frame = (frame * (1 - alpha) + rgb * alpha).astype(np.uint8)

        return frame
    

def draw_landmarks(frame, points):
    for (x, y) in points:
        cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
    return frame

def draw_overlay_hat(frame, hat_img, points):
    """
    anchor landmarks used:
      234  → left temple
      454  → right temple
      10   → top of forehead (center)
    the hat bottom edge aligns with the temple line,
    and rises above the head scaled to the hat aspect ratio.
    """
    h_frame, w_frame = frame.shape[:2]
    h_hat, w_hat = hat_img.shape[:2]

    left  = points[234]   # left temple
    right = points[454]   # right temple
    top   = points[10]    # forehead top center

    # hat width = temple-to-temple distance with padding
    temple_vec = right - left
    hat_width_px = np.linalg.norm(temple_vec) * 1.0

    # hat height based on asset aspect ratio
    aspect = h_hat / w_hat
    hat_height_px = hat_width_px * aspect

    # direction vectors
    right_dir = temple_vec / np.linalg.norm(temple_vec)  # horizontal axis
    up_dir = np.array([right_dir[1], -right_dir[0]])      # perpendicular (up)

    # midpoint between temples, shifted up to sit on top of head
    mid = (left + right) / 2
    hat_offset = up_dir * hat_height_px * 0.85  # nudge up so hat sits above forehead

    # 4 destination corners of the hat on the frame (bottom-left, bottom-right, top-right, top-left)
    half_w = right_dir * hat_width_px / 2
    dst_pts = np.array([
        mid - half_w + hat_offset - up_dir * hat_height_px * 0.05,  # bottom-left
        mid + half_w + hat_offset - up_dir * hat_height_px * 0.05,  # bottom-right
        mid + half_w + hat_offset + up_dir * hat_height_px * 0.95,  # top-right
        mid - half_w + hat_offset + up_dir * hat_height_px * 0.95,  # top-left
    ], dtype=np.float32)

    # 4 source corners of the hat png
    src_pts = np.array([
        [0, h_hat],
        [w_hat, h_hat],
        [w_hat, 0],
        [0, 0],
    ], dtype=np.float32)

    # Warp hat to match head angle
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(hat_img, M, (w_frame, h_frame))

    # Alpha composite using the warped alpha channel
    alpha = warped[:, :, 3:4] / 255.0
    rgb   = warped[:, :, :3]
    frame = (frame * (1 - alpha) + rgb * alpha).astype(np.uint8)

    return frame

hat_img = cv2.imread(HAT_PATH, cv2.IMREAD_UNCHANGED)
if hat_img is None:
    raise FileNotFoundError(f"Could not load hat image at {HAT_PATH}")
if hat_img.shape[2] != 4:
    raise ValueError("Hat PNG must have an alpha (transparency) channel")

# --- Setup ---
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
smoother = LandmarkSmoother(
    min_cutoff=1.0,   # tweak for stillness smoothing
    beta=0.01         # tweak for motion responsiveness
)


with vision.FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        model_renderer = ModelRenderer(MODEL_3D_PATH, frame.shape)  # init after first cap.read()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        results = landmarker.detect(mp_image)

        if results.face_landmarks:
            smoothed = smoother.smooth(results.face_landmarks[0], frame.shape[:2])
            # frame = draw_landmarks(frame, smoothed)
            # frame = draw_overlay_hat(frame, hat_img, smoothed)
            if results.facial_transformation_matrixes:
                pose = np.array(results.facial_transformation_matrixes[0].data).reshape(4, 4)
                frame = model_renderer.render(frame, pose, smoothed, None)

        cv2.imshow("Face Mesh - One Euro Filter", frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()