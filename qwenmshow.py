"""
Face wireframe overlay: MediaPipe FaceLandmarker + OpenGL (GLSL).

Camera frames are captured with OpenCV, landmark detection runs with
MediaPipe's FaceLandmarker, and both the video and the face mesh wireframe
are drawn with GLSL shaders rendered through moderngl on a Pygame window.

Dependencies (pip):
    pip install opencv-python mediapipe moderngl pygame numpy

Model:
    Download the FaceLandmarker model and point MODEL_PATH at it:
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
"""

import os

# Request an OpenGL 3.3 core-profile context (needed for the #version 330
# shaders below). Must be set before pygame creates its window.
os.environ.setdefault("SDL_GL_CONTEXT_MAJOR_VERSION", "3")
os.environ.setdefault("SDL_GL_CONTEXT_MINOR_VERSION", "3")
os.environ.setdefault("SDL_GL_CONTEXT_FLAGS", "1")           # forward compatible
os.environ.setdefault("SDL_GL_CONTEXT_PROFILE_MASK", "1")    # core profile

import cv2
import numpy as np
import pygame
import moderngl
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
# MediaPipe ships the face-mesh topology, so we do not have to hard-code
# any facial structure ourselves - the tessellation tells us every edge.
from mediapipe. import FACIALLANDMARKS_TESSELATION

MODEL_PATH = "face_landmarker.task"   # <-- download and place it here
CAM_INDEX = 0
WIDTH, HEIGHT = 1280, 720
FPS = 60

# --------------------------------------------------------------------------
# GLSL
# --------------------------------------------------------------------------
QUAD_VS = """
#version 330
in vec2 in_position;
in vec2 in_texcoord;
out vec2 v_uv;
void main() {
    v_uv = in_texcoord;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

QUAD_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D u_texture;
out vec4 frag_color;
void main() {
    frag_color = texture(u_texture, v_uv);
}
"""

LINE_VS = """
#version 330
in vec2 in_position;
in float v_index;
uniform float u_time;
out vec4 v_color;

void main() {
    // Gentle travelling glow along each edge + a slow global pulse.
    float pulse = 0.75 + 0.25 * sin(u_time * 4.0 + v_index * 0.05);
    v_color = vec4(0.10 * pulse, 0.85 * pulse, 1.0 * pulse, 0.95);
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

LINE_FS = """
#version 330
in vec4 v_color;
out vec4 frag_color;
void main() {
    frag_color = v_color;
}
"""


def unique_edges(tessellation):
    """Turn the face-mesh triangles into a deduplicated edge list."""
    edges = set()
    for tri in tessellation:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edges.add((a, b) if a < b else (b, a))
    return list(edges)


def main():
    edges = unique_edges(FACIALLANDMARKS_TESSELATION)
    num_edges = len(edges)
    print(f"Face mesh: {len(FACIALLANDMARKS_TESSELATION)} triangles, "
          f"{num_edges} unique edges")

    # ---------------- MediaPipe ----------------
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        num_faces=1,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    # ---------------- Pygame window + GL context ----------------
    pygame.init()
    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF
    )
    pygame.display.set_caption("MediaPipe FaceLandmarker + GLSL wireframe")

    ctx = moderngl.use_context()  # adopt the context Pygame created

    quad_prog = ctx.program([
        ctx.vertex_shader(QUAD_VS, verify=False),
        ctx.fragment_shader(QUAD_FS, verify=False),
    ])
    line_prog = ctx.program([
        ctx.vertex_shader(LINE_VS, verify=False),
        ctx.fragment_shader(LINE_FS, verify=False),
    ])

    quad_vao = ctx.fullscreen_vertex_arrays()

    # Each edge = one line segment = 2 vertices (position.xy + per-edge
    # index used for the shader animation).
    line_data = np.zeros((num_edges * 2, 3), dtype=np.float32)
    line_vbo = ctx.vertex_buffer(line_data, dynamic=True)
    line_vao = ctx.vertex_arrays(line_vbo, "2f 1f")

    texture = ctx.texture((WIDTH, HEIGHT), 3)  # RGB8

    # ---------------- Main loop ----------------
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAM_INDEX}")

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        ok, frame = cap.read()
        if not ok:
            break

        # Mirror the webcam (selfie view). We flip once, up front, so the
        # landmarks (computed on the same image) stay in sync with the
        # pixels we upload to the texture.
        frame = cv2.resize(cv2.flip(frame, 1), (WIDTH, HEIGHT),
                           interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- detect ---
        result = face_landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        )

        # --- render ---
        ctx.clear(0.0, 0.0, 0.0, 1.0)

        texture.write(frame_rgb.tobytes())
        quad_prog["u_texture"].value = texture
        texture.use(0)
        quad_vao.render(moderngl.TRIANGLES)

        if result.face_landmarks:
            lms = result.face_landmarks[0]
            # Normalized image coords (x right, y down) -> clip space.
            for i, (a, b) in enumerate(edges):
                pa = (2.0 * lms[a].x - 1.0, 1.0 - 2.0 * lms[a].y)
                pb = (2.0 * lms[b].x - 1.0, 1.0 - 2.0 * lms[b].y)
                line_data[i * 2]     = (*pa, float(i))
                line_data[i * 2 + 1] = (*pb, float(i))
            line_vbo.write(line_data.tobytes())

            ctx.enable(moderngl.BLEND)
            ctx.blend_func(moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            line_prog["u_time"].write(np.float32(pygame.time.get_ticks() / 1000.0))
            line_vao.render(moderngl.LINES, vertices=num_edges * 2)
            ctx.disable(moderngl.BLEND)

        pygame.display.flip()
        pygame.display.set_caption(
            f"Face wireframe - {clock.get_fps():.0f} FPS  (ESC to quit)"
        )
        clock.tick(FPS)

    cap.release()
    face_landmarker.close()
    pygame.quit()


if __name__ == "__main__":
    main()