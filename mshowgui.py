"""Dear PyGui shell for the mshow3 AR pipeline.

The pipeline (capture -> landmarker -> stabilize -> GL render) runs in a
worker thread and publishes its output frames into shared slots. The UI
thread blits the newest slot into a texture per view, every DPG frame.

Run:  conda activate media && python mshowgui.py
"""
import threading
import time
import traceback

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

import VideoDenoiser
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import mshow3 as ms
from mshow3 import (
    DEBUG,
    GLB_PATH,
    HEAD_PATH,
    MODEL_PATH,
    SEGMENTATION_MODEL,
    GLBRenderer,
    HeadSegmenter,
    MatrixStabilizer,
    build_skeleton,
    bone_tree_debug,
    create_mediapipe_image,
)

CAM_INDEX = 0
DENOISE_PANE = (320, 240)  # per-pane size of the Denoise Debug view


# ── Shared state ──────────────────────────────────────────────────
class SharedState:
    """Frame slots + flags shared between pipeline worker and UI thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frames = {}
        self._flags = {"denoise": False, "head_mask": False}
        self.stop = threading.Event()
        self.error = None
        self.renderer = None  # set by worker once GL is up

    def publish(self, **frames):
        with self._lock:
            self._frames.update(frames)

    def get(self, key):
        with self._lock:
            return self._frames.get(key)

    def set_flag(self, key, value):
        with self._lock:
            self._flags[key] = value

    def get_flag(self, key):
        with self._lock:
            return self._flags.get(key)


def rgba_floats(bgr):
    """BGR uint8 HxWx3 -> flat RGBA floats for dpg dynamic textures."""
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
    return np.asarray(rgba, dtype=np.float32).ravel() / 255.0


# ── Pipeline worker ───────────────────────────────────────────────
def pipeline_worker(state):
    """Runs the mshow3 loop. Owns the camera and ALL GL resources:
    the moderngl context is thread-bound, so the renderer must be
    created and used here only."""
    cap = None
    try:
        cap = cv2.VideoCapture(CAM_INDEX)
        ok, cam = cap.read()
        if not ok:
            raise RuntimeError(f"camera {CAM_INDEX} read failed")

        h, w = cam.shape[:2]
        renderer = GLBRenderer(GLB_PATH, HEAD_PATH, w, h)
        state.renderer = renderer
        stabilizer = MatrixStabilizer()
        segmenter = HeadSegmenter(SEGMENTATION_MODEL)
        denoiser = None
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
            result_callback=ms.landmarkerAsyncCallback,
        )
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            frame_count = 0
            while not state.stop.is_set():
                ok, cam = cap.read()
                if not ok:
                    break

                source = cam
                if state.get_flag("denoise"):
                    if denoiser is None:  # lazy: model loads on first use
                        denoiser = VideoDenoiser.VideoDenoiser()
                    source = denoiser.denoise(cam)
                pair = np.hstack([
                    cv2.resize(cam, DENOISE_PANE),
                    cv2.resize(source, DENOISE_PANE),
                ])

                head_mask = (
                    segmenter.get_head_mask(source)
                    if state.get_flag("head_mask") else None
                )

                mp_image = create_mediapipe_image(source)
                timestamp_ms = time.monotonic_ns() // 1_000_000
                landmarker.detect_async(mp_image, timestamp_ms)

                result = ms.LANDMARKER_RESULT
                if result and result.facial_transformation_matrixes:
                    face_matrix = np.array(
                        result.facial_transformation_matrixes[0]).reshape(4, 4)
                    smoothed = stabilizer.stabilize(
                        face_matrix, timestamp_ms, ms.LANDMARKER_RESULT_TS)
                    mx, my, mz = ms.NECK_TO_FACE_METRIC
                    head_bone.local = ms.create_translation_matrix(mx, my, mz).T
                    neck.local = smoothed @ ms.create_translation_matrix(
                        -mx, -my, -mz).T
                    ar = renderer.render(
                        source, head_bone.world(), head_mask,
                        result.face_landmarks[0],
                    )
                    state.publish(ar=ar, fbo=renderer.last_fbo, denoise=pair)

                    frame_count += 1
                    if DEBUG and frame_count % 30 == 0:
                        print(bone_tree_debug(head_bone))
    except Exception:
        state.error = traceback.format_exc()
    finally:
        if cap is not None:
            cap.release()


# ── Views ─────────────────────────────────────────────────────────
class ImageStreamView:
    """A window showing one shared frame slot as an updating texture.
    Closing it frees the texture; reopen from the Views menu."""

    label = "View"
    key = None
    spawn_pos = (60, 60)

    def __init__(self, app):
        self.app = app
        self.window = None
        self.texture = None
        self.closed = False

    def build(self, frame):
        h, w = frame.shape[:2]
        self.texture = f"{self.key}_tex"
        self.window = f"{self.key}_win"
        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(w, h, rgba_floats(frame), tag=self.texture)
        with dpg.window(label=self.label, tag=self.window,
                        pos=self.spawn_pos, on_close=self._on_close):
            dpg.add_image(self.texture)

    def _on_close(self, *args):
        self.closed = True
        self.window = None
        self.texture = None

    def update(self):
        if self.closed:
            return
        frame = self.app.state.get(self.key)
        if frame is None:
            return
        if self.texture is None:  # first frame arrived -> create pane
            self.build(frame)
            return
        dpg.set_value(self.texture, rgba_floats(frame))


class ARView(ImageStreamView):
    label = "AR 3D Model"
    key = "ar"
    spawn_pos = (360, 40)


class FboDebugView(ImageStreamView):
    label = "FBO Debug"
    key = "fbo"
    spawn_pos = (1020, 40)


class DenoiseDebugView(ImageStreamView):
    label = "Denoise Debug"
    key = "denoise"
    spawn_pos = (360, 540)


class ControlsView:
    """Main-window controls + Views menu. A non-image view."""

    def __init__(self, app):
        self.app = app

    def build(self):
        with dpg.menu_bar():
            with dpg.menu(label="Views"):
                for v in self.app.stream_views:
                    dpg.add_menu_item(
                        label=f"Show {v.label}",
                        callback=lambda s, a, v=v: setattr(v, "closed", False),
                    )
        dpg.add_text("Debug layers")
        dpg.add_checkbox(label="Facemesh points", default_value=True,
                         callback=lambda s, a: self._set("show_facemesh", a))
        dpg.add_checkbox(label="TNB axis frames", default_value=DEBUG,
                         callback=lambda s, a: self._set("show_axes", a))
        dpg.add_checkbox(label="Shaded ghost head", default_value=DEBUG,
                         callback=lambda s, a: self._set("ghost_shaded", a))
        dpg.add_separator()
        dpg.add_text("Pipeline")
        dpg.add_checkbox(label="Head-mask occlusion", default_value=False,
                         callback=lambda s, a: self.app.state.set_flag("head_mask", a))
        dpg.add_checkbox(label="Video denoise", default_value=False,
                         callback=lambda s, a: self.app.state.set_flag("denoise", a))

    def _set(self, attr, value):
        renderer = self.app.state.renderer
        if renderer is not None:  # worker may still be starting up
            setattr(renderer, attr, value)


# ── App ───────────────────────────────────────────────────────────
class MShowApp:
    def __init__(self):
        self.state = SharedState()
        self.stream_views = [
            ARView(self), FboDebugView(self), DenoiseDebugView(self),
        ]
        self.controls = ControlsView(self)
        self.main_window = None
        self.worker = None
        self._error_shown = False

    def run(self):
        dpg.create_context()
        dpg.create_viewport(title="mshow3", width=1280, height=800)

        with dpg.window(tag="main_win", label="mshow3", pos=(10, 10),
                        width=300, height=280, no_collapse=True) as self.main_window:
            self.controls.build()

        self.worker = threading.Thread(
            target=pipeline_worker, args=(self.state,), daemon=True)
        self.worker.start()

        dpg.setup_dearpygui()
        dpg.show_viewport()
        while dpg.is_dearpygui_running():   # manual loop: tick every frame
            self.tick()
            dpg.render_dearpygui_frame()
        self.shutdown()

    def tick(self):
        if self.state.error and not self._error_shown:
            self._error_shown = True
            print(self.state.error)
            dpg.stop_dearpygui()
            return
        for v in self.stream_views:
            v.update()

    def shutdown(self):
        self.state.stop.set()
        self.worker.join(timeout=2.0)
        dpg.destroy_context()


if __name__ == "__main__":
    MShowApp().run()
