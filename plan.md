# mshow3 Product Plan — DOD Migration

## Decisions locked
| Topic | Decision |
|---|---|
| Layout | 2 files: `mshow3.py` = headless core, `mshowgui.py` = UI shell |
| cv2 | Remove all UI code (`imshow`, `waitKey`, `destroyAllWindows`). Keep `VideoCapture` feed. |
| GUI framework | `imgui_bundle` (hello_imgui shell). dearpygui is thrown away. |
| Config | JSON beside assets; data-driven asset table + tuning knobs |
| Rendering | moderngl bound to the shared GLFW/hello_imgui window context (not standalone). AR scene renders to a texture in that context; ImGui draws it by GL id. **Zero readback.** |
| Threading | GL work (AR render + ImGui) on the main thread. CPU stages (capture, detect, segment, denoise) on a worker thread. MediaPipe detect stays async. |
| V1 milestone | One asset type (hat) on the descriptor table, end-to-end |
| Output | Live preview only, no recording |

## Why the rendering change matters
Dear ImGui can draw a raw OpenGL texture by ID, which dearpygui could not. So the composite never has to come back to the CPU:
- GLFW provides the window and its GL context (it does not render).
- moderngl binds to that same context via `moderngl.create_context()` while the window context is current, instead of `create_standalone_context()`.
- moderngl's VBOs/VAOs/textures then live in the shared context; ImGui displays the AR texture through its `.glo` id. No `fbo.read`, no numpy composite, no texture re-upload.
- Consequence: GL contexts are thread-bound, so the AR render moves to the main thread beside ImGui.

## Target architecture

```
Worker thread (CPU only):
  capture -> [denoise?] -> mediapipe detect (async) -> [segment?] -> publish frames+detections to slots

Main thread (GL + UI):
  read newest slot -> stabilize -> rig -> render AR (moderngl, shared context) -> ImGui presents
    AR texture by GL id, UI overlays drawn on top

mshow3.py (headless core, no GUI imports)
  config.json ──> load_config() ──> PipelineConfig (defaults merged over JSON)
  Rig          : flat bone table (name, parent index) + world-transform buffer, top-down loop
  AssetTable   : descriptor rows (id, glb, attach bone, local 4x4, material, depth, enabled)
                 -> one (vao, prog, index_count) per asset at load; renderer iterates the table
  Detector     : async sink object (not globals); callback writes (result, ts) atomically
  GLBRenderer  : constructor takes the shared moderngl context (no standalone context)
  main()       : headless smoke/benchmark entry + self-checks

mshowgui.py (imgui_bundle shell)
  imports mshow3 core only; no duplicated pipeline logic
  worker thread  : CPU stages -> SharedState slots
  main thread    : hello_imgui loop + AR render + controls (toggles, "Reload config")
```

## Config schema (`config.json`, beside assets)

```json
{
  "camera": { "index": 0 },
  "models": { "face_landmarker": "face_landmarker.task", "segmentation": "selfie_multiclass_256x256.tflite" },
  "stabilizer": { "min_cutoff": 0.35, "beta": 0.05, "gate_cm": 8.0, "gate_deg": 25.0, "stale_ms": 500 },
  "rig": [ {"name": "root", "parent": null}, {"name": "neck", "parent": 0}, {"name": "head", "parent": 1} ],
  "assets": [
    { "id": "hat", "glb": "Pirate hat.glb", "attach_bone": "head",
      "local": { "translate": [0, -13, 4], "scale": 1.3, "flip_y": true, "rotate": [210, "y"] },
      "material": "standard", "depth": "opaque", "enabled": true },
    { "id": "ghost_head", "glb": "head.glb", "attach_bone": "head",
      "local": { "scale": 4.2, "translate": [0, -4, 4] },
      "material": "ghost", "depth": "occluder", "enabled": true }
  ],
  "portrait": { "yaw_gain": 5.0, "pitch_gain": 0.5, "roll_gain": 1.0, "neck_y": 0.55,
                "neck_z": 0.0, "tint": [0.3, 0.8, 1.0], "glow_strength": 0.55 },
  "display": { "show_facemesh": true, "show_hologram": true, "show_axes": false, "ghost_shaded": true, "glow": true }
}
```

The `local` block is the DOD payoff: any transform — including the `flip_y` hack currently buried in `HAT_LOCAL` — becomes data. `load_config()` builds real 4x4s once; missing keys fall back to current defaults so the app boots with no config present.

## Phases

### Phase 0 — Config layer + dead code removal
- Add `load_config(path=None)` and `PipelineConfig` (dict + defaults merge).
- Delete dead code: `draw_facemesh`, `LandmarkSmoother`, duplicate `trimesh.load` in `_load_ghost_head`, all commented-out blocks, the DEBUG re-segment inside `render()`, the `segmenter` global.
- Remove cv2 UI: `cv2.imshow`, `cv2.waitKey`, `cv2.destroyAllWindows`, and the imshow debug branches in `main()`.
- Check: config self-test — given a JSON, `load_config` returns the expected 4x4 for `flip_y`+`rotate`, and boots with defaults when the file is missing.

### Phase 1 — Data-driven skeleton
- Replace `Bone`/`build_skeleton`/`world()` with a flat `Rig`: `names`, `parent_idx`, local transforms as a numpy array; `rig.world()` fills a preallocated world buffer top-down in one loop.
- `bone_tree_debug` reads the world buffer (kept for the GUI debug text).
- Check: 3-bone rig world matrices match the old recursive `world()` for a set of random locals.

### Phase 2 — Asset table (V1 milestone: hat on the table)
- Move `HAT_LOCAL` and `GHOST_LOCAL` construction into `load_config`; renderer builds VAOs from the table.
- Render loop: `for asset in table: model = bone_world[attach] @ asset.local; draw`.
- `material` maps to shader (standard | ghost); `depth` maps to draw flags (occluder = color mask off, as today).
- Check: hat renders in the same pose with the table as with the old constant (visual A/B + one assert on the computed `model` matrix).

### Phase 3 — Frame data, not globals
- Add a `DetectorSink` object: `__call__(result, image, ts)` stores `(result, ts)` in one slot write (single tuple assignment — removes the result/ts tearing risk).
- Remove `LANDMARKER_RESULT` / `LANDMARKER_RESULT_TS` globals; pipeline owns the sink and reads the slot.
- Check: unit test that two sequential callback writes never pair a stale result with a fresh ts.

### Phase 4 — Shared-context rendering (zero readback)
- `GLBRenderer` stops creating a standalone context; it takes the window's moderngl context (bound from the GLFW/hello_imgui context on the main thread).
- Camera frame is uploaded as a texture and drawn as the fullscreen background quad; assets render over it with depth as today.
- The portrait FBO and bloom passes stay GPU-side. The final AR texture is drawn by ImGui via its `.glo` id.
- Delete `composite_frame`, the mask numpy math, `fbo.read`, `np.flipud` (handled by UV/texture orientation). No CPU readback at all in the live path.
- Check: same visual result as Phase 2 (A/B), and a profile confirming zero readbacks in the live loop.

### Phase 5 — GUI migration to imgui_bundle
- Replace the dearpygui shell with hello_imgui. Delete `rgba_floats`, dynamic textures, DPG window/view plumbing.
- Thread split: worker thread = capture + detect + segment (+ denoise), publishes CPU frames to slots; main thread = stabilize + rig + AR render + hello_imgui loop.
- Controls: runtime toggles (facemesh, axes, ghost, denoise, head-mask), asset enable checkboxes, "Reload config" (reloads JSON, rebuilds assets/tuning).
- Debug views (FBO/Denoise) become ImGui child windows drawing the shared textures.
- Check: GUI runs, toggles work, config reload applies without restart, FPS stable.

### Phase 6 — Validation
- Headless smoke entry in `mshow3.py`: run N frames with the camera, assert frame dimensions and rough FPS, no display.
- All self-checks pass (`_check_portrait_projection`, config, rig, sink).
- Manual GUI pass on real hardware.

## Out of scope (explicitly)
- Multi-threaded denoise/segment stages (flag-gated, on the worker as-is)
- Recording / file output
- Multiple simultaneous asset types beyond the two descriptor rows above