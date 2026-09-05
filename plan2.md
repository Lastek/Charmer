# mshow3 Product Plan — DOD Migration (revision 2)

Revision 1 was reviewed against `mshow3.py`/`mshowgui.py` and the `media`
env. The core rendering premise (moderngl on hello_imgui's context, ImGui
drawing the AR texture by GL id, zero readback) was the single riskiest
assumption. It is now validated by a thrown spike, not argued from docs.

## Validated (spike, ran in `media` env)

| Result | Finding |
|---|---|
| Context binding | `moderngl.create_context()` (detect mode) binds to hello_imgui's GL context. Not a separate standalone context. |
| Framebuffer | VAO/program render into an FBO, `fbo.read` returns correct pixels in that context. |
| Texture namespace | ImGui drew a moderngl texture by integer GL id into the window — `65536` pixels matched (256x256). Same context, same texture id namespace. |
| Orientation | With default UVs the texture lands vertically flipped; `uv0=(0,1), uv1=(1,0)` restores top-down (`np.flipud` dies here). |

Hard requirements surfaced by the spike, now promoted to plan decisions:

1. Force `renderer_backend_type = open_gl3`. hello_imgui's default
   (`first_available`) may pick DirectX on Windows, which has no GL context
   at all — moderngl then has nothing to bind.
2. Request a GL **3.3 core** profile (`OpenGlOptions(major=3, minor=3,
   use_core_profile=True)`). hello_imgui defaults to 3.2; moderngl requires >=3.3.
3. imgui_bundle 1.92 wraps the id: `imgui.image(ImTextureRef(tex.glo),
   ImVec2(w, h), uv0=(0,1), uv1=(1,0))` — a raw `int`/tuple raises.
4. The GL render happens in `custom_background` (window framebuffer current);
   ImGui controls + `image()` in `show_gui`. moderngl context is created once
   in `post_init`.

## Decisions locked (unchanged unless noted)

| Topic | Decision |
|---|---|
| Layout | `mshow3.py` headless core, `mshowgui.py` UI shell |
| cv2 | Remove all UI code; keep `VideoCapture`. |
| GUI | `imgui_bundle` (hello_imgui). dearpygui removed. **Requires GL 3.3 core + `open_gl3`, see above.** |
| Config | JSON beside assets; data-driven table + knobs |
| Rendering | moderngl bound to the shared hello_imgui context; AR renders to a texture; ImGui draws it by GL id. Zero readback. |
| Threading | GL work (AR render + ImGui) on main thread; CPU stages (capture, detect, segment, denoise) on worker. MediaPipe detect stays async. |
| Context injection | `GLBRenderer.__init__` takes an *injected* moderngl context. It creates a standalone one only for headless checks. |
| V1 milestone | One asset type (hat) on the descriptor table, end-to-end |
| Output | Live preview only, no recording |

## Why the rendering change matters

Dear ImGui draws a raw GL texture by id; dearpygui cannot. Removing the
CPU round-trip (`composite_frame`, `fbo.read`, `np.flipud`) is the whole
DOD payoff. The spike proved the pieces; the reasoning now stands on
evidence: same GL context → same texture namespace → ImGui shows the
render without `fbo.read`.

A consequence that revision 1 called out and the spike confirms: the GL
context is thread-bound, so AR render moves to the main thread beside
ImGui. The worker only touches numpy.

## Target architecture

```
Worker thread (CPU only):
  capture -> [denoise?] -> mediapipe detect (async) -> [segment?] -> publish to slots

Main thread (GL + UI):
  read newest slot -> stabilize -> rig -> render AR (moderngl, custom_background)
    -> ImGui draws AR texture by id + controls (show_gui)

mshow3.py (headless core, no GUI imports)
  config.json ──> load_config() ──> PipelineConfig (defaults merged over JSON)
  Rig          : flat bone table (name, parent index) + world-transform buffer, top-down loop
  AssetTable   : descriptor rows (id, glb, attach bone, local 4x4, material, depth, enabled)
                 -> one (vao, prog, index_count) per asset at load; renderer iterates the table
  DetectorSink : async sink object; callback writes (result, ts) in one slot write
  GLBRenderer  : constructor takes the shared moderngl context (standalone only for headless checks)
  main()       : headless smoke/benchmark entry + self-checks

mshowgui.py (imgui_bundle shell)
  imports mshow3 core only; no duplicated pipeline logic
  worker thread  : CPU stages -> SharedState slots
  main thread    : post_init builds the compiler context, custom_background renders AR,
                   show_gui draws the texture + controls, "Reload config"
```

Timing caveat carried from the old code: `MatrixStabilizer` lives on the
main thread now, but its OneEuro `dt` must still come from *detection*
timestamps flowing through the slot, not the main thread's wall clock.
Otherwise the filter constants silently change meaning.

## Config schema (`config.json`, beside assets)

Same shape as revision 1 with two fixes (see notes below the block).

```json
{
  "camera": { "index": 0 },
  "models": { "face_landmarker": "face_landmarker.task", "segmentation": "selfie_multiclass_256x256.tflite" },
  "stabilizer": { "min_cutoff": 0.35, "beta": 0.05, "gate_cm": 8.0, "gate_deg": 25.0, "stale_ms": 500 },
  "rig": [ {"name": "root", "parent": null}, {"name": "neck", "parent": 0}, {"name": "head", "parent": 1} ],
  "assets": [
    { "id": "hat", "glb": "Pirate hat.glb", "attach_bone": "head",
      "local": { "translate": [0, -13, 4], "scale": 1.3, "flip_y": true, "rotate_deg": [210, "y"] },
      "material": "standard", "depth": "opaque", "enabled": true },
    { "id": "ghost_head", "glb": "head.glb", "attach_bone": "head",
      "local": { "scale": 4.2, "translate": [0, -4, 4] },
      "material": "ghost", "depth": "occluder", "enabled": true }
  ],
  "portrait": { "yaw_gain": 3.0, "pitch_gain": 0.5, "roll_gain": 1.0, "neck_y": 0.55,
                "neck_z": 0.0, "tint": [0.3, 0.8, 1.0], "glow_strength": 4.7 },
  "display": { "show_facemesh": true, "show_hologram": true, "show_axes": false, "ghost_shaded": false, "glow": true }
}
```

Notes on the schema:

- **`local` build order is now pinned.** Today's hat is
  `T(0,-13,4) @ diag(1.3,1.3,1.3)`, then `local[1,1] = -1`, then `@ R_y(210°)`.
  The config must reproduce exactly `local = T(translate) @ S(scale with Y
  negated if flip_y) @ R(rotate_deg)`. `flip_y` negates the scale's *Y
  component before rotation*, not a post-rotation reflection — that ordering
  is non-obvious and must be documented in `load_config`.
- `portrait.glow_strength: 4.7` matches the code's `GLOW_STRENGTH` (bloom
  add-back). Revision 1 wrote `0.55`, which was a different, unbacked value.
- Not in the schema (V1 defaults, not knobs): `DEADBAND_CM/DEG`,
  `TRACK_RATE_*`, `HOLD_FRAMES`, `DECAY_RATE`, `NEUTRAL_T`, and the ghost
  light rig (`GHOST_LIGHTS_POS/COLOR`, `GHOST_ALPHA`). Defaults-merge covers
  them; promote to knobs only if they need live tuning later (YAGNI).

## Phases

### Phase 0 — Config layer + dead code removal (headless, safe)
- Add `load_config(path=None)` + `PipelineConfig` (dict + defaults merge).
- Delete dead code: `draw_facemesh`, `LandmarkSmoother`, duplicate
  `trimesh.load` in `_load_ghost_head` (`mshow3.py:891-892`), commented-out
  blocks, the DEBUG re-segment inside `render()` (`mshow3.py:1246-1256`), the
  `segmenter` global.
- Remove cv2 UI: `imshow`, `waitKey`, `destroyAllWindows`, the debug imshow
  branches in `main()`.
- Check: `load_config` builds the expected 4x4 for `flip_y`+`rotate_deg`,
  and boots with defaults when the file is absent.

### Phase 1 — Data-driven skeleton (headless, safe)
- Replace `Bone`/`build_skeleton`/`world()` with flat `Rig`:
  `names`, `parent_idx`, local transforms as a numpy array; `rig.world()`
  fills a preallocated world buffer top-down in one loop.
- `bone_tree_debug` reads the world buffer (GUI debug text keeps working).
- Check: 3-bone world matrices match the old recursive `world()` for random
  locals (`test_rig.py` #2/#3 carry over).

### Phase 2 — Asset table (V1: hat on the table)
- Move `HAT_LOCAL`/`GHOST_LOCAL` into `load_config`; renderer builds VAOs
  from the table. `for asset in table: model = bone_world[attach] @ asset.local`.
- `material` -> shader (standard | ghost); `depth` -> draw flags (occluder =
  color mask off, unchanged).
- Check: assert the computed `model` matrix equals the old constant
  (`test_rig.py` #3), plus a visual A/B against the old binary (still using
  the standalone context at this phase).

### Phase 3 — Frame data, not globals
- Add `DetectorSink`: `__call__(result, image, ts)` stores `(result, ts)` in
  one tuple write (removes the result/ts tear that `LANDMARKER_RESULT` +
  `LANDMARKER_RESULT_TS` have today, `mshow3.py:1264-1268`).
- Pipeline owns the sink; delete the two module globals.
- Check: two sequential writes never pair a stale result with a fresh ts.

### Phase 4 — Shared-context rendering + hello_imgui shell (one atomic phase)
Revision 1 split these; they cannot be separate. Deleting the standalone
context (`mshowgui.py:76-146`) leaves no window for Phase 2's A/B, so the
GUI swap and the context swap land together.

- `GLBRenderer` takes an injected moderngl context; no `create_standalone_context`.
- hello_imgui shell: `post_init` creates the context (GL 3.3 core,
  `open_gl3`); `custom_background` renders; `show_gui` draws
  `imgui.image(ImTextureRef(tex.glo), ..., uv0=(0,1), uv1=(1,0))` + controls.
- Camera frame uploaded as the fullscreen background texture; assets over it
  with depth, as today. Portrait FBO + bloom stay GPU-side.
- Segmentation mask becomes a texture upload (a write, not a readback) and
  is applied as alpha/stencil in the composite shader.
- Delete `composite_frame`, the mask numpy math, `fbo.read`, `np.flipud`.
- Controls: facemesh/axes/ghost/denoise/head-mask toggles, asset enable,
  "Reload config". Debug views (FBO/Denoise) as ImGui child windows.
- Check: screenshot the same frame from the old standalone render and the
  new shared-context render and diff (coordinate conversion is the risk);
  profile confirms zero `fbo.read` in the live loop.

### Phase 5 — Validation
- Headless smoke entry in `mshow3.py`: N frames with the camera, assert
  frame dimensions and rough FPS, no display.
- All self-checks pass (`_check_portrait_projection`, config, rig, sink).
- Manual GUI pass on real hardware.

## Out of scope
- Multi-threaded denoise/segment stages (flag-gated, on the worker as-is).
- Recording / file output.
- More than the two descriptor rows above.