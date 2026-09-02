# mshow3 Architectural Assessment

Scope: `mshow3.py` (main pipeline) and `mshowgui.py` (experimental DPG shell).
Goal: find where to shift to Data Oriented Design (DOD) so the pipeline can handle different asset types. Criteria: Performance, Scalability, Reliability, Maintainability.

## Pipeline at a glance

```
capture -> [denoise?] -> mediapipe landmarker (async) -> segment? -> stabilizer
        -> bone tree -> GL render -> CPU readback -> composite -> show / UI slot
```

The landmarker is the only async stage (LIVE_STREAM callback writes globals). Everything downstream of it is one serial per-frame chain. The UI shell runs the same chain in a worker thread and blits published frame slots.

## Findings

### Performance
- `GLBRenderer.render()` rebuilds two point-cloud VAOs *and their VBOs* every frame (`set_facemesh`, mshow3.py:995). Points are 478 vertices; the data is genuinely new each frame, but the buffers should be allocated once and updated with `buffer.write()`.
- GPU->CPU readback every frame: `fbo.read` + `np.frombuffer` + `flipud().copy()` (mshow3.py:1165). Readback stalls the GPU pipeline and is the single biggest frame-rate ceiling. This is the load-bearing decision to revisit for product work: composite on GPU (render direct to texture) and never bring the frame to CPU except for display.
- Per-frame numpy allocations in the hot loop: view/proj/model/normal matrices, `composite_frame` alpha/rgb buffers, `pts`/`overlay`/`portrait` arrays. Small but pervasive; they are the exact allocations DOD removes.
- `_load_ghost_head` calls `trimesh.load(path)` twice (mshow3.py:828-829) — one load is wasted on every startup.
- `MatrixStabilizer` is already DOD-shaped: preallocated `out` buffers, no per-frame allocs beyond scipy's fixed-size conversions (documented). Good baseline to copy.

### Scalability
- Serial pipeline, one frame in flight. The landmarker's async overlap is canceled by the synchronous readback + composite that follows.
- Heavy optional stages (denoise, segmentation) run inline on the main thread when toggled; there is no frame-skip, priority, or decoupling, so enabling them directly costs frames.
- In the GUI, `state.publish` stores full-resolution copies of several views per frame (ar, fbo, denoise pair) — unbounded memory growth if views accumulate. A ring/double buffer is the fix.
- The async callback globals (`LANDMARKER_RESULT`, `LANDMARKER_RESULT_TS`) are written by MediaPipe's internal thread and read without any sync in the main loop — data path is implicit, so adding stages means threading more ad-hoc state.

### Reliability
- Race/tearing risk: callback writes `LANDMARKER_RESULT` then `LANDMARKER_RESULT_TS` as two statements; the loop reads them as two statements. A callback firing between the reads pairs a new result with an old timestamp. Wrapped-in-one-tuple write would remove it.
- DEBUG mode re-runs the segmenter inside `render()` (mshow3.py:1184) against the global `segmenter`, duplicating the stage that already ran upstream. It is dead weight and a confusing second path.
- Lots of dead/commented code (alternate `get_head_mask`, commented `_load_glb`, commented imshow blocks, `DEBUG` branches). Confuses review and hides what actually runs.
- `_check_portrait_projection` self-check exists and is good; keep that pattern as the pipeline grows.

### Maintainability
- The pipeline body is duplicated between `mshow3.main()` and `mshowgui.pipeline_worker()` (near-identical loop). Two copies already; a third interface will fork a third. The shared flow should be one function that stages run inside.
- Asset data is hardcoded in module constants: `HAT_LOCAL`, `GHOST_LOCAL`, `NECK_TO_FACE_METRIC`, `HOLO_TINT`, `PORTRAIT_*`, light rig, gains. Adding an asset type today means editing code. The GUI already exists to tune parameters — the next step is the parameters becoming data.
- Bone rig is an OOP tree (`Bone`, `build_skeleton`, recursive `world()`). Three bones now, so harmless; it is the pattern to flatten before bones multiply.

## Recommended DOD shifts (in priority order)

1. **Asset descriptor table (data-driven assets).** Replace hardcoded attach constants with a flat table of asset records: `(asset id, glb path, attach bone, local transform, shader/material id, depth behavior, toggle)`. The render loop iterates the table; adding an asset type = adding a row, not editing code. This is the change that directly answers "work with different asset types." The portrait tuning knobs (`PORTRAIT_*`, gains) are the same category of data and should live in the same config layer the GUI edits.

2. **Flat bone buffer.** Store bones as an SoA: one array of 4x4 matrices (or 3x4), a parent-index array, a world-transform buffer. Compute world matrices top-down in one loop. Attachment transforms live in the asset table, not on bones. This kills the recursive class chain and makes the rig data, not code.

3. **Explicit frame data struct instead of globals.** Define one `FrameData` (or SoA ring of frame slots) that moves through stages: capture -> detect -> stabilize -> rig -> render. The callback writes the pair `(result, ts)` atomically into a slot; stages consume it. Also gives the GUI a single publish point and removes the implicit-global coupling.

4. **Stop CPU readback; render GPU-side.** Keep the renderer output as a texture and let the display/compositor path (cv2 window or DPG) consume it. If readback must stay for debug, use persistent buffers + `readinto`, and keep the numpy post (`flipud`/mask/composite) on reused buffers rather than fresh arrays.

5. **Persistent GPU buffers for the point cloud.** Allocate VAO/VBO once, `buffer.write()` per frame. A constant-size allocation profile is the DOD property to enforce across all per-frame work.

## Notes
- Reuse `MatrixStabilizer`'s style (preallocated out-buffers, no per-frame allocs) as the template for every per-frame stage.
- The `bone_tree_debug` / `z_raw_span` debug outputs are separate data passes; keep them behind the `DEBUG` flag as one decoupled observer so the product path never pays for them.
- When the pipeline becomes one shared function (finding 6), keep the mshow3 self-check and add one per new stage (per AGENTS.md).