# Cube pose with FoundationPose + D405

Get the 6D pose of a 6 cm cube from a RealSense **D405**, using FoundationPose
(model-based). Scripts live in this folder.

## The cube (must match mesh ↔ real object)

A plain single-color cube is rotationally symmetric → the pose is ambiguous and
jumps around. So each face gets a **distinct high-contrast color + a letter**
(the letter breaks the 90° in-plane ambiguity of a solid face). Build your real
6 cm cube to match these references:

- `mesh/sticker_sheet_A4.pdf` — **print this** (A4 landscape, each tile exactly 60 mm).
  Print at 100% / Actual size (no fit-to-page); verify the 60 mm ruler with a real ruler.
- `mesh/sticker_sheet.png` — same tiles as a plain image (no fixed scale).
- `mesh/cube_preview.png`  — faithful 3D render (same nvdiffrast pipeline as inference).
- `mesh/cube_net.png`      — unfolded layout.

| face        | color   | letter | RGB             |
|-------------|---------|--------|-----------------|
| +X (right)  | red     | A      | (220, 20, 20)   |
| -X (left)   | green   | B      | (20, 180, 60)   |
| +Y (back)   | blue    | C      | (30, 70, 230)   |
| -Y (front)  | yellow  | D      | (245, 220, 20)  |
| +Z (top)    | cyan    | F      | (20, 200, 220)  |
| -Z (bottom) | magenta | E      | (225, 30, 200)  |

**Two ways to apply it:**
- Fold-up net: print `mesh/cube_net_A4.pdf`, cut the cross, fold, wrap the cube.
  Zero orientation decisions. (The +Z magenta E looks backwards in the flat net on
  purpose — it becomes upright once folded into the top face.)
- Individual stickers + this rule for the letter's "up" direction:
  the 4 side faces (red/green/blue/yellow) → letter top points to the **cyan (+Z)** face;
  the top & bottom (cyan/magenta) → letter top points to the **blue (+Y)** face.

`mesh/cube_net_guide.png` shows the whole layout; `mesh/cube_preview.png` shows
each face on its own.

The mesh defines the object coordinate frame; the pose FoundationPose outputs is
this frame's pose in the camera. Real cube appearance must match the mesh or the
estimated frame is meaningless.

## Environments

Use **`env_isaaclab`** for everything except capture. Verified: torch 2.7.0+cu128
(CUDA on RTX 4090), nvdiffrast.torch, trimesh, cv2, open3d, scipy, PIL; the repo's
`mycpp` native ext is built and `weights/` are present, so `run_demo.py` runs.
NOTE: you must fully `conda activate env_isaaclab` (calling the python binary
directly does NOT load torch — the activation script sets the paths).

| step                         | conda env        | why                                |
|------------------------------|------------------|------------------------------------|
| `make_cube_mesh.py`          | `env_isaaclab`   | trimesh + PIL                      |
| `capture_d405.py`            | `env_isaaclab`   | pyrealsense2 (installed 2.58.2)    |
| `make_mask.py`               | `env_isaaclab`   | cv2                                |
| `run_demo.py` (inference)    | `env_isaaclab`   | full torch/nvdiffrast stack + weights |

## Pipeline

```bash
# 1. Mesh — 6 cm, 6 colored+lettered faces. Verify extents print [0.06 0.06 0.06].
conda activate env_isaaclab
python cube/make_cube_mesh.py --edge 0.06
python cube/preview_cube.py          # optional: faithful 3D render to check the result
python cube/print_stickers.py        # A4-landscape PDF, tiles exactly 60 mm, to print

# 2. Capture RGB-D from the D405 (depth auto-converted to uint16 millimeters,
#    aligned to color, cam_K.txt written from color intrinsics).
#    GUI: SPACE=save, c=continuous, q=quit.   Start with a single frame to test.
python cube/capture_d405.py   # same env_isaaclab
#    headless alternative: python cube/capture_d405.py --num 100 --interval 0.1

# 3. First-frame mask (only the first frame needs one).
conda activate env_isaaclab
python cube/make_mask.py            # draw a box around the cube, GrabCut refines

# 4. Inference.
python run_demo.py \
  --mesh_file  /disk2/FoundationPose/cube/mesh/textured.obj \
  --test_scene_dir /disk2/FoundationPose/cube \
  --debug 2

# 5. Verify the pose (gold standard): re-project the mesh onto the frames.
python cube/verify_pose.py           # writes debug/overlay/*.png
```

## Live demo (real-time tracking + visualization)

```bash
conda activate env_isaaclab
python cube/live_demo.py
```
Drag a box around the cube on the first frame (ENTER) to register, then it tracks
the live D405 stream: the real frame shows the 3D box + XYZ axes on the cube, the
top-right corner shows a virtual cube at the recognized orientation, and the
bottom shows the xyz translation. Keys: `r` re-init, `q`/ESC quit.

## Minimal validation (do this before a long sequence)

1. `make_cube_mesh.py` prints `extents [0.06 0.06 0.06]` → units are meters. ✓
2. Capture **one** frame, make its mask, run `run_demo.py --debug 2`.
   Open `debug/track_vis/00000.png`: the drawn 3D box + XYZ axes must sit on the
   cube with the right orientation. If the axes are tilted/offset, the pose is off.
   For a stricter check run `verify_pose.py` and open `debug/overlay/00000_zoom.png`:
   the re-projected mesh colors/letters must land exactly on the real faces (no ghosting).
   Also sanity-check the `register() sorted scores` in the log have a clear leader —
   if they are all nearly equal, the object looks symmetric and orientation is unreliable.
3. Only then capture the full sequence and run tracking.

## Directory layout

```
cube/
├── mesh/
│   ├── textured.obj / material.mtl / material_0.png   # CAD model (meters) <- --mesh_file
│   ├── sticker_sheet_A4.pdf # individual stickers: A4 landscape, tiles exactly 60 mm
│   ├── cube_net_A4.pdf      # fold-up net: A4 landscape, cells 60 mm (print+fold+wrap)
│   ├── cube_net_guide.png   # net layout preview (E looks backwards on purpose)
│   ├── sticker_sheet.png   # individual tiles as a plain image
│   ├── cube_preview.png    # faithful 3D render, each face on its own
│   └── cube_net.png        # simple unfolded reference
├── rgb/    00000.png ...           # filled by capture_d405.py
├── depth/  00000.png ...           # uint16, millimeters
├── masks/  00000.png               # first-frame mask (make_mask.py)
├── cam_K.txt                       # 3x3 color intrinsics
├── make_cube_mesh.py
├── preview_cube.py
├── print_stickers.py
├── make_net_guide.py
├── capture_d405.py
├── make_mask.py
├── verify_pose.py
└── live_demo.py
```

## Gotchas (in order of likelihood)

1. **Symmetry** — real cube must have the 6 distinct colors+letters above, matching
   the mesh. (A plain black cube → all pose scores nearly equal → random rotation.)
2. **Depth units** — handled: D405 reports `depth_scale = 0.0001` (0.1 mm/unit);
   `capture_d405.py` converts raw → meters → uint16 mm. Don't double-convert.
3. **Mesh units** — `textured.obj` is in meters (edge 0.06). Verify with the
   extents print.
