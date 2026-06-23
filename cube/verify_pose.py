#!/usr/bin/env python3
"""Verify estimated poses by re-projecting the textured mesh onto the RGB frames.

For each frame it renders the mesh (via FoundationPose's own nvdiffrast pipeline)
at the estimated ob_in_cam pose and alpha-blends it over the photo. If the
rendered colors/letters land exactly on the real cube's faces with no ghosting,
the pose (translation AND rotation) is correct.

Reads poses from <debug_dir>/ob_in_cam/*.txt (written by run_demo.py).
Writes blended overlays to <debug_dir>/overlay/*.png (full frame + zoomed crop).

Run inside env_isaaclab, after run_demo.py.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Utils import make_mesh_tensors, nvdiffrast_render  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    code_dir = os.path.dirname(os.path.realpath(__file__))
    repo = os.path.dirname(code_dir)
    ap.add_argument("--scene_dir", default=code_dir)
    ap.add_argument("--mesh_file", default=os.path.join(code_dir, "mesh", "textured.obj"))
    ap.add_argument("--debug_dir", default=os.path.join(repo, "debug"))
    ap.add_argument("--alpha", type=float, default=0.55, help="render blend weight")
    args = ap.parse_args()

    K = np.loadtxt(os.path.join(args.scene_dir, "cam_K.txt")).reshape(3, 3)
    mesh = trimesh.load(args.mesh_file)
    mt = make_mesh_tensors(mesh)
    out_dir = os.path.join(args.debug_dir, "overlay")
    os.makedirs(out_dir, exist_ok=True)

    pose_files = sorted(glob.glob(os.path.join(args.debug_dir, "ob_in_cam", "*.txt")))
    if not pose_files:
        raise SystemExit(f"No poses in {args.debug_dir}/ob_in_cam. Run run_demo.py first.")

    for pf in pose_files:
        sid = os.path.splitext(os.path.basename(pf))[0]
        rgb = cv2.imread(os.path.join(args.scene_dir, "rgb", f"{sid}.png"))
        if rgb is None:
            continue
        H, W = rgb.shape[:2]
        T = torch.as_tensor(np.loadtxt(pf)[None], device="cuda", dtype=torch.float)
        color, depth, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=T,
                                            mesh_tensors=mt, use_light=False)
        ren = (color[0].clip(0, 1).cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
        m = depth[0].cpu().numpy() > 0
        out = rgb.copy()
        out[m] = (args.alpha * ren[m] + (1 - args.alpha) * rgb[m]).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"{sid}.png"), out)
        if m.any():
            ys, xs = np.where(m)
            cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
            crop = out[max(0, cy - 80):cy + 80, max(0, cx - 100):cx + 100]
            crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(out_dir, f"{sid}_zoom.png"), crop)

    print(f"wrote {len(pose_files)} overlays to {out_dir}")
    print("Check that rendered colors/letters sit exactly on the real cube (no ghosting).")


if __name__ == "__main__":
    main()
