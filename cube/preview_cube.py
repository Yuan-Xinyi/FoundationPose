#!/usr/bin/env python3
"""Faithful 3D preview of the textured cube, rendered with FoundationPose's own
nvdiffrast pipeline (identical to what register/track see).

Renders corner views (3 faces each) + 6 face-on views, so you can see exactly
which color+letter is on which face and replicate it on the real cube.

Output: cube/mesh/cube_preview.png   (run inside env_isaaclab)
"""
import os
import sys

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Utils import make_mesh_tensors, nvdiffrast_render  # noqa: E402

CODE_DIR = os.path.dirname(os.path.realpath(__file__))
W = H = 480
FX = FY = 1100.0
K = np.array([[FX, 0, W / 2], [0, FY, H / 2], [0, 0, 1]], dtype=np.float64)
DIST = 0.22  # camera distance (m)


def look_at_pose(dir_obj, dist):
    """ob_in_cam (4x4, openCV) placing object origin at `dist` along -dir from cam."""
    d = np.asarray(dir_obj, float)
    d /= np.linalg.norm(d)
    p = d * dist                       # camera position in object space
    z_c = -d                           # camera looks toward origin
    up = np.array([0, 0, 1.0])
    if abs(np.dot(up, z_c)) > 0.95:    # avoid degeneracy for top/bottom views
        up = np.array([0, 1.0, 0])
    x_c = np.cross(up, z_c); x_c /= np.linalg.norm(x_c)
    y_c = np.cross(z_c, x_c)
    Rwc = np.stack([x_c, y_c, z_c], axis=0)
    t = -Rwc @ p
    T = np.eye(4); T[:3, :3] = Rwc; T[:3, 3] = t
    return T


def render(mesh_tensors, dir_obj):
    pose = torch.as_tensor(look_at_pose(dir_obj, DIST)[None], device="cuda", dtype=torch.float)
    color, depth, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=pose,
                                        mesh_tensors=mesh_tensors, use_light=False)
    rgb = (color[0].clip(0, 1).cpu().numpy() * 255).astype(np.uint8)
    mask = depth[0].cpu().numpy() > 0
    out = np.full((H, W, 3), 245, np.uint8)   # light background
    out[mask] = rgb[mask]
    return out


def label(img, text):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except Exception:
        f = ImageFont.load_default()
    d.rectangle([0, 0, im.width, 30], fill=(30, 30, 30))
    d.text((8, 4), text, fill=(255, 255, 255), font=f)
    return np.array(im)


def main():
    mesh = trimesh.load(os.path.join(CODE_DIR, "mesh", "textured.obj"))
    mt = make_mesh_tensors(mesh)

    views = [
        ((1, 1, 1), "corner: +X(A) +Y(C) +Z(E)"),
        ((-1, -1, -1), "corner: -X(B) -Y(D) -Z(F)"),
        ((1, 0, 0), "+X  red  A"),
        ((-1, 0, 0), "-X  green  B"),
        ((0, 1, 0), "+Y  blue  C"),
        ((0, -1, 0), "-Y  yellow  D"),
        ((0, 0, 1), "+Z  magenta  E"),
        ((0, 0, -1), "-Z  cyan  F"),
    ]
    tiles = [label(render(mt, d), txt) for d, txt in views]

    # arrange: 2 corner views on top row, 6 face views in a 2x3 block below
    def hstack(imgs, gap=6):
        h = imgs[0].shape[0]
        canvas = [np.full((h, gap, 3), 255, np.uint8)]
        out = []
        for im in imgs:
            out.append(im); out.append(np.full((h, gap, 3), 255, np.uint8))
        return np.hstack(out[:-1])

    row0 = hstack(tiles[0:2])
    row1 = hstack(tiles[2:5])
    row2 = hstack(tiles[5:8])
    # pad rows to same width
    wmax = max(row0.shape[1], row1.shape[1], row2.shape[1])
    def pad(r):
        if r.shape[1] < wmax:
            r = np.hstack([r, np.full((r.shape[0], wmax - r.shape[1], 3), 255, np.uint8)])
        return r
    gap = np.full((8, wmax, 3), 255, np.uint8)
    full = np.vstack([pad(row0), gap, pad(row1), gap, pad(row2)])

    out_path = os.path.join(CODE_DIR, "mesh", "cube_preview.png")
    Image.fromarray(full).save(out_path)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
