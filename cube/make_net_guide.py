#!/usr/bin/env python3
"""Fold-up cube net: the 6 faces laid out in a cross, each rendered (faithfully,
via FoundationPose's nvdiffrast) in the orientation it must have so that FOLDING
the net produces exactly the textured.obj cube. Cut the cross out, fold along the
lines, and it wraps the 6 cm cube with every letter correctly placed & oriented.

Net layout (cross), band = -X +Y +X -Y wraps around +Z; +Z on top, -Z on bottom:

              [ +Z / E ]
    [ -X/B ][ +Y/C ][ +X/A ][ -Y/D ]
              [ -Z / F ]

Outputs: cube/mesh/cube_net_guide.png  and  cube/mesh/cube_net_A4.pdf (cells 60 mm).
Run inside env_isaaclab.
"""
import argparse
import os
import sys

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Utils import make_mesh_tensors, nvdiffrast_render  # noqa: E402
from make_cube_mesh import FACE_NAMES, COLOR_NAMES, LETTERS  # noqa: E402

CODE_DIR = os.path.dirname(os.path.realpath(__file__))
RES = 512
FX = 6000.0          # long focal length + far camera => near-orthographic
DIST = 1.2

# face index -> (outward normal, paper-up world dir, (row, col) in a 3x4 grid)
NET = {
    0: ((1, 0, 0),  (0, 0, 1),  (1, 2)),   # +X red A
    1: ((-1, 0, 0), (0, 0, 1),  (1, 0)),   # -X green B
    2: ((0, 1, 0),  (0, 0, 1),  (1, 1)),   # +Y blue C
    3: ((0, -1, 0), (0, 0, 1),  (1, 3)),   # -Y yellow D
    4: ((0, 0, 1),  (0, -1, 0), (0, 1)),   # +Z magenta E  (folds up from +Y)
    5: ((0, 0, -1), (0, 1, 0),  (2, 1)),   # -Z cyan F      (folds down from +Y)
}
NROWS, NCOLS = 3, 4


def pose_from(n, up, dist):
    n = np.asarray(n, float); up = np.asarray(up, float)
    z_c = -n / np.linalg.norm(n)                 # camera looks toward origin
    x_c = np.cross(up, z_c); x_c /= np.linalg.norm(x_c)
    y_c = np.cross(z_c, x_c)
    Rwc = np.stack([x_c, y_c, z_c], 0)
    t = -Rwc @ (n / np.linalg.norm(n) * dist)
    T = np.eye(4); T[:3, :3] = Rwc; T[:3, 3] = t
    return T


def render_face(mt, n, up):
    K = np.array([[FX, 0, RES / 2], [0, FX, RES / 2], [0, 0, 1]], float)
    pose = torch.as_tensor(pose_from(n, up, DIST)[None], device="cuda", dtype=torch.float)
    color, depth, _ = nvdiffrast_render(K=K, H=RES, W=RES, ob_in_cams=pose,
                                        mesh_tensors=mt, use_light=False)
    rgb = (color[0].clip(0, 1).cpu().numpy() * 255).astype(np.uint8)
    mask = depth[0].cpu().numpy() > 0
    if mask.any():                               # tight crop to the face quad
        ys, xs = np.where(mask)
        rgb = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return rgb


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edge_mm", type=float, default=60.0)
    args = ap.parse_args()

    mesh = trimesh.load(os.path.join(CODE_DIR, "mesh", "textured.obj"))
    mt = make_mesh_tensors(mesh)
    faces = {fi: render_face(mt, n, up) for fi, (n, up, _) in NET.items()}

    # ---- PNG preview (with labels + fold lines) ----
    cell = 360
    canvas = Image.new("RGB", (NCOLS * cell, NROWS * cell), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    try:
        f = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except Exception:
        f = ImageFont.load_default()
    for fi, (n, up, (r, c)) in NET.items():
        tile = Image.fromarray(faces[fi]).resize((cell, cell))
        canvas.paste(tile, (c * cell, r * cell))
        d.rectangle([c * cell, r * cell, c * cell + cell - 1, r * cell + cell - 1],
                    outline=(0, 0, 0), width=2)
        d.text((c * cell + 6, r * cell + 6),
               f"{FACE_NAMES[fi].split()[0]} {COLOR_NAMES[fi]} {LETTERS[fi]}",
               fill=(0, 0, 0), font=f)
    png_path = os.path.join(CODE_DIR, "mesh", "cube_net_guide.png")
    canvas.save(png_path)

    # ---- A4 landscape PDF, each cell exactly edge_mm ----
    A4_W, A4_H, MM = 297.0, 210.0, 25.4
    t = args.edge_mm
    fig = plt.figure(figsize=(A4_W / MM, A4_H / MM))
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, A4_W); bg.set_ylim(0, A4_H); bg.axis("off")
    block_w, block_h = NCOLS * t, NROWS * t
    left = (A4_W - block_w) / 2.0
    bottom = (A4_H - block_h) / 2.0
    for fi, (n, up, (r, c)) in NET.items():
        x = left + c * t
        y = bottom + (NROWS - 1 - r) * t          # row 0 at top
        ax = fig.add_axes([x / A4_W, y / A4_H, t / A4_W, t / A4_H])
        ax.imshow(faces[fi]); ax.axis("off")
        bg.add_patch(Rectangle((x, y), t, t, fill=False, ec="k", lw=0.6))
    bg.text(left, bottom - 6, f"Cut the cross, fold on the lines (each square = {t:.0f} mm). "
            f"Print at 100% / Actual size.", fontsize=8, va="top")
    pdf_path = os.path.join(CODE_DIR, "mesh", "cube_net_A4.pdf")
    fig.savefig(pdf_path, format="pdf")

    print("wrote", png_path)
    print("wrote", pdf_path)


if __name__ == "__main__":
    main()
