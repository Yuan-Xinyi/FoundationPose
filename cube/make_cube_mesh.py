#!/usr/bin/env python3
"""Generate a textured cube mesh (color + letter per face) for FoundationPose.

Each of the 6 faces gets a distinct high-contrast COLOR and a distinct LETTER.
- The color breaks face-to-face symmetry.
- The letter adds an intra-face feature that breaks the 90-degree rotation
  ambiguity you get with a plain solid-color face (this is why the Isaac DexCube
  uses lettered faces). Result: a unique, stable pose.

Because letters are intra-face detail, the mesh uses a TEXTURE atlas (not vertex
colors). FoundationPose's make_mesh_tensors() takes the TextureVisuals branch.

Units are METERS. Set --edge to the real measured edge length (default 0.06 = 6 cm).

Outputs (in cube/mesh/):
    textured.obj / material.mtl / material_0.png   <- pass textured.obj to run_demo.py
    sticker_sheet.png   printable: 6 labeled tiles (color+letter) to stick on the cube
    cube_net.png        unfolded reference (which color+letter on which face)

After generating, run  preview_cube.py  for a faithful 3D view of the result.
"""
import argparse
import os

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

# Face order: +X, -X, +Y, -Y, +Z, -Z
FACE_NAMES = ["+X (right)", "-X (left)", "+Y (back)", "-Y (front)", "+Z (top)", "-Z (bottom)"]
FACE_COLORS = [
    (220, 20, 20),    # red
    (20, 180, 60),    # green
    (30, 70, 230),    # blue
    (245, 220, 20),   # yellow
    (20, 200, 220),   # cyan
    (225, 30, 200),   # magenta
]
COLOR_NAMES = ["red", "green", "blue", "yellow", "cyan", "magenta"]
LETTERS = ["A", "B", "C", "D", "F", "E"]
COLS, ROWS = 3, 2   # texture atlas layout


def _font(size):
    for name in ("DejaVuSans-Bold.ttf", "DejaV_Sans_Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _text_color(bg):
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    return (0, 0, 0) if lum > 140 else (255, 255, 255)


FACE_ROTATIONS = [0, 0, 0, 0, 180, 180]


def draw_tile(size, color, letter, rotation=0):
    img = Image.new("RGB", (size, size), color)
    d = ImageDraw.Draw(img)
    font = _font(int(size * 0.7))
    tc = _text_color(color)
    bb = d.textbbox((0, 0), letter, font=font)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    d.text(((size - w) / 2 - bb[0], (size - h) / 2 - bb[1]), letter, fill=tc, font=font)
    if rotation:
        img = img.rotate(rotation, expand=False)
    return img


def build_atlas(tile=512):
    img = Image.new("RGB", (COLS * tile, ROWS * tile), (0, 0, 0))
    for fi in range(6):
        c, r = fi % COLS, fi // COLS
        img.paste(draw_tile(tile, FACE_COLORS[fi], LETTERS[fi], FACE_ROTATIONS[fi]), (c * tile, r * tile))
    # FoundationPose's make_mesh_tensors flips v (uv[:,1]=1-uv[:,1]); pre-flip the
    # atlas vertically so that flip cancels and each face renders its intended,
    # upright tile.
    return Image.fromarray(np.flipud(np.array(img)))


def build_cube(edge_m):
    """24 vertices (4 per face) + UVs into the 3x2 atlas.

    Each face is generated from a right-handed in-plane frame (u_dir, v_dir, n)
    with u_dir x v_dir = n (outward). Uniform handedness across all 6 faces means
    a single texture-orientation convention makes every letter render upright and
    un-mirrored when viewed from outside.
    """
    h = edge_m / 2.0
    NORMALS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    UPS     = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 1, 0), (0, 1, 0)]
    vertices, faces, uv = [], [], []
    inset = 0.04
    for fi, (n, up) in enumerate(zip(NORMALS, UPS)):
        n = np.array(n, float); v_dir = np.array(up, float)
        u_dir = np.cross(v_dir, n)          # u x v = n  (right-handed, outward)
        c, r = fi % COLS, fi // COLS
        u0, u1 = (c + inset) / COLS, (c + 1 - inset) / COLS
        v0, v1 = (r + inset) / ROWS, (r + 1 - inset) / ROWS
        # corners at (s,t) in {-1,+1}; (s,t)->(u,v). U reversed so text reads
        # correctly from outside; V handled by the pre-flipped atlas.
        corner = lambda s, t: (n + u_dir * s + v_dir * t) * h
        base = len(vertices)
        vertices.extend([corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)])
        uv.extend([(u1, v0), (u0, v0), (u0, v1), (u1, v1)])
        faces.append([base + 0, base + 1, base + 2])
        faces.append([base + 0, base + 2, base + 3])
    return (np.array(vertices, np.float64), np.array(faces, np.int64), np.array(uv, np.float64))


def build_sticker_sheet(edge_m, out_path, tile=512):
    """6 labeled tiles in a 2x3 grid, for printing and sticking on the cube."""
    pad, lab = 24, 64
    cell = tile + lab
    W = 3 * tile + 4 * pad
    H = 2 * cell + 3 * pad
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f = _font(34)
    for fi in range(6):
        c, r = fi % 3, fi // 3
        x = pad + c * (tile + pad)
        y = pad + r * (cell + pad)
        img.paste(draw_tile(tile, FACE_COLORS[fi], LETTERS[fi], FACE_ROTATIONS[fi]), (x, y))
        d.text((x, y + tile + 12),
               f"{FACE_NAMES[fi].split()[0]}  {COLOR_NAMES[fi]}  ({LETTERS[fi]})",
               fill=(0, 0, 0), font=f)
    note = _font(28)
    d.text((pad, H - 34), f"Print so each square = {edge_m*100:.1f} cm (the cube edge).",
           fill=(0, 0, 0), font=note)
    img.save(out_path)


def build_net(edge_m, out_path, tile=150):
    layout = {4: (1, 0), 1: (0, 1), 2: (1, 1), 0: (2, 1), 3: (3, 1), 5: (1, 2)}
    img = Image.new("RGB", (tile * 4, tile * 3), (40, 40, 40))
    d = ImageDraw.Draw(img)
    big, small = _font(54), _font(18)
    for fi, (c, r) in layout.items():
        x0, y0 = c * tile, r * tile
        tile_img = draw_tile(tile - 4, FACE_COLORS[fi], LETTERS[fi], FACE_ROTATIONS[fi])
        img.paste(tile_img, (x0 + 2, y0 + 2))
        tc = _text_color(FACE_COLORS[fi])
        d.text((x0 + 18, y0 + 8), f"{FACE_NAMES[fi].split()[0]} {COLOR_NAMES[fi]}", fill=tc, font=small)
        bb = d.textbbox((0, 0), LETTERS[fi], font=big)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        d.text((x0 + (tile - w) / 2 - bb[0], y0 + (tile - h) / 2 - bb[1] + 10),
               LETTERS[fi], fill=tc, font=big)
    img.save(out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    code_dir = os.path.dirname(os.path.realpath(__file__))
    ap.add_argument("--edge", type=float, default=0.06,
                    help="Cube edge length in METERS (measure your real cube!). Default 0.06")
    ap.add_argument("--out_dir", type=str, default=os.path.join(code_dir, "mesh"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # clean any previous mesh outputs so we don't leave a stale .ply / atlas
    for f in ("textured.ply", "textured.obj", "material.mtl", "material_0.png"):
        p = os.path.join(args.out_dir, f)
        if os.path.exists(p):
            os.remove(p)

    verts, faces, uv = build_cube(args.edge)
    atlas = build_atlas()
    material = trimesh.visual.material.SimpleMaterial(image=atlas)
    visual = trimesh.visual.TextureVisuals(uv=uv, material=material, image=atlas)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, visual=visual, process=False)

    obj_path = os.path.join(args.out_dir, "textured.obj")
    mesh.export(obj_path)
    build_sticker_sheet(args.edge, os.path.join(args.out_dir, "sticker_sheet.png"))
    build_net(args.edge, os.path.join(args.out_dir, "cube_net.png"))

    print("=== Cube mesh (color + letter) written ===")
    print(f"  mesh    : {obj_path}")
    print(f"  files   : {sorted(os.listdir(args.out_dir))}")
    print(f"  extents : {mesh.extents}  (meters; should all == {args.edge})")
    print()
    print("=== Face -> color + letter ===")
    for n, col, cn, le in zip(FACE_NAMES, FACE_COLORS, COLOR_NAMES, LETTERS):
        print(f"  {n:14s} -> {cn:8s} '{le}'  RGB{col}")
    print()
    reloaded = trimesh.load(obj_path)
    print(f"Reload check: extents {reloaded.extents}, visual {type(reloaded.visual).__name__}")
    print("Next: python cube/preview_cube.py   (faithful 3D render of the result)")


if __name__ == "__main__":
    main()
