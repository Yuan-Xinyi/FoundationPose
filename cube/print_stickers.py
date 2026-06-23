#!/usr/bin/env python3
"""Make an A4-landscape PDF of the 6 cube face stickers, each EXACTLY the cube
edge length (default 60 mm) so they can be printed and stuck on the real cube.

Print it at 100% / "Actual size" (NOT "Fit to page" / "Shrink to fit"), then
check the 60 mm calibration ruler at the bottom with a real ruler.

Output: cube/mesh/sticker_sheet_A4.pdf   (run inside env_isaaclab)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from make_cube_mesh import (FACE_COLORS, FACE_NAMES, COLOR_NAMES, LETTERS, FACE_ROTATIONS, draw_tile)

A4_W, A4_H = 297.0, 210.0          # mm, landscape
MM_PER_IN = 25.4


def mmfrac(x_mm, y_mm, w_mm, h_mm):
    return [x_mm / A4_W, y_mm / A4_H, w_mm / A4_W, h_mm / A4_H]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    code_dir = os.path.dirname(os.path.realpath(__file__))
    ap.add_argument("--edge_mm", type=float, default=60.0, help="tile size in mm (cube edge)")
    ap.add_argument("--col_gap", type=float, default=20.0)
    ap.add_argument("--out", type=str, default=os.path.join(code_dir, "mesh", "sticker_sheet_A4.pdf"))
    args = ap.parse_args()
    t = args.edge_mm

    fig = plt.figure(figsize=(A4_W / MM_PER_IN, A4_H / MM_PER_IN))

    # background axes in mm coordinates for borders / labels / ruler
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, A4_W); bg.set_ylim(0, A4_H)
    bg.axis("off")

    # layout: 3 cols x 2 rows, centered horizontally
    block_w = 3 * t + 2 * args.col_gap
    left = (A4_W - block_w) / 2.0
    label_h, inter = 9.0, 12.0
    total_h = t + label_h + inter + t + label_h
    top = (A4_H - total_h) / 2.0
    row_top_y = [A4_H - top - t,                       # row 0 tile bottom-y
                 A4_H - top - t - label_h - inter - t]  # row 1 tile bottom-y

    for fi in range(6):
        c, r = fi % 3, fi // 3
        x = left + c * (t + args.col_gap)
        y = row_top_y[r]
        # tile image (identical to the mesh tiles)
        ax = fig.add_axes(mmfrac(x, y, t, t))
        ax.imshow(np.array(draw_tile(600, FACE_COLORS[fi], LETTERS[fi], FACE_ROTATIONS[fi])))
        ax.axis("off")
        # cut border exactly on the 60 mm edge
        bg.add_patch(Rectangle((x, y), t, t, fill=False, ec="k", lw=0.6))
        # corner crop marks
        m = 4.0
        for cx, sx in ((x, 1), (x + t, -1)):
            for cy, sy in ((y, 1), (y + t, -1)):
                bg.plot([cx, cx + sx * m], [cy, cy], "k-", lw=0.5)
                bg.plot([cx, cx], [cy, cy + sy * m], "k-", lw=0.5)
        bg.text(x, y - 6.5, f"{FACE_NAMES[fi].split()[0]}  {COLOR_NAMES[fi]}  ({LETTERS[fi]})",
                fontsize=9, va="top")

    # 60 mm calibration ruler at the bottom
    rx, ry = left, 12.0
    bg.plot([rx, rx + t], [ry, ry], "k-", lw=1.0)
    for k in range(0, int(t) + 1, 10):
        bg.plot([rx + k, rx + k], [ry, ry + 2.5], "k-", lw=1.0)
        bg.text(rx + k, ry + 3.5, f"{k}", fontsize=6, ha="center", va="bottom")
    bg.text(rx + t + 4, ry, f"<- this line must measure exactly {t:.0f} mm. "
            f"Print at 100% / Actual size (no fit-to-page).", fontsize=8, va="center")

    fig.savefig(args.out, format="pdf")
    print("wrote", args.out)
    print(f"A4 landscape, each tile = {t:.0f} mm. Print at 100% and verify the ruler.")


if __name__ == "__main__":
    main()
