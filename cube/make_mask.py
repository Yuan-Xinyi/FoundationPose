#!/usr/bin/env python3
"""Create the first-frame segmentation mask FoundationPose needs to register.

FoundationPose only needs a mask for the FIRST frame (the one it registers on);
tracking takes over afterwards. The mask is white (object) on black (background),
same filename as the rgb frame, under cube/masks/.

Two ways to run:
  GUI (default): draw a box around the cube, GrabCut refines it to a tight mask.
      python make_mask.py
      -> a window opens; drag a rectangle around the cube, press ENTER/SPACE.
         Then: s=save  r=redo box  q=quit
  Headless: pass the box explicitly (x,y,w,h in pixels), GrabCut runs, mask saved.
      python make_mask.py --rect 270,190,140,150

Run inside the env that has cv2 (here: `conda activate wrs`).
"""
import argparse
import glob
import os

import cv2
import numpy as np


def grabcut(img_bgr, rect, iters=5):
    mask = np.zeros(img_bgr.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr, mask, tuple(rect), bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    code_dir = os.path.dirname(os.path.realpath(__file__))
    ap.add_argument("--out_dir", type=str, default=code_dir)
    ap.add_argument("--frame", type=str, default=None,
                    help="rgb frame to mask (default: first file in rgb/).")
    ap.add_argument("--rect", type=str, default=None,
                    help="Headless box 'x,y,w,h'. If omitted, opens a GUI to draw it.")
    args = ap.parse_args()

    rgb_dir = os.path.join(args.out_dir, "rgb")
    mask_dir = os.path.join(args.out_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    frame = args.frame
    if frame is None:
        files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
        if not files:
            raise SystemExit(f"No rgb frames in {rgb_dir}. Capture first (capture_d405.py).")
        frame = files[0]
    img = cv2.imread(frame)
    if img is None:
        raise SystemExit(f"Could not read {frame}")
    name = os.path.basename(frame)
    out_path = os.path.join(mask_dir, name)

    if args.rect:
        x, y, w, h = (int(v) for v in args.rect.split(","))
        mask = grabcut(img, (x, y, w, h))
        cv2.imwrite(out_path, mask)
        print(f"Mask saved: {out_path}  (foreground px = {int((mask > 0).sum())})")
        return

    # GUI flow
    print("Drag a rectangle around the cube, then press ENTER or SPACE.")
    rect = cv2.selectROI("select cube", img, showCrosshair=False)
    cv2.destroyWindow("select cube")
    if rect[2] == 0 or rect[3] == 0:
        raise SystemExit("Empty box, aborted.")

    while True:
        mask = grabcut(img, rect)
        overlay = img.copy()
        overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        cv2.putText(overlay, "s=save  r=redo box  q=quit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("mask preview", overlay)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):
            cv2.imwrite(out_path, mask)
            print(f"Mask saved: {out_path}  (foreground px = {int((mask > 0).sum())})")
            break
        if key == ord('r'):
            cv2.destroyWindow("mask preview")
            rect = cv2.selectROI("select cube", img, showCrosshair=False)
            cv2.destroyWindow("select cube")
            continue
        if key in (ord('q'), 27):
            print("Aborted, nothing saved.")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
