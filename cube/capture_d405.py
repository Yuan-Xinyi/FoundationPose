#!/usr/bin/env python3
"""Live RGB-D capture from an Intel RealSense D405 into FoundationPose format.

Writes (matching demo_data/mustard0 layout):
    cube/rgb/00000.png      RGB, 8-bit
    cube/depth/00000.png    depth, 16-bit uint16, MILLIMETERS  (FoundationPose divides by 1000)
    cube/cam_K.txt          3x3 color intrinsics (written once)

Depth is aligned to the color frame, so rgb[i] and depth[i] are pixel-aligned and
share the color intrinsics in cam_K.txt.

Run inside the env that has pyrealsense2 (here: `conda activate wrs`).

Interactive keys (a preview window must be visible):
    SPACE  save one frame
    c      toggle continuous capture (save every frame)
    q/ESC  quit

Headless / scripted alternative:
    python capture_d405.py --num 100 --interval 0.1   # auto-grab 100 frames, no GUI
"""
import argparse
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs


def make_dirs(out_dir):
    for sub in ("rgb", "depth", "masks"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)


def write_cam_K(out_dir, intr):
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]], dtype=np.float64)
    np.savetxt(os.path.join(out_dir, "cam_K.txt"), K, fmt="%.8f")
    return K


def save_frame(out_dir, idx, color_rgb, depth_mm):
    name = f"{idx:05d}.png"
    # imageio/cv2: store RGB as-is; cv2.imwrite expects BGR, so convert
    cv2.imwrite(os.path.join(out_dir, "rgb", name), cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "depth", name), depth_mm)  # uint16 png
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    code_dir = os.path.dirname(os.path.realpath(__file__))
    ap.add_argument("--out_dir", type=str, default=code_dir)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--num", type=int, default=0,
                    help="If >0: headless auto-capture this many frames then exit (no GUI).")
    ap.add_argument("--interval", type=float, default=0.0,
                    help="Seconds between auto-captured frames (with --num).")
    ap.add_argument("--start_idx", type=int, default=0)
    args = ap.parse_args()

    make_dirs(args.out_dir)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()  # raw_z16 * depth_scale = meters
    align = rs.align(rs.stream.color)

    # warm up auto-exposure
    for _ in range(15):
        pipeline.wait_for_frames()

    frames = align.process(pipeline.wait_for_frames())
    color_intr = frames.get_color_frame().profile.as_video_stream_profile().intrinsics
    K = write_cam_K(args.out_dir, color_intr)
    print(f"depth_scale = {depth_scale}  (raw*scale = meters)")
    print(f"cam_K.txt written:\n{K}")
    print(f"Saving to: {args.out_dir}/rgb , /depth   (depth = uint16 millimeters)")

    idx = args.start_idx
    headless = args.num > 0
    if not headless:
        print("Keys:  SPACE=save one  |  c=toggle continuous  |  q/ESC=quit")
    continuous = False
    saved = 0

    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            cframe = frames.get_color_frame()
            dframe = frames.get_depth_frame()
            if not cframe or not dframe:
                continue

            color_rgb = np.asarray(cframe.get_data())            # HxWx3 RGB uint8
            depth_raw = np.asarray(dframe.get_data())            # HxW uint16, raw z16
            depth_m = depth_raw.astype(np.float32) * depth_scale  # meters
            depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)

            do_save = False
            if headless:
                do_save = True
            else:
                # preview: color + colorized depth side by side
                dcol = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_mm, alpha=255.0 / max(1, depth_mm.max())),
                    cv2.COLORMAP_JET)
                vis = np.hstack([cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR), dcol])
                cv2.putText(vis, f"saved={saved} idx={idx} {'[CONT]' if continuous else ''}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("D405  (SPACE=save  c=continuous  q=quit)", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('c'):
                    continuous = not continuous
                if key == 32 or continuous:   # SPACE
                    do_save = True

            if do_save:
                save_frame(args.out_dir, idx, color_rgb, depth_mm)
                idx += 1
                saved += 1
                if headless:
                    print(f"  saved {saved}/{args.num}", end="\r")
                    if saved >= args.num:
                        break
                    if args.interval > 0:
                        time.sleep(args.interval)
    finally:
        pipeline.stop()
        if not headless:
            cv2.destroyAllWindows()

    print(f"\nDone. Saved {saved} frame(s). First frame: rgb/{args.start_idx:05d}.png")
    print("Next: create the first-frame mask ->  python make_mask.py")


if __name__ == "__main__":
    main()
