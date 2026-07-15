#!/usr/bin/env python3
"""Live D405 + FoundationPose pose tracking with on-screen visualization.

Shows the live color stream with the estimated pose drawn on the real cube
(3D box + XYZ axes), plus a "virtual cube" rendered at the recognized
orientation in the top-right corner.

Startup: draw a box around the cube on the first frame (ENTER), GrabCut makes
the registration mask, then it tracks live.

Keys:  r = re-init (draw box again)   q/ESC = quit

Run inside env_isaaclab (needs pyrealsense2 + the FoundationPose stack).
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from estimater import *                      # FoundationPose, ScorePredictor, PoseRefinePredictor, dr, draw_* , make_mesh_tensors, nvdiffrast_render
from make_mask import grabcut                # reuse the box->mask helper

CODE_DIR = os.path.dirname(os.path.realpath(__file__))


def build_estimator(mesh_file, debug=0, debug_dir="/tmp/fp_live"):
    os.makedirs(debug_dir, exist_ok=True)
    mesh = trimesh.load(mesh_file)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                         mesh=mesh, scorer=ScorePredictor(), refiner=PoseRefinePredictor(),
                         glctx=dr.RasterizeCudaContext(), debug=debug, debug_dir=debug_dir)
    mt = make_mesh_tensors(mesh)
    return est, mesh, to_origin, bbox, mt


def render_corner(pose, mt, obj_diam, size=190, fx=620.0, fit_frac=0.7):
    """Render the mesh at the estimated ORIENTATION -> RGB panel.
    Camera distance auto-fits so the object spans ~fit_frac of the panel (works for any object size)."""
    dist = max(0.15, fx * obj_diam / (fit_frac * size))
    R = pose[:3, :3]
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [0, 0, dist]
    K = np.array([[fx, 0, size / 2], [0, fx, size / 2], [0, 0, 1]], float)
    Tt = torch.as_tensor(T[None], device="cuda", dtype=torch.float)
    color, depth, _ = nvdiffrast_render(K=K, H=size, W=size, ob_in_cams=Tt,
                                        mesh_tensors=mt, use_light=True,
                                        light_pos=np.array([0, 0, 0]), w_ambient=0.6, w_diffuse=0.6)
    rgb = (color[0].clip(0, 1).cpu().numpy() * 255).astype(np.uint8)
    mask = depth[0].cpu().numpy() > 0
    panel = np.full((size, size, 3), 38, np.uint8)
    panel[mask] = rgb[mask]
    return panel


def annotate(color_rgb, pose, K, to_origin, bbox, mt, fps=None):
    """Draw box + axes on the real frame and paste the virtual cube top-right. RGB in/out."""
    center_pose = pose @ np.linalg.inv(to_origin)
    vis = draw_posed_3d_box(K, img=color_rgb.copy(), ob_in_cam=center_pose, bbox=bbox)
    vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.05, K=K, thickness=2,
                        transparency=0, is_input_rgb=True)
    H, W = vis.shape[:2]
    obj_diam = float(np.linalg.norm(bbox[1] - bbox[0]))   # object diagonal, for auto-fit framing
    panel = render_corner(pose, mt, obj_diam)
    s, mgn = panel.shape[0], 10
    y0, x0 = mgn, W - mgn - s
    vis[y0:y0 + s, x0:x0 + s] = panel
    cv2.rectangle(vis, (x0, y0), (x0 + s, y0 + s), (255, 255, 255), 1)
    cv2.putText(vis, "recognized pose", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    t = pose[:3, 3]
    cv2.putText(vis, f"xyz(m): {t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}", (10, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if fps is not None:
        cv2.putText(vis, f"{fps:4.1f} FPS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return vis


def select_mask(color_rgb):
    """Draw a box on the first frame -> GrabCut -> boolean mask."""
    bgr = color_rgb[..., ::-1].copy()
    rect = cv2.selectROI("init: drag box around cube, ENTER", bgr, showCrosshair=False)
    cv2.destroyWindow("init: drag box around cube, ENTER")
    if rect[2] == 0 or rect[3] == 0:
        return None
    return grabcut(bgr, rect) > 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh_file", default=os.path.join(CODE_DIR, "mesh", "textured.obj"))
    ap.add_argument("--serial", default=None, help="指定相机序列号 (D435=938422071322)；不传则用第一台")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--est_refine_iter", type=int, default=5)
    ap.add_argument("--track_refine_iter", type=int, default=2)
    args = ap.parse_args()

    import pyrealsense2 as rs
    set_logging_format(); set_seed(0)
    est, mesh, to_origin, bbox, mt = build_estimator(args.mesh_file)

    pipeline = rs.pipeline(); config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    for _ in range(15):
        pipeline.wait_for_frames()

    def grab():
        fr = align.process(pipeline.wait_for_frames())
        c = np.asarray(fr.get_color_frame().get_data())            # RGB
        d = np.asarray(fr.get_depth_frame().get_data()).astype(np.float32) * depth_scale  # meters
        intr = fr.get_color_frame().profile.as_video_stream_profile().intrinsics
        K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], float)
        return c, d, K

    print("Drawing init box... (ENTER to confirm, c to cancel selection)")
    color, depth, K = grab()
    mask = select_mask(color)
    if mask is None:
        print("No mask selected, abort."); pipeline.stop(); return
    pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
    print("registered. tracking... (r=re-init, q=quit)")

    win = "FoundationPose live (r=re-init  q=quit)"
    t_prev, fps = time.time(), 0.0
    try:
        while True:
            color, depth, K = grab()
            pose = est.track_one(rgb=color, depth=depth, K=K, iteration=args.track_refine_iter)
            now = time.time(); fps = 0.9 * fps + 0.1 * (1.0 / max(1e-3, now - t_prev)); t_prev = now
            vis = annotate(color, pose, K, to_origin, bbox, mt, fps=fps)
            cv2.imshow(win, vis[..., ::-1])                        # RGB->BGR
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('r'):
                m = select_mask(color)
                if m is not None:
                    pose = est.register(K=K, rgb=color, depth=depth, ob_mask=m, iteration=args.est_refine_iter)
    finally:
        pipeline.stop(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
