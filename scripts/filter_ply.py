#!/usr/bin/env python3
"""
Remove 3DGS floaters / junk by cropping to a horizontal radius around the
camera path, while preserving tall structures (pillars, ceiling).

The camera walked around the subject, so in the HORIZONTAL plane the real
scene is close to the camera path; anything far away horizontally is a floater
(transient people, weak-geometry artifacts). Tall structures extend in the
VERTICAL direction, which we deliberately do NOT crop.

Usage:
  python3 filter_ply.py input.ply output.ply cameras.bin \
      --max-horiz-dist 4.0 [--dry-run]
"""

import argparse
import os
import struct

import numpy as np
from plyfile import PlyData, PlyElement


def read_images_bin(path):
    """Return (centers, up_vectors) as (N,3) float64 arrays."""
    data = open(path, 'rb').read()
    n = struct.unpack_from('<Q', data, 0)[0]
    pos = 8
    centers = []
    ups = []
    for _ in range(n):
        pos += 4  # image_id
        qw, qx, qy, qz, tx, ty, tz = struct.unpack_from('<7d', data, pos)
        pos += 56
        pos += 4  # camera_id
        end = data.index(b'\x00', pos)
        pos = end + 1
        npts = struct.unpack_from('<Q', data, pos)[0]
        pos += 8 + npts * 24  # 2x double + int64 per point2D

        # world -> camera rotation from quaternion (w,x,y,z)
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ])
        t = np.array([tx, ty, tz])
        centers.append(-R.T @ t)
        # camera's +Y axis expressed in world coords = row 1 of R
        ups.append(R[1, :])
    return np.array(centers), np.array(ups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ply_in')
    ap.add_argument('ply_out')
    ap.add_argument('cameras_bin')
    ap.add_argument('--max-horiz-dist', type=float, default=4.0,
                    help='Max horizontal distance to nearest camera (keep <= this).')
    ap.add_argument('--min-opacity', type=float, default=-4.5,
                    help='Drop gaussians with logit opacity below this '
                         '(actual opacity = sigmoid(value)); -10 = off.')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print stats only, do not write output.')
    args = ap.parse_args()

    centers, ups = read_images_bin(args.cameras_bin)
    up = ups.mean(0)
    up /= np.linalg.norm(up)  # vertical axis (pointing roughly camera-up)
    print(f"vertical axis (world): {np.round(up, 3)}")
    print(f"cameras: {len(centers)}")

    ply = PlyData.read(args.ply_in)
    v = ply['vertex'].data
    x = v['x'].astype(np.float64)
    y = v['y'].astype(np.float64)
    z = v['z'].astype(np.float64)
    N = len(x)
    print(f"gaussians: {N}")

    pts = np.stack([x, y, z], 1)
    # Project everything onto the horizontal plane (perpendicular to `up`).
    def horiz(v):
        return v - (v @ up)[:, None] * up[None, :]

    hpts = horiz(pts)
    hcam = horiz(centers)

    # min horizontal distance to any camera (chunked)
    hdist = np.full(N, np.inf)
    CH = 20000
    for i in range(0, N, CH):
        d = np.linalg.norm(hpts[i:i+CH][:, None, :] - hcam[None, :, :], axis=2)
        hdist[i:i+CH] = d.min(1)

    print("\nhorizontal distance to nearest camera (world units):")
    for th in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10]:
        print(f"  <{th:>4}: {(hdist<th).sum():>7} ({(hdist<th).mean()*100:5.1f}%)")

    op = v['opacity'].astype(np.float64)
    opmask = op >= args.min_opacity
    print(f"\nopacity filter (logit >= {args.min_opacity}): "
          f"remove {int((~opmask).sum())} ({(~opmask).mean()*100:.1f}%)")

    mask = (hdist <= args.max_horiz_dist) & opmask
    kept = int(mask.sum())
    print(f"combined: keep {kept} / {N} ({kept/N*100:.1f}%)  "
          f"remove {N-kept} ({(N-kept)/N*100:.1f}%)")

    if args.dry_run:
        return

    filtered = v[mask]
    el = PlyElement.describe(filtered, 'vertex')
    PlyData([el], text=False).write(args.ply_out)
    print(f"wrote {args.ply_out} ({os.path.getsize(args.ply_out)/1e6:.1f} MB)")


if __name__ == '__main__':
    main()
