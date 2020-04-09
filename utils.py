import numpy as np

def depth_to_3d(depth, coords, cam):
    coords = np.array(coords, dtype=int)
    ix = coords[:, 0]
    iy = coords[:, 1]
    depth = depth[iy, ix]

    zs = depth / cam.scale
    xs = (ix - cam.cx) * zs / cam.fx
    ys = (iy - cam.cy) * zs / cam.fy
    return np.column_stack([xs, ys, zs])



