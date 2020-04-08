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


def convert_depth_frame_to_pointcloud(depth_image, camera_intrinsics):
    """
    Convert the depthmap to a 3D point cloud
    Parameters:
    -----------
    depth_frame 	 	 : rs.frame()
                           The depth_frame containing the depth map
    camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
    Return:
    ----------
    x : array
        The x values of the pointcloud in meters
    y : array
        The y values of the pointcloud in meters
    z : array
        The z values of the pointcloud in meters
    """

    height, width = depth_image.shape

    nx = np.linspace(0, width - 1, width)
    ny = np.linspace(0, height - 1, height)
    u, v = np.meshgrid(nx, ny)
    # from pprint import pprint
    # pprint(dir(camera_intrinsics.ppx))
    # print(camera_intrinsics.ppx.fget)
    # exit()
    x = (u.flatten() - camera_intrinsics.ppx) / camera_intrinsics.fx
    y = (v.flatten() - camera_intrinsics.ppy) / camera_intrinsics.fy

    z = depth_image.flatten() / 1000;
    x = np.multiply(x, z)
    y = np.multiply(y, z)

    x = x[np.nonzero(z)]
    y = y[np.nonzero(z)]
    z = z[np.nonzero(z)]

    return np.stack([x, y, z], axis=1)
