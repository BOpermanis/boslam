import numpy as np
import g2o
import sys
from pprint import pprint
from itertools import product
import cv2

# pprint(dir(pangolin.AxisDirection))
# sys.exit()

from collections import defaultdict


def get_camera_intrinsics(cam, flag_full=True):
    x, y = cam.principal_point
    f = cam.focal_length
    if flag_full:
        K = np.zeros((4, 4))
        K[3, 3] = 1
    else:
        K = np.zeros((3, 4))

    K[1, 1] = K[0, 0] = f
    K[2, 2] = 1
    K[0, 2] = x
    K[1, 2] = y

    return K


def show_pnts(*args, cameras=()):
    pangolin.CreateWindowAndBind('Main', 640, 480)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Define Projection and initial ModelView matrix
    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
        pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
    handler = pangolin.Handler3D(scam)

    # Create Interactive View in window
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0 / 480.0)
    dcam.SetHandler(handler)

    n = len(args)
    col1 = np.asarray([1, 0, 0])
    col2 = np.asarray([0, 100, 0])
    dict_pnt_cols = {k: (col1 * a + col2 * (1 - a)).astype(np.int32) for k, a in
                     enumerate(np.linspace(start=0.0, stop=1.0, num=n))}

    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        dcam.Activate(scam)

        for k, pnts in enumerate(args):
            gl.glPointSize(5)
            gl.glColor3f(*dict_pnt_cols[k])
            # print(pnts)
            # sys.exit()
            pangolin.DrawPoints(pnts)

        for camera in cameras:
            # Draw camera
            gl.glLineWidth(1)
            gl.glColor3f(0.0, 0.0, 1.0)
            pangolin.DrawCamera(camera, 0.5, 0.75, 0.8)

        pangolin.FinishFrame()


def ba_optimization(inds_cameras, inds_pnts, pixels, flag_verbose=True):
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
    # solver = g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())
    # solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
    # pprint([v for v in dir(g2o) if "solver" in v.lower()])
    # sys.exit()

    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(solver)

    focal_length = 1000
    principal_point = (320, 240)
    cam = g2o.CameraParameters(focal_length, principal_point, 0)
    cam.set_id(0)
    optimizer.add_parameter(cam)

    for i in set(inds_cameras):
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_id(i)
        R, t = np.random.normal(0, 2, size=(3, 3)), np.random.normal(0, 10, size=(3,))
        R, t = np.eye(3), np.zeros(3)
        pose_estimate = g2o.SE3Quat(R, t)
        print(R.shape, t.shape)
        exit()
        v_se3.set_estimate(pose_estimate)
        if i < 1:
            v_se3.set_fixed(True)
        optimizer.add_vertex(v_se3)

    # point_id = np.max(inds_cameras)
    inliers = dict()
    sse = defaultdict(float)

    # adding
    for i in set(inds_pnts):
        # point_id += 1
        vp = g2o.VertexSBAPointXYZ()
        vp.set_id(i)
        vp.set_marginalized(True)
        vp.set_estimate(np.random.normal(0, 1, (3,)))
        optimizer.add_vertex(vp)
        inliers[i] = i

    dict_verticePair2length = {}

    # pprint([v for v in dir(g2o) if 'edge' in v.lower()])
    # pprint([v for v in dir(g2o.EdgePointXYZ)])
    # sys.exit()
    # inds = np.unique(inds_pnts)
    # for a, b in product(inds, inds):
    #     l = np.linalg.norm(cube_pnts[a - 100, :] - cube_pnts[b - 100, :])
    #     if l == 1.0:
    #         edge = g2o.EdgePointXYZ()
    #         edge.set_vertex(0, optimizer.vertex(a))
    #         edge.set_vertex(1, optimizer.vertex(b))
    #         # print(edge.measurement_dimension())
    #         # sys.exit()
    #         edge.set_measurement(np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32))
    #
    #         edge.set_information(np.identity(2) / 100)
    #         # edge.set_robust_kernel(g2o.RobustKernelHuber())
    #
    #         edge.set_parameter_id(0, 0)
    #         optimizer.add_edge(edge)

    ## adding all constraints
    for id_cam, id_pnt, pixel in zip(inds_cameras, inds_pnts, pixels):
        edge = g2o.EdgeProjectXYZ2UV()
        edge.set_vertex(0, optimizer.vertex(id_pnt))
        edge.set_vertex(1, optimizer.vertex(id_cam))
        edge.set_measurement(pixel)

        edge.set_information(np.identity(2))
        edge.set_robust_kernel(g2o.RobustKernelHuber())

        edge.set_parameter_id(0, 0)
        optimizer.add_edge(edge)

    if flag_verbose:
        print('num vertices:', len(optimizer.vertices()))
        print('num edges:', len(optimizer.edges()))
        print('Performing full BA:')
    optimizer.initialize_optimization()
    optimizer.set_verbose(flag_verbose)
    optimizer.optimize(200)

    estimates = {}
    for i in inliers:
        vp = optimizer.vertex(i)
        estimates[inliers[i]] = np.asarray(vp.estimate().tolist())

    return estimates


def generate_data():
    cube_pnts = np.asarray(list(product(*[[0, 1]] * 3)))

    focal_length = 1000
    principal_point = (320, 240)
    cam = g2o.CameraParameters(focal_length, principal_point, 0)
    cam.set_id(0)

    true_poses = []
    num_pose = 100
    poses = []
    for i in range(num_pose):
        # pose here means transform points from world coordinates to camera coordinates
        pose = np.identity(4)
        pose[:3, 3] = [i * 0.04 - 1, 0, -8]
        poses.append(pose)

        pose = g2o.SE3Quat(pose[:3, :3], pose[:3, 3])
        true_poses.append(pose)

    point_id = num_pose
    inliers = dict()

    inds_cameras, inds_pnts, pixels = [], [], []

    for i, point in enumerate(cube_pnts):
        visible = []
        for j, pose in enumerate(true_poses):
            z = cam.cam_map(pose * point)
            z += np.random.normal(0, 3, (2,))

            if 0 <= z[0] < 640 and 0 <= z[1] < 480:
                visible.append((j, z))
        if len(visible) < 2:
            continue

        for j, z in visible:
            inds_cameras.append(j)
            inds_pnts.append(point_id)
            pixels.append(z)

        inliers[point_id] = i
        point_id += 1

    return inds_cameras, inds_pnts, pixels, cube_pnts, poses


def generate_data2():
    cube_pnts = np.asarray(list(product(*[[0, 1]] * 3)))

    focal_length = 1000
    principal_point = (320, 240)
    cam = g2o.CameraParameters(focal_length, principal_point, 0)
    cam.set_id(0)

    true_poses = []
    num_pose = 100
    poses = []
    a = np.pi / 2

    R = np.asarray([
        [np.cos(a), -np.sin(a)],
        [np.sin(a), np.cos(a)],
    ])

    for i in range(num_pose):
        if i < 50:
            pose = np.identity(4)
            pose[:3, 3] = [i * 0.04 - 1, 0, -8]
        elif i < 100:
            pose = np.identity(4)
            pose[1:3, 1:3] = R
            pose[:3, 3] = [0, 4, (i - 50) * 0.04 - 1]
        else:
            pose = np.identity(4)
            pose[0, 0] = R[0, 0]
            pose[0, 2] = R[0, 1]
            pose[2, 0] = R[1, 0]
            pose[2, 2] = R[1, 1]
            pose[:3, 3] = [0, 4, (i - 50) * 0.04 - 1]

        poses.append(pose)

        pose = g2o.SE3Quat(pose[:3, :3], pose[:3, 3])
        true_poses.append(pose)

    point_id = num_pose
    inliers = dict()

    inds_cameras, inds_pnts, pixels = [], [], []

    for i, point in enumerate(cube_pnts):
        visible = []
        for j, pose in enumerate(true_poses):
            z = cam.cam_map(pose * point)
            z += np.random.normal(0, 3, (2,))

            if 0 <= z[0] < 640 and 0 <= z[1] < 480:
                visible.append((j, z))
        if len(visible) < 2:
            continue

        for j, z in visible:
            inds_cameras.append(j)
            inds_pnts.append(point_id)
            pixels.append(z)

        inliers[point_id] = i
        point_id += 1

    return inds_cameras, inds_pnts, pixels, cube_pnts, poses


if __name__ == "__main__":
    import OpenGL.GL as gl
    import pangolin
    from pprint import pprint

    # inds_cameras, inds_pnts, pixels, cube_pnts, poses = generate_data()
    inds_cameras, inds_pnts, pixels, cube_pnts, poses = generate_data2()

    estimates = ba_optimization(inds_cameras, inds_pnts, pixels)

    estimates = [estimates[k] for k in sorted(list(estimates.keys()))]

    estimates = np.stack(estimates)

    print(np.min(estimates), np.max(estimates), estimates.shape)

    show_pnts(cube_pnts, estimates, cameras=poses)

    # np.save("/home/slam_data/data/head_estimates.npy", estimates)
