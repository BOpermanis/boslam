import numpy as np
import g2o

from config import chi2_sig_value
from slam.nodes import KeyFrame, MapPoint


class BundleAdjustment(g2o.SparseOptimizer):
    def __init__(self, cam):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)
        self.cam = cam

    def optimize(self, max_iterations=20, flag_verbose=True):
        super().initialize_optimization()
        super().set_verbose(flag_verbose)
        super().optimize(max_iterations)

    def add_pose(self, kf: KeyFrame, cam, fixed=False):
        pose_id, R, t = kf.idf(), kf.Rf(), kf.tf()
        if len(t.shape) == 2:
            t = t[:, 0]
        pose_estimate = g2o.SE3Quat(R, t)
        sbacam = g2o.SBACam(pose_estimate.orientation(), pose_estimate.position())
        if hasattr(cam, 'fx'):
            sbacam.set_cam(cam.fx, cam.fy, cam.cx, cam.cy, cam.baseline)
        else:
            cx, cy = cam.principal_point
            fx, fy = cam.focal_length, cam.focal_length
            sbacam.set_cam(fx, fy, cx, cy, cam.baseline)

        v_se3 = g2o.VertexCam()
        v_se3.set_id(pose_id * 2)   # internal id
        v_se3.set_estimate(sbacam)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_point(self, mp: MapPoint, fixed=False, marginalized=True):
        point_id, point = mp.idf(), mp.pt3df()
        v_p = g2o.VertexSBAPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_estimate(point)
        v_p.set_marginalized(marginalized)
        v_p.set_fixed(fixed)
        super().add_vertex(v_p)

    def add_edge(self, mp, kf, measurement, information=np.identity(2), robust_kernel=g2o.RobustKernelHuber(np.sqrt(chi2_sig_value))):   # 95% CI
        edge = g2o.EdgeProjectP2MC()
        point_id, pose_id = mp.idf(), kf.idf()
        pose = g2o.SE3Quat(kf.Rf(), kf.tf())#[:, 0])
        pixel = self.cam.cam_map(pose * mp.pt3d)
        if 0 <= pixel[0] < 640 and 0 <= pixel[1] < 480:
            edge.set_vertex(0, self.vertex(point_id * 2 + 1))
            edge.set_vertex(1, self.vertex(pose_id * 2))
            edge.set_measurement(measurement)   # projection
            edge.set_information(information)

            if robust_kernel is not None:
                edge.set_robust_kernel(robust_kernel)
            super().add_edge(edge)

    def get_pose(self, pose_id):
        m = self.vertex(pose_id * 2).estimate().matrix()
        return m[:3, :3], m[:3, 3]

    def get_point(self, point_id):
        return self.vertex(point_id * 2 + 1).estimate().T

    def outlier_edges(self):
        bad_edges = []
        for e in self.edges():
            if e.chi2() > chi2_sig_value:
                id_mp, id_kf = [_.id() for _ in e.vertices()]
                id_mp = int((id_mp - 1) / 2)
                id_kf = int(id_kf / 2)
                bad_edges.append((id_kf, id_mp))
        return bad_edges


def CamOnlyBA(pixels, points, cam, R=None, t=None, information=np.identity(2), robust_kernel=g2o.RobustKernelHuber(np.sqrt(chi2_sig_value)), num_iter=10, flag_verbose=False):

    try:
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(solver)

        if True: #R is None:
            R, t = np.eye(3), np.zeros((3,))
        if len(t.shape) == 2:
            t = t[:, 0]
        pose_estimate = g2o.SE3Quat(R, t)

        sbacam = g2o.SBACam(pose_estimate.orientation(), pose_estimate.position())
        if hasattr(cam, 'fx'):
            sbacam.set_cam(cam.fx, cam.fy, cam.cx, cam.cy, cam.baseline)
        else:
            cx, cy = cam.principal_point
            fx, fy = cam.focal_length, cam.focal_length
            sbacam.set_cam(fx, fy, cx, cy, cam.baseline)

        v_se3 = g2o.VertexCam()
        v_se3.set_id(0)  # internal id
        v_se3.set_estimate(sbacam)

        optimizer.add_vertex(v_se3)

        for point_id, (pixel, point) in enumerate(zip(pixels, points)):
            v_p = g2o.VertexSBAPointXYZ()
            v_p.set_id(point_id + 1)
            v_p.set_estimate(point)
            v_p.set_marginalized(True)
            v_p.set_fixed(True)
            optimizer.add_vertex(v_p)

            edge = g2o.EdgeProjectP2MC()
            edge.set_vertex(0, optimizer.vertex(point_id + 1))
            edge.set_vertex(1, optimizer.vertex(0))
            edge.set_measurement(pixel)  # projection
            edge.set_information(information)
            if robust_kernel is not None:
                edge.set_robust_kernel(robust_kernel)
            optimizer.add_edge(edge)

        optimizer.initialize_optimization()
        optimizer.set_verbose(flag_verbose)
        optimizer.optimize(num_iter)

        inliers = []
        for e in optimizer.edges():
            if e.chi2() <= chi2_sig_value:
                id_mp, _ = [_.id() for _ in e.vertices()]
                inliers.append(id_mp - 1)

        m = optimizer.vertex(0).estimate().matrix()
        # print(m.shape, m[:3, :3].shape, m[:3, 3].shape)
        return True, m[:3, :3], m[:3, 3], np.asarray(inliers)
    except Exception as e:
        print(e)
        return False, None, None, None


