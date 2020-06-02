data_dir = "/home/slam_data/data_sets/boslam_data"

chi2_sig_value = 7.815

# culling params
min_common_ratio = 0.9
min_found_ratio = 0.25
min_n_obs = 3

min_kp_in_frame = 40


# orbslam params
d_hamming_max = 30

# minimal number of matches between keyframes
min_matches_cg = 15

# minimal number of matches between kf in essential graph
min_matches_eg = 100

dbow_tresh = 0.1

num_frames_from_last_kf = 3
num_frames_from_last_relocation = 10


import numpy as np
cam_mat = np.asarray([
    [384.23901367, 0., 322.43237305],
    [0., 384.23901367, 239.65332031],
    [0., 0., 1.]
])

distCoeffs = np.zeros((8, 1), dtype=np.float32)

fx, fy, cx, cy = 384.239013671875, 384.239013671875, 322.432373046875, 239.6533203125
f = 384.239013671875
principal_point = 322.432373046875, 239.6533203125
baseline = 0.06
width = 640
height = 480
