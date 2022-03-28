import scipy.io as sio
import numpy as np

data = sio.loadmat('/home/keb-kz/ss-dpc-net/data/libviso2-estimates/stereo/2011_09_26_drive_0001.mat')

pose_kitti = np.empty((data['poses_est'].shape[2],12))
gt_kitti = np.empty((data['poses_gt'].shape[2],12))

for i in range(pose_kitti.shape[0]):
    pose_kitti[i,0:4] = data['poses_est'][0,:,i]
    pose_kitti[i,4:8] = data['poses_est'][1,:,i]
    pose_kitti[i,8:12] = data['poses_est'][2,:,i]

for i in range(pose_kitti.shape[0]):
    gt_kitti[i,0:4]  = data['poses_gt'][0,:,i]
    gt_kitti[i,4:8]  = data['poses_gt'][1,:,i]
    gt_kitti[i,8:12] = data['poses_gt'][2,:,i]

np.savetxt('gt.kitti', gt_kitti)
np.savetxt('pose.kitti', pose_kitti)
