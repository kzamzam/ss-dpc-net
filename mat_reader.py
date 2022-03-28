import scipy.io as sio
import numpy as np
from evo.tools import file_interface
from evo.core import metrics

data = sio.loadmat('/home/keb-kz/ss-dpc-net/data/libviso2-estimates/stereo/2011_09_26_drive_0005.mat') # mat of results
#print(data.keys())
#traj = data['poses_est']
#gt = data['poses_gt']

#gt = gt.transpose(2,0,1)
#traj = traj.transpose(2,0,1)
con = 'test'
traj = data[con]['est_traj'][0][0]
print(traj.shape)
gt = data[con]['gt_traj'][0][0]
print(gt.shape)

gt_kitti = np.empty((gt.shape[0],12))
for i in range(gt.shape[0]):
    gt_kitti[i,0:4] = gt[i,0,:]
    gt_kitti[i,4:8] = gt[i,1,:]
    gt_kitti[i,8:12] = gt[i,2,:]

np.savetxt('gt.kitti', gt_kitti)

# traj_kitti = np.empty((gt.shape[0],12))
# for i in range(gt.shape[0]):
#     traj_kitti[i,0:4] =  traj [i,0,:]
#     traj_kitti[i,4:8] =  traj [i,1,:]
#     traj_kitti[i,8:12] = traj[i,2,:]

# np.savetxt('traj.kitti', traj_kitti)



gt_kitti = file_interface.read_kitti_poses_file('gt.kitti')
min = 100
epoch = 0
for x in range(traj.shape[0]):
    traj_kitti = np.empty((traj.shape[1],12))
    for i in range(traj.shape[1]):
        traj_kitti[i,0:4] = traj[x,i,0,:]
        traj_kitti[i,4:8] = traj[x,i,1,:]
        traj_kitti[i,8:12] = traj[x,i,2,:]

    np.savetxt('tmp.kitti', traj_kitti)
    traj_kitti = file_interface.read_kitti_poses_file('tmp.kitti')

    traj_kitti.align(gt_kitti)
    pose_relation = metrics.PoseRelation.translation_part

    set = (gt_kitti, traj_kitti)

    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(set)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    print(ape_stat)
    if ape_stat < min:
        min = ape_stat
        epoch = x


traj_kitti = np.empty((traj.shape[1],12))
for i in range(traj.shape[1]):
    traj_kitti[i,0:4] = traj[epoch,i,0,:]
    traj_kitti[i,4:8] = traj[epoch,i,1,:]
    traj_kitti[i,8:12] = traj[epoch,i,2,:]

np.savetxt('traj.kitti', traj_kitti)

