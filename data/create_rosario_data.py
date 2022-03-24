import numpy as np
import scipy.io as sio
import imageio
import os
import concurrent.futures
from PIL import Image
import argparse
from liegroups import SE3
import glob

from torch import greater

parser = argparse.ArgumentParser(description='arguments.')
parser.add_argument("--source_dir", type=str, default='/home/keb-kz/Thesis/VIO/rosario_dataset_pics/')
parser.add_argument("--target_dir", type=str, default='/home/keb-kz/Thesis/VIO/rosario-dpc-processed/')
parser.add_argument("--remove_static", action='store_true', default=True)
args = parser.parse_args()

target_dir = args.target_dir
os.makedirs(target_dir, exist_ok=True)
seq_info = {}
sequences = ['seq1','seq2']

args.height = 240 # 240 or 360
args.width = 376  # 376 or 564

msckf_dir = 'rosario_estimates/'
        
def load_image(img_file):
    img_height = args.height #240 #360 #
    img_width = args.width #376 #564 
    img = np.array(Image.open(img_file))
    orig_img_height = img.shape[0]
    orig_img_width = img.shape[1]
    zoom_y = img_height/orig_img_height
    zoom_x = img_width/orig_img_width
#    img = np.array(Image.fromarray(img).crop([425, 65, 801, 305]))
    img = np.array(Image.fromarray(img).resize((img_width, img_height)))
    return img, zoom_x, zoom_y, orig_img_width, orig_img_height

    ###Iterate through all specified KITTI sequences and extract raw data, and trajectories
for i,seq in enumerate(sequences):     
    print('Sequence being processed:',seq)
    msckf_data = sio.loadmat(msckf_dir+seq+'.mat')  #sparse VO
    print('gt: ', msckf_data['poses_gt'].shape, '  vo: ',msckf_data['poses_est'].shape)
    seq_dir = os.path.join(target_dir, seq)
    os.makedirs(seq_dir, exist_ok=True)
    os.makedirs(os.path.join(seq_dir, 'image_02'), exist_ok=True)
    os.makedirs(os.path.join(seq_dir, 'image_03'), exist_ok=True)
    
    ###store filenames of camera data, and intrinsic matrix
    k_cam2 = np.zeros((3,3))
    k_cam2[0,0] = 260.66287231 
    k_cam2[0,2] = 324.40546129
    k_cam2[1,1] = 257.44598389
    k_cam2[1,2] = 147.88597818
    k_cam2[2,2] = 1
    
    seq_info['intrinsics'] =k_cam2.reshape((-1,3,3)).repeat(msckf_data['poses_gt'].shape[2],0)
    i = 0
    with concurrent.futures.ProcessPoolExecutor() as executor: 
        # put file names for cam2 in np array
        cam2_files = sorted(glob.glob(args.source_dir+seq+'/image_02/*.jpg'))
        for filename, output in zip(cam2_files, executor.map(load_image, cam2_files)):
            img, zoomx, zoomy, orig_img_width, orig_img_height = output
            new_filename = os.path.join(target_dir, filename.split(args.source_dir)[1])#.replace('.png','.jpg')
            imageio.imwrite(new_filename, img)
            seq_info['intrinsics'][i,0] *= zoomx
            seq_info['intrinsics'][i,1] *= zoomy
            cam2_files[i] = np.array(new_filename)
            i+=1
        i = 0
        cam3_files = os.listdir(args.source_dir+seq+'/image_03')
        for n,file in enumerate(cam3_files):
            file = args.source_dir+seq+'/image_03/'+file
            cam3_files[n] = file
        for filename, output in zip(cam3_files, executor.map(load_image, cam3_files)):
            img, zoomx, zoomy, orig_img_width, orig_img_height = output
            new_filename = os.path.join(target_dir, filename.split(args.source_dir)[1]).replace('.png','.jpg')
            imageio.imwrite(new_filename, img)
            cam3_files[i] = np.array(new_filename)
            i+=1

    seq_info['cam_02'] = np.array(cam2_files)
    seq_info['cam_03'] = np.array(cam3_files)
        
    ###Import libviso2 estimate for correcting
    stereo_traj = msckf_data['poses_est'].transpose(2,0,1)
    ### store the ground truth pose
    seq_info['sparse_gt_pose'] = msckf_data['poses_gt'].transpose(2,0,1)
    stereo_seq_info = seq_info.copy()
    ### store the VO pose estimates to extract 
    stereo_seq_info['sparse_vo'] = stereo_traj
    
    ###filter out frames with low rotational or translational velocities
    for seq_info in [stereo_seq_info]:
        max = 0
        sum = 0
        count = 0
        limit = 0.025
        if args.remove_static:
            print("Removing Static frames from {}".format(seq))            
            deleting = True
            
            while deleting:
                greater_count = 0
                idx_list = []
                sparse_traj = np.copy(seq_info['sparse_vo'])
                for i in range(0,sparse_traj.shape[0]-1,2):
                    T2 = SE3.from_matrix(sparse_traj[i+1,:,:], normalize=True).inv()
                    T1 = SE3.from_matrix(sparse_traj[i,:,:], normalize=True)
                    dT = T2.dot(T1)
                    pose_vec = dT.log()
                    trans_norm = np.linalg.norm(pose_vec[0:3])
                    rot_norm = np.linalg.norm(pose_vec[3:6])
                    # print('Frame: ',i)
                    # print('trans: ', trans_norm, ' rot: ', rot_norm)
                    # print('-------------------------------------')
                    sum += rot_norm
                    count += 1
                    if rot_norm > max: 
                        max = rot_norm
                        print("The maximum till now is: ", max)
                    if rot_norm > limit:
                        greater_count += 1
                    if trans_norm < 0.001 and rot_norm < 0.0004: #0.007
                        idx_list.append(i)
                if len(idx_list) == 0:
                    deleting = False
                
                print('deleting {} frames'.format(len(idx_list)))
                print('original length: {}'.format(seq_info['cam_02'].shape))
                
                seq_info['intrinsics'] = np.delete(seq_info['intrinsics'],idx_list,axis=0)
                seq_info['cam_02'] = np.delete(seq_info['cam_02'],idx_list,axis=0)
                seq_info['cam_03'] = np.delete(seq_info['cam_03'],idx_list,axis=0)
                seq_info['sparse_gt_pose'] = np.delete(seq_info['sparse_gt_pose'],idx_list,axis=0)
                seq_info['sparse_vo'] = np.delete(seq_info['sparse_vo'],idx_list,axis=0)
                print('final length: {}'.format(seq_info['cam_02'].shape))
                print ('average rot :', sum/count)
                print('Total more than limit: ', greater_count)

    
    sio.savemat(seq_dir + '/stereo_data.mat', stereo_seq_info)
