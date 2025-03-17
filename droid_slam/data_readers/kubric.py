
import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp
import json

from lietorch import SE3
from .base import RGBDDataset, RGBDMotionDataset
from .stream import RGBDStream

cur_path = osp.dirname(osp.abspath(__file__))
test_split = osp.join(cur_path, 'tartan_test.txt')
test_split = open(test_split).read().split()


class KubricStatic(RGBDDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, mode='training', **kwargs):
        self.mode = mode
        self.n_frames = 2
        super(KubricStatic, self).__init__(name='KubricStatic', **kwargs)

    @staticmethod 
    def is_test_scene(scene):
        return False

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building KubricStatic dataset")

        scene_info = {}
        scenes = glob.glob(osp.join(self.root, '*'))
        for scene in tqdm(sorted(scenes)):
            images = sorted(glob.glob(osp.join(scene, 'rgba*.png')))
            depths = sorted(glob.glob(osp.join(scene, 'depth*.npy')))
            
            with open(osp.join(scene, 'metadata.json')) as f:
                metadata = json.load(f)
            cam = metadata['camera']
            print(metadata.keys())
            print(cam.keys())
            W, H = metadata['metadata']['resolution']
            K = cam['K']
            poses = np.array(cam['poses'])
            quaternions = np.array(cam['quaternions'])
            poses = np.concatenate([poses, quaternions], axis=1)
            poses[:, [1, 2]] = -poses[:, [1, 2]] # up to down, forward to backward
            poses[:, [4, 5]] = -poses[:, [4, 5]] # up to down, forward to backward
            poses[:,:3] /= KubricStatic.DEPTH_SCALE
            field_of_view = cam['field_of_view']
            focal_length = cam['focal_length']
            fx = 0.5 * W / np.tan(0.5 * float(field_of_view)) # need debub
            fy = 0.5 * H / np.tan(0.5 * float(field_of_view))
            cx = 0.5 * W
            cy = 0.5 * H
            intrinsics = np.array([fx, fy, cx, cy]) * len(images)
            # graph of co-visible frames based on flow
            graph = self.build_frame_graph(poses, depths, intrinsics)

            scene = '/'.join(scene.split('/'))
            scene_info[scene] = {'images': images, 'depths': depths, 
                'poses': poses, 'intrinsics': intrinsics, 'graph': graph}

        return scene_info

    @staticmethod
    def calib_read():
        return np.array([320.0, 320.0, 320.0, 240.0]) # fx fy cx cy

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        depth = np.load(depth_file) / KubricStatic.DEPTH_SCALE
        depth[depth==np.nan] = 1.0
        depth[depth==np.inf] = 1.0
        return depth
    
class KubricDynamic(RGBDMotionDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, mode='training', **kwargs):
        self.mode = mode
        self.n_frames = 2
        super(KubricDynamic, self).__init__(name='KubricDynamic', **kwargs)

    @staticmethod 
    def is_test_scene(scene):
        return False

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building KubricDynamic dataset")

        scene_info = {}
        scenes = glob.glob(osp.join(self.root, '*'))
        for scene in tqdm(sorted(scenes)):
            images = sorted(glob.glob(osp.join(scene, 'rgba*.png')))
            depths = sorted(glob.glob(osp.join(scene, 'depth*.npy')))
            movement_maps = sorted(glob.glob(osp.join(scene, 'mask*.png')))
            
            with open(osp.join(scene, 'metadata.json')) as f:
                metadata = json.load(f)
            cam = metadata['camera']
            W, H = metadata['metadata']['resolution']
            K = cam['K']
            poses = np.array(cam['poses'])
            quaternions = np.array(cam['quaternions'])
            poses = np.concatenate([poses, quaternions], axis=1)
            poses[:, [1, 2]] = -poses[:, [1, 2]] # up to down, forward to backward
            poses[:, [4, 5]] = -poses[:, [4, 5]] # up to down, forward to backward
            poses[:,:3] /= KubricStatic.DEPTH_SCALE
            field_of_view = cam['field_of_view']
            focal_length = cam['focal_length']
            fx = 0.5 * W / np.tan(0.5 * float(field_of_view)) # need debub
            fy = 0.5 * H / np.tan(0.5 * float(field_of_view))
            cx = 0.5 * W
            cy = 0.5 * H
            intrinsics = np.array([fx, fy, cx, cy]) * len(images)

            # graph of co-visible frames based on flow
            graph = self.build_frame_graph(poses, depths, intrinsics)

            scene = '/'.join(scene.split('/'))
            scene_info[scene] = {'images': images, 'depths': depths, 'movement_maps': movement_maps,
                'poses': poses, 'intrinsics': intrinsics, 'graph': graph}

        return scene_info

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        depth = np.load(depth_file) / KubricDynamic.DEPTH_SCALE
        depth[depth==np.nan] = 1.0
        depth[depth==np.inf] = 1.0
        return depth
    
    @staticmethod
    def movement_map_read(movement_file):
        return np.load(movement_file)


class KubricStaticStream(RGBDStream):
    def __init__(self, datapath, **kwargs):
        super(KubricStaticStream, self).__init__(datapath=datapath, **kwargs)

    def _build_dataset_index(self):
        """ build list of images, poses, depths, and intrinsics """
        self.root = 'datasets/KubricStatic'

        scene = osp.join(self.root, self.datapath)
        image_glob = osp.join(scene, 'rgba*.png')
        images = sorted(glob.glob(image_glo))

        poses = np.loadtxt(osp.join(scene, 'pose.txt'), delimiter=' ')
        with open(osp.join(scene, 'metadata.json')) as f:
            metadata = json.load(f)

        poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]

        poses = SE3(torch.as_tensor(poses))
        poses = poses[[0]].inv() * poses
        poses = poses.data.cpu().numpy()

        intrinsic = self.calib_read(self.datapath)
        intrinsics = np.tile(intrinsic[None], (len(images), 1))

        self.images = images[::int(self.frame_rate)]
        self.poses = poses[::int(self.frame_rate)]
        self.intrinsics = intrinsics[::int(self.frame_rate)]

    @staticmethod
    def calib_read(datapath):
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)


class KubricStaticTestStream(RGBDStream):
    def __init__(self, datapath, **kwargs):
        super(KubricStaticTestStream, self).__init__(datapath=datapath, **kwargs)

    def _build_dataset_index(self):
        """ build list of images, poses, depths, and intrinsics """
        self.root = 'datasets/mono'
        image_glob = osp.join(self.root, self.datapath, '*.png')
        images = sorted(glob.glob(image_glob))

        with open(osp.join(scene, 'metadata.json')) as f:
            metadata = json.load(f)
        cam = metadata['camera']
        W, H = metadata['metadata']['resolution']
        K = cam['K']
        poses = np.array(cam['poses'])
        quaternions = np.array(cam['quaternions'])
        poses = np.concatenate([poses, quaternions], axis=1)
        poses[:, [1, 2]] = -poses[:, [1, 2]] # up to down, forward to backward
        poses[:, [4, 5]] = -poses[:, [4, 5]] # up to down, forward to backward
        field_of_view = cam['field_of_view']
        focal_length = cam['focal_length']
        fx = 0.5 * W / np.tan(0.5 * float(field_of_view)) # need debub
        fy = 0.5 * H / np.tan(0.5 * float(field_of_view))
        cx = 0.5 * W
        cy = 0.5 * H
        intrinsic = np.array([fx, fy, cx, cy]) * len(images)

        poses = SE3(torch.as_tensor(poses))
        poses = poses[[0]].inv() * poses
        poses = poses.data.cpu().numpy()

        intrinsics = np.tile(intrinsic[None], (len(images), 1))

        self.images = images[::int(self.frame_rate)]
        self.poses = poses[::int(self.frame_rate)]
        self.intrinsics = intrinsics[::int(self.frame_rate)]

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)