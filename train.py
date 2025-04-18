import sys
sys.path.append('droid_slam')

import cv2
import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_readers.factory import dataset_factory

from lietorch import SO3, SE3, Sim3
from geom import losses
from geom.losses import geodesic_loss, residual_loss, flow_loss
from geom.graph_utils import build_frame_graph

# network
from droid_net import DroidNet
from logger import Logger

# DDP training
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from geom.graph_utils import graph_to_edge_list, keyframe_indicies
from torch_scatter import scatter_mean



def setup_ddp(gpu, args):
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',     
    	world_size=args.world_size,                              
    	rank=gpu)

    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def train_stage1(gpu, args):
    """ Test to make sure project transform correctly maps points """

    # coordinate multiple GPUs
    setup_ddp(gpu, args)
    rng = np.random.default_rng(12345)

    N = args.n_frames
    model = DroidNet()
    model.cuda()
    model.train()

    model = DDP(model, device_ids=[gpu], find_unused_parameters=True)

    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt))

    # fetch dataloader
    db = dataset_factory(['tartan', 'kubric_static'], datapath=args.datapath, n_frames=args.n_frames, fmin=args.fmin, fmax=args.fmax)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        db, shuffle=True, num_replicas=args.world_size, rank=gpu)

    train_loader = DataLoader(db, batch_size=args.batch, sampler=train_sampler, num_workers=2)

    # fetch optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
        args.lr, args.steps1, pct_start=0.01, cycle_momentum=False)

    logger = Logger(args.name, scheduler)
    should_keep_training = True
    total_steps = 0

    while should_keep_training:
        for i_batch, item in enumerate(train_loader):
            optimizer.zero_grad()

            images, poses, disps, intrinsics, *optinal_data = [x.to('cuda') if x is not None else None for x in item]
            print('image1', images.shape)
            mono_depth = None
            if len(optinal_data) > 0:
                mono_depth = optinal_data[0]

            # convert poses w2c -> c2w
            Ps = SE3(poses).inv()
            Gs = SE3.IdentityLike(Ps)

            # randomize frame graph
            if np.random.rand() < 0.5:
                print('disp', disps.shape)
                graph = build_frame_graph(poses, disps, intrinsics, num=args.edges)
            else:
                graph = OrderedDict()
                for i in range(N):
                    graph[i] = [j for j in range(N) if i!=j and abs(i-j) <= 2]
            
            # fix first to camera poses
            Gs.data[:,0] = Ps.data[:,0].clone()
            Gs.data[:,1:] = Ps.data[:,[1]].clone()
            if mono_depth is None:
                disp0 = torch.ones_like(disps[:,:,3::8,3::8])
            else:
                disp0 = mono_depth[:,:,3::8,3::8].detach()

            # perform random restarts
            r = 0
            while r < args.restart_prob:
                r = rng.random()
                
                intrinsics0 = intrinsics / 8.0
                print('image2', images.shape)
                poses_est, disps_est, residuals, _, _ = model(Gs, images, disp0, None, intrinsics0, 
                    graph, num_steps=args.iters, fixedp=2)
                print('disps_est', disps_est[0].shape)
                print('poses_est', poses_est[0].shape, Ps.shape)
                print('residuals', residuals[0].shape)
                geo_loss, geo_metrics = losses.geodesic_loss(Ps, poses_est, graph, do_scale=False)
                res_loss, res_metrics = losses.residual_loss(residuals)
                flo_loss, flo_metrics = losses.flow_loss(Ps, disps, poses_est, disps_est, intrinsics, graph)

                loss = args.w1 * geo_loss + args.w2 * res_loss + args.w3 * flo_loss
                loss.backward()

                Gs = poses_est[-1].detach()
                disp0 = disps_est[-1][:,:,3::8,3::8].detach()

            metrics = {}
            metrics.update(geo_metrics)
            metrics.update(res_metrics)
            metrics.update(flo_metrics)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            
            total_steps += 1

            if gpu == 0:
                logger.push(metrics)

            if total_steps % 100 == 0 and gpu == 0:
                logger.write_images('disps', disps, disps_est[-1])

            if total_steps % 10000 == 0 and gpu == 0:
                PATH = 'checkpoints/%s_stage1_%06d.pth' % (args.name, total_steps)
                torch.save(model.state_dict(), PATH)

            if total_steps >= args.steps1:
                should_keep_training = False
                break

    dist.destroy_process_group()
    if gpu == 0:
        PATH = 'checkpoints/%s_stage1.pth' % (args.name)
        torch.save(model.state_dict(), PATH)

    return PATH
                
def train_stage2(gpu, args):
    """ Test to make sure project transform correctly maps points """

    # coordinate multiple GPUs
    setup_ddp(gpu, args)
    rng = np.random.default_rng(12345)

    N = args.n_frames
    model = DroidNet()
    model.cuda()
    model.train()

    model = DDP(model, device_ids=[gpu], find_unused_parameters=True)

    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt))

    # fetch dataloader
    db = dataset_factory(['kubric_dynamic'], datapath=args.datapath, n_frames=args.n_frames, fmin=args.fmin, fmax=args.fmax)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        db, shuffle=True, num_replicas=args.world_size, rank=gpu)

    train_loader = DataLoader(db, batch_size=args.batch, sampler=train_sampler, num_workers=2)

    # fetch optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
        args.lr, args.steps2, pct_start=0.01, cycle_momentum=False)

    logger = Logger(args.name, scheduler)
    should_keep_training = True
    total_steps = 0

    while should_keep_training:
        for i_batch, item in enumerate(train_loader):
            optimizer.zero_grad()

            images, poses, disps, intrinsics, *optional_data = [x.to('cuda') if x is not None else None for x in item]
            mono_depth = None
            motion_masks = None
            if len(optional_data) > 0:
                motion_masks = optional_data[-1]
            if len(optional_data) > 1:
                mono_depth = optional_data[-2]

            # convert poses w2c -> c2w
            Ps = SE3(poses).inv()
            Gs = SE3.IdentityLike(Ps)
            Ms = motion_masks
            batch_size, num_frames, H, W, C = Ms.shape
            Ms = torch.nn.functional.interpolate(Ms.view(batch_size*num_frames, H, W,C).permute(0, 3, 1, 2), scale_factor=0.125, mode='bilinear').permute(0, 2, 3, 1).view(batch_size, num_frames, H//8, W//8, C).detach()
            # randomize frame graph
            if np.random.rand() < 0.5:
                graph = build_frame_graph(poses, disps, intrinsics, num=args.edges)
            
            else:
                graph = OrderedDict()
                for i in range(N):
                    graph[i] = [j for j in range(N) if i!=j and abs(i-j) <= 2]
            
            # fix first to camera poses
            Gs.data[:,0] = Ps.data[:,0].clone()
            Gs.data[:,1:] = Ps.data[:,[1]].clone()
            if mono_depth is None:
                disp0 = torch.ones_like(disps[:,:,3::8,3::8])
            else:
                disp0 = mono_depth[:,:,3::8,3::8].detach()

            # perform random restarts
            r = 0
            while r < args.restart_prob:
                r = rng.random()
                
                intrinsics0 = intrinsics / 8.0
                poses_est, disps_est, residuals, mot_prob, _ = model(Gs, images, disp0, None, intrinsics0, 
                    graph, num_steps=args.iters, fixedp=2)

                geo_loss, geo_metrics = losses.geodesic_loss(Ps, poses_est, graph, do_scale=False)
                ii, _, _ = graph_to_edge_list(graph)
                ii = ii.to(device=mot_prob[0].device, dtype=torch.long)
                _, ix = torch.unique(ii, return_inverse=True)
                mot_prob = [scatter_mean(m, ix, dim=1) for m in mot_prob]

                motion_loss, mot_metrics = losses.motion_loss(Ms, mot_prob)

                loss = args.w1 * geo_loss + args.w4 * motion_loss
                loss.backward()

                Gs = poses_est[-1].detach()
                disp0 = disps_est[-1][:,:,3::8,3::8].detach()

            metrics = {}
            metrics.update(geo_metrics)
            metrics.update(mot_metrics)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            
            total_steps += 1

            if gpu == 0:
                logger.push(metrics)

            if total_steps % 100 == 0 and gpu == 0:
                logger.write_images('motion_mask', Ms, mot_prob[-1])
                logger.write_images('disps', disps, disps_est[-1])

            if total_steps % 10000 == 0 and gpu == 0:
                PATH = 'checkpoints/%s_stage2_%06d.pth' % (args.name, total_steps)
                torch.save(model.state_dict(), PATH)

            if total_steps >= args.steps2:
                should_keep_training = False
                break

    dist.destroy_process_group()
    if gpu == 0:
        PATH = 'checkpoints/%s_final.pth' % (args.name, total_steps)
        torch.save(model.state_dict(), PATH)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help='name your experiment')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--datasets', nargs='+', help='lists of datasets for training')
    parser.add_argument('--datapath', default='datasets/TartanAir', help="path to dataset directory")
    parser.add_argument('--gpus', type=int, default=4)

    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--iters', type=int, default=15)
    parser.add_argument('--steps1', type=int, default=100000)
    parser.add_argument('--steps2', type=int, default=150000)
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--clip', type=float, default=2.5)
    parser.add_argument('--n_frames', type=int, default=7)

    parser.add_argument('--w1', type=float, default=10.0)
    parser.add_argument('--w2', type=float, default=0.01)
    parser.add_argument('--w3', type=float, default=0.05)
    parser.add_argument('--w4', type=float, default=0.1)

    parser.add_argument('--fmin', type=float, default=8.0)
    parser.add_argument('--fmax', type=float, default=96.0)
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--edges', type=int, default=24)
    parser.add_argument('--restart_prob', type=float, default=0.2)

    args = parser.parse_args()

    args.world_size = args.gpus
    print(args)

    import os
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    args = parser.parse_args()
    args.world_size = args.gpus

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    args.ckpt = mp.spawn(train_stage1, nprocs=args.gpus, args=(args,))
    mp.spawn(train_stage2, nprocs=args.gpus, args=(args,))

