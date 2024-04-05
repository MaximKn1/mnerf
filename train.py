import sys
import time

import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(sys.path[0], '../..'))

from dataloader.with_colmap import DataLoaderWithCOLMAP
from utils.training_utils import set_randomness, mse2psnr, save_checkpoint
from utils.pos_enc import encode_position
from utils.volume_op import volume_sampling_ndc, volume_rendering
from utils.comp_ray_dir import comp_ray_dir_cam_fxfy
from utils.comp_ate import compute_ate
from models.nerf_models import OfficialNerf
from models.intrinsics import LearnFocal
from models.poses import LearnPose


def store_current_pose(pose_net, pose_history_dir, epoch_i):
    pose_net.eval()

    num_cams = pose_net.module.num_cams if isinstance(pose_net, torch.nn.DataParallel) else pose_net.num_cams

    c2w_list = []
    for i in range(num_cams):
        c2w = pose_net(i)  # (4, 4)
        c2w_list.append(c2w)

    c2w_list = torch.stack(c2w_list)  # (N, 4, 4)
    c2w_list = c2w_list.detach().cpu().numpy()

    np.save(os.path.join(pose_history_dir, str(epoch_i).zfill(6)), c2w_list)
    return


def model_render_image(c2w, rays_cam, t_vals, near, far, H, W, fxfy, model, perturb_t, sigma_noise_std,
                       args, rgb_act_fn):
    """Render an image or pixels.
    :param c2w:         (4, 4)                  pose to transform ray direction from cam to world.
    :param rays_cam:    (someH, someW, 3)       ray directions in camera coordinate, can be random selected
                                                rows and cols, or some full rows, or an entire image.
    :param t_vals:      (N_samples)             sample depth along a ray.
    :param fxfy:        a float or a (2, ) torch tensor for focal.
    :param perturb_t:   True/False              whether add noise to t.
    :param sigma_noise_std: a float             std dev when adding noise to raw density (sigma).
    :rgb_act_fn:        sigmoid()               apply an activation fn to the raw rgb output to get actual rgb.
    :return:            (someH, someW, 3)       volume rendered images for the input rays.
    """
    # (H, W, N_sample, 3), (H, W, 3), (H, W, N_sam)
    sample_pos, _, ray_dir_world, t_vals_noisy = volume_sampling_ndc(c2w, rays_cam, t_vals, near, far,
                                                                     H, W, fxfy, perturb_t)

    # encode position: (H, W, N_sample, (2L+1)*C = 63)
    pos_enc = encode_position(sample_pos, levels=args.pos_enc_levels, inc_input=args.pos_enc_inc_in)

    # encode direction: (H, W, N_sample, (2L+1)*C = 27)
    if args.use_dir_enc:
        ray_dir_world = F.normalize(ray_dir_world, p=2, dim=2)  # (H, W, 3)
        dir_enc = encode_position(ray_dir_world, levels=args.dir_enc_levels, inc_input=args.dir_enc_inc_in)  # (H, W, 27)
        dir_enc = dir_enc.unsqueeze(2).expand(-1, -1, args.num_sample, -1)  # (H, W, N_sample, 27)
    else:
        dir_enc = None

    # inference rgb and density using position and direction encoding.
    rgb_density = model(pos_enc, dir_enc)  # (H, W, N_sample, 4)

    render_result = volume_rendering(rgb_density, t_vals_noisy, sigma_noise_std, rgb_act_fn)
    rgb_rendered = render_result['rgb']  # (H, W, 3)
    depth_map = render_result['depth_map']  # (H, W)

    result = {
        'rgb': rgb_rendered,  # (H, W, 3)
        'sample_pos': sample_pos,  # (H, W, N_sample, 3)
        'depth_map': depth_map,  # (H, W)
        'rgb_density': rgb_density,  # (H, W, N_sample, 4)
    }

    return result


def eval_one_epoch_img(eval_c2ws, scene_train, model, focal_net, pose_param_net,
                       my_devices, args, epoch_i, writer, rgb_act_fn):
    model.eval()
    focal_net.eval()
    pose_param_net.eval()

    fxfy = focal_net(0)
    ray_dir_cam = comp_ray_dir_cam_fxfy(scene_train.H, scene_train.W, fxfy[0], fxfy[1])
    t_vals = torch.linspace(scene_train.near, scene_train.far, args.num_sample, device=my_devices)  # (N_sample,) sample position
    N_img = eval_c2ws.shape[0]

    rendered_img_list = []
    rendered_depth_list = []

    for i in range(N_img):
        time0 = time.time()
        c2w = eval_c2ws[i].to(my_devices)  # (4, 4)

        # split an image to rows when the input image resolution is high
        rays_dir_cam_split_rows = ray_dir_cam.split(args.num_rows_eval_img, dim=0)
        rendered_img = []
        rendered_depth = []
        for rays_dir_rows in rays_dir_cam_split_rows:
            render_result = model_render_image(c2w, rays_dir_rows, t_vals, scene_train.near, scene_train.far,
                                               scene_train.H, scene_train.W, fxfy,
                                               model, False, 0.0, args, rgb_act_fn)
            rgb_rendered_rows = render_result['rgb']  # (num_rows_eval_img, W, 3)
            depth_map = render_result['depth_map']  # (num_rows_eval_img, W)

            rendered_img.append(rgb_rendered_rows)
            rendered_depth.append(depth_map)

        # combine rows to an image
        rendered_img = torch.cat(rendered_img, dim=0)
        rendered_depth = torch.cat(rendered_depth, dim=0).unsqueeze(0)  # (1, H, W)
        dt = time.time()-time0
        print(f"Image {i} render time: {dt}")

        # for vis
        rendered_img_list.append(rendered_img.cpu().numpy())
        rendered_depth_list.append(rendered_depth.cpu().numpy())

    # random display an eval image to tfboard
    rand_num = np.random.randint(low=0, high=N_img)
    disp_img = np.transpose(rendered_img_list[rand_num], (2, 0, 1))  # (3, H, W)
    disp_depth = rendered_depth_list[rand_num]  # (1, H, W)
    writer.add_image('eval_img', disp_img, global_step=epoch_i)
    writer.add_image('eval_depth', disp_depth, global_step=epoch_i)


def eval_one_epoch_traj(scene_train, pose_param_net):
    pose_param_net.eval()

    est_c2ws_train = torch.stack([pose_param_net(i) for i in range(scene_train.N_imgs)])  # (N, 4, 4)
    colmap_c2ws_train = scene_train.c2ws  # (N, 4, 4) torch
    stats_tran, stats_rot, stats_scale = compute_ate(est_c2ws_train, colmap_c2ws_train, align_a2b='sim3')

    return stats_tran, stats_rot, stats_scale


def train_one_epoch(scene_train, optimizer_nerf, optimizer_focal, optimizer_pose, model, focal_net, pose_param_net,
                    my_devices, args, rgb_act_fn, epoch_i):
    model.train()

    if epoch_i >= args.start_refine_pose_epoch:
        pose_param_net.train()
    else:
        pose_param_net.eval()

    if epoch_i >= args.start_refine_focal_epoch:
        focal_net.train()
    else:
        focal_net.eval()

    t_vals = torch.linspace(scene_train.near, scene_train.far, args.num_sample, device=my_devices)  # (N_sample,) sample position
    N_img, H, W = scene_train.N_imgs, scene_train.H, scene_train.W
    L2_loss_epoch = []

    # shuffle the training imgs
    ids = np.arange(N_img)
    np.random.shuffle(ids)

    for i in ids:
        img = scene_train.imgs[i].to(my_devices)  # (H, W, 3)

        if epoch_i >= args.start_refine_focal_epoch:
            fxfy = focal_net(0)
            ray_dir_cam = comp_ray_dir_cam_fxfy(H, W, fxfy[0], fxfy[1])
        else:
            fxfy = scene_train.focal
            ray_dir_cam = scene_train.ray_dir_cam.to(my_devices)

        if epoch_i >= args.start_refine_pose_epoch:
            c2w = pose_param_net(i)  # (4, 4)
        else:
            with torch.no_grad():
                c2w = pose_param_net(i)  # (4, 4)

        # sample pixel on an image and their rays for training.
        r_id = torch.randperm(H, device=my_devices)[:args.train_rand_rows]  # (N_select_rows)
        c_id = torch.randperm(W, device=my_devices)[:args.train_rand_cols]  # (N_select_cols)
        ray_selected_cam = ray_dir_cam[r_id][:, c_id]  # (N_select_rows, N_select_cols, 3)
        img_selected = img[r_id][:, c_id]  # (N_select_rows, N_select_cols, 3)

        # render an image using selected rays, pose, sample intervals, and the network
        render_result = model_render_image(c2w, ray_selected_cam, t_vals, scene_train.near, scene_train.far,
                                           scene_train.H, scene_train.W, fxfy,
                                           model, True, 0.0, args, rgb_act_fn)  # (N_select_rows, N_select_cols, 3)
        rgb_rendered = render_result['rgb']  # (N_select_rows, N_select_cols, 3)

        L2_loss = F.mse_loss(rgb_rendered, img_selected)  # loss for one image

        L2_loss.backward()
        optimizer_nerf.step()
        optimizer_focal.step()
        optimizer_pose.step()
        optimizer_nerf.zero_grad()
        optimizer_focal.zero_grad()
        optimizer_pose.zero_grad()

        L2_loss_epoch.append(L2_loss.item())

    L2_loss_epoch_mean = np.mean(L2_loss_epoch)  # loss for all images.
    mean_losses = {
        'L2': L2_loss_epoch_mean,
    }
    return mean_losses


def main(args):
    '''Training'''
    for epoch_i in tqdm(range(args.epoch), desc='epochs'):
        rgb_act_fn = torch.sigmoid
        train_epoch_losses = train_one_epoch(scene_train, optimizer_nerf, optimizer_focal, optimizer_pose,
                                             model, focal_net, pose_param_net, my_devices, args, rgb_act_fn, epoch_i)
        train_L2_loss = train_epoch_losses['L2']
        scheduler_nerf.step()
        scheduler_focal.step()
        scheduler_pose.step()

        train_psnr = mse2psnr(train_L2_loss)
        writer.add_scalar('train/mse', train_L2_loss, epoch_i)
        writer.add_scalar('train/psnr', train_psnr, epoch_i)
        writer.add_scalar('train/lr', scheduler_nerf.get_lr()[0], epoch_i)
        logger.info('{0:6d} ep: Train: L2 loss: {1:.4f}, PSNR: {2:.3f}'.format(epoch_i, train_L2_loss, train_psnr))
        tqdm.write('{0:6d} ep: Train: L2 loss: {1:.4f}, PSNR: {2:.3f}'.format(epoch_i, train_L2_loss, train_psnr))

        pose_history_milestone = list(range(0, 100, 5)) + list(range(100, 1000, 100)) + list(range(1000, 10000, 1000))
        if epoch_i in pose_history_milestone:
            with torch.no_grad():
                if args.store_pose_history:
                    store_current_pose(pose_param_net, pose_history_dir, epoch_i)

        if epoch_i % args.eval_cam_interval == 0 and epoch_i > 0:
            with torch.no_grad():
                eval_stats_tran, eval_stats_rot, eval_stats_scale = eval_one_epoch_traj(scene_train, pose_param_net)
                writer.add_scalar('eval/traj/translation', eval_stats_tran['mean'], epoch_i)
                writer.add_scalar('eval/traj/rotation', eval_stats_rot['mean'], epoch_i)
                writer.add_scalar('eval/traj/scale', eval_stats_scale['mean'], epoch_i)

                logger.info('{0:6d} ep Traj Err: translation: {1:.6f}, rotation: {2:.2f} deg, scale: {3:.2f}'.format(epoch_i,
                                                                                                                     eval_stats_tran['mean'],
                                                                                                                     eval_stats_rot['mean'],
                                                                                                                     eval_stats_scale['mean']))
                tqdm.write('{0:6d} ep Traj Err: translation: {1:.6f}, rotation: {2:.2f} deg, scale: {3:.2f}'.format(epoch_i,
                                                                                                                    eval_stats_tran['mean'],
                                                                                                                    eval_stats_rot['mean'],
                                                                                                                    eval_stats_scale['mean']))

                fxfy = focal_net(0)
                tqdm.write('Est fx: {0:.2f}, fy {1:.2f}, COLMAP focal: {2:.2f}'.format(fxfy[0].item(), fxfy[1].item(),
                                                                                       scene_train.focal))
                logger.info('Est fx: {0:.2f}, fy {1:.2f}, COLMAP focal: {2:.2f}'.format(fxfy[0].item(), fxfy[1].item(),
                                                                                        scene_train.focal))
                if torch.is_tensor(fxfy):
                    L1_focal = torch.abs(fxfy - scene_train.focal).mean().item()
                else:
                    L1_focal = np.abs(fxfy - scene_train.focal).mean()
                writer.add_scalar('eval/L1_focal', L1_focal, epoch_i)

        if epoch_i % args.eval_img_interval == 0 and epoch_i > 0:
            with torch.no_grad():
                eval_one_epoch_img(eval_c2ws, scene_train, model, focal_net, pose_param_net, my_devices,
                                   args, epoch_i, writer, rgb_act_fn)

                # save the latest model.
                save_checkpoint(epoch_i, model, optimizer_nerf, experiment_dir, ckpt_name='latest_nerf')
                save_checkpoint(epoch_i, focal_net, optimizer_focal, experiment_dir, ckpt_name='latest_focal')
                save_checkpoint(epoch_i, pose_param_net, optimizer_pose, experiment_dir, ckpt_name='latest_pose')
    return
