#!/usr/bin/env python3

import sys
import signal
import pickle
import argparse

import cv2
import numpy as np
from skimage.io import imread
from skimage.color import rgba2rgb
from scipy.optimize import least_squares

from utils import get_char, stat_dict, find_images
from utils.find_centers import find_centers
from utils.point_config import CalibrationConfig

CALL_OFF = False

def signal_handler(sig, frame):
    print("Break requested, exiting ...")
    global CALL_OFF
    CALL_OFF = True

signal.signal(signal.SIGINT, signal_handler)

def distotion_map(img_pts, img_size, p_0, k):
    r_vec = np.empty(img_pts.shape[0])
    r_vec[:] = (img_pts[:, 0] - p_0[0]) * (img_pts[:, 0] - p_0[0]) / img_size[0] / img_size[0] +\
                 (img_pts[:, 1] - p_0[1]) * (img_pts[:, 1] - p_0[1]) / img_size[1] / img_size[1]

    delta = np.ones(len(r_vec))
    accu = r_vec.copy()

    for k_i in k:
        delta[:] = delta[:] + k_i * accu[:]
        accu[:] = accu[:] * r_vec[:]

    map_pts = np.empty(img_pts.shape)

    map_pts[:, 0] = (img_pts[:, 0] - p_0[0]) * delta[:] + p_0[0]
    map_pts[:, 1] = (img_pts[:, 1] - p_0[1]) * delta[:] + p_0[1]

    return map_pts

def reproject(wld_pts, s_mat, r_mat, t_vec, p_0, k, img_size):
    h_mat = np.eye(3, dtype=float)
    h_mat[:2, :2] = np.matmul(s_mat, r_mat)
    h_mat[:2, 2] = np.matmul(s_mat, t_vec)
    img_pts = np.matmul(wld_pts, h_mat.T)[:,:2]

    if p_0 is None:
        p_0 = [img_size[0] / 2.0, img_size[1] / 2.0]

    return distotion_map(img_pts, img_size, p_0, k)

def pars2x(s_mat, p_0, k, r_mat, t_vec, cfg):
    pars = [s_mat[0,0], s_mat[0,1], s_mat[1,1]]
    if not cfg.model.fix_center:
        pars.extend(p_0)
    pars.extend([k_i for i, k_i in enumerate(k) if not cfg.model.fix_k[i]])
    for i, r_i in enumerate(r_mat):
        t_i = t_vec[i]
        pars.extend([r_i[0,0], r_i[0,1], r_i[1,0], r_i[1,1], t_i[0], t_i[1]])
    return np.array(pars)

def x2pars(pars, cfg):
    s_mat = np.zeros((2,2), dtype=float)
    p_0 = None
    k = []

    s_mat[0,0], s_mat[0,1], s_mat[1,1] = pars[0:3]
    cnt = 3

    if not cfg.model.fix_center:
        p_0 = [pars[3], pars[4]]
        cnt = 5

    for i, fix_ki in enumerate(cfg.model.fix_k):
        if not fix_ki:
            k.append(pars[cnt+i])
            cnt += 1

    r_mat = []
    t_vec = []
    for _i in range(cnt, len(pars), 6):
        _r = np.zeros((2,2), dtype=float)
        _t = np.zeros(2, dtype=float)
        _r[0,0], _r[0,1], _r[1,0], _r[1,1], _t[0], _t[1] = pars[_i:_i+6]
        r_mat.append(_r)
        t_vec.append(_t)

    return s_mat, p_0, k, r_mat, t_vec

def reproject_error(wld_pts, img_pts, pars, img_size, cfg):
    s_mat, p_0, k, r_mat, t_vec = x2pars(pars, cfg)
    err = []
    for i, pts_i in enumerate(img_pts):
        rep_pts = reproject(wld_pts, s_mat, r_mat[i], t_vec[i], p_0, k, img_size)
        err.append(pts_i - rep_pts)
    return np.array(err).ravel()

def rep_error(ctrs, rep_ctrs):
    diff = rep_ctrs - ctrs
    return np.sqrt(np.mean(diff * diff))

def calibrate_cam(image_files, cfg):
    np.set_printoptions(precision=3)

    img_points = []
    world_points = []
    h_matrix = []

    img_size = None

    quit_now = False
    ctrs = None
    verbose = cfg.verbose

    err_list = []

    def work_func(img, img_name):
        nonlocal cfg

        mtx = None
        err = None
        homo_wp = None
        num = None

        try:
            ctrs, num = find_centers(img, cfg)
            assert (num[1]>0 and num[0]>0), "No points found"
            homo_wp = cfg.auto_points(num[1], num[0])
            homo_wp[:,2] = 1
            mtx, _ = cv2.findHomography(homo_wp, ctrs)
            rep_ctrs = (np.einsum('ij, kj', mtx, homo_wp).T)[:,:2]
            err = rep_error(ctrs, rep_ctrs)
        except Exception as ex:
            print(f'Error occuredred when processing {img_name}')
            print(ex)
            ctrs = None
        return ctrs, homo_wp, mtx, err, num

    for img_name in image_files:
        if quit_now:
            break

        if cfg.verbose:
            print('Processing ' + img_name)
        img = imread(img_name)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = rgba2rgb(img)

        if img_size is None:
            img_size = img.shape[:2][::-1]
            width, height = img_size
            c_x = width / 2.0
            c_y = height / 2.0
            img_ctr = np.array((c_x, c_y))
        else:
            if img.shape[:2][::-1] != img_size:
                raise RuntimeError('Incompatible image size')

        order = 0

        ctrs, homo_wp, mtx, err, num = work_func(img, img_name)

        cmd = 'n'
        acpt = True
        quit_now = False

        if CALL_OFF:
            break

        if cfg.show_images:
            while True:
                print('Command: "r" to retry, "a" to accept, "e" to exclude, "q" to stop')
                cmd = get_char().lower()

                if cmd == 'r':
                    print('Retry ...')
                    order += 1
                    ctrs, homo_wp, mtx, err, num = work_func(img, img_name)
                elif cmd == 'a':
                    break
                elif cmd == 'e':
                    acpt = False
                    break
                elif cmd == 'q':
                    quit_now = True
                    acpt =False
                    break
        else:
            acpt = False
            for _t in range(1, cfg.num_trials):
                order = _t
                if CALL_OFF:
                    break
                elif (not ctrs is None) and err < cfg.reprojection_error_threshold:
                    acpt = True
                    break
                else:
                    if ctrs is None:
                        print('No points found. Retry ...')
                    else:
                        print(f'Error {err} is too large. Retry ...')
                    ctrs, homo_wp, mtx, err, num = work_func(img, img_name)

        if acpt and not ctrs is None:
            img_points.append(ctrs)
            world_points.append(cfg.auto_points(num[1], num[0]))
            if verbose:
                print('Homography matrix:')
                print(mtx)
                print(f"Reprojection error is: {err}")
            h_matrix.append(mtx)
            err_list.append(err)

        if verbose:
            print()

    if len(img_points) < 4:
        print('Not enough images.')
        sys.exit(0)
    else:
        print(f'{len(img_points)} images processed out of {len(image_files)} in total')
        if verbose:
            print(f'Image size is {img_size}')

    if verbose:
        print('Start calibration:')

    g_matrices = []
    for _h_mat in enumerate(h_matrix):
        _det = np.linalg.det(_h_mat)
        g_matrices.append(
            [
                _h_mat[0,0] * _h_mat[0,0] + _h_mat[0,1] * _h_mat[0,1],
                2.0 * (_h_mat[0,0] * _h_mat[1,0] + _h_mat[0,1] * _h_mat[1,1]),
                _h_mat[1,1] * _h_mat[1,1] + _h_mat[1,0] * _h_mat[1,0],
                _det * _det
            ]
        )

    g_matrices = np.array(g_matrices)
    b_sum = np.sum(
        np.matmul(
            np.linalg.inv(np.matmul(g_matrices.T, g_matrices)),
            g_matrices.T
        ), axis = 1
    )
    #print(G)
    #print(b)
    #print(np.matmul(G, b))

    b_mat_inv = np.linalg.inv(np.array([[b_sum[0], b_sum[1]], [b_sum[1], b_sum[2]]]))
    #print(B)
    #print(b[0]*b[2] - b[1]*b[1])
    #print(b[3])

    s_mat = np.linalg.cholesky(b_mat_inv).T
    s_inv = np.linalg.inv(s_mat)

    if verbose:
        print('Initial camera matrix is')
        print(s_mat)

    s_0 = s_mat.copy()

    r_matrix = list(map(lambda H: np.matmul(s_inv, H[:2,:2]), h_matrix))
    t_vec = list(map(lambda H: np.matmul(s_inv, H[:2,2]), h_matrix))

    err_list = np.array(err_list)
    err0 = np.mean(err_list)
    if verbose:
        print(f'Before nonlinear fit, reprojection error is {err0}+-{np.std(err_list)}')

    pars_init = pars2x(s_mat, img_ctr, [0], r_matrix, t_vec, cfg)
    err_func = lambda x: reproject_error(homo_wp, img_points, x, img_size, cfg)

    if verbose:
        print('Start least square fit ...')
    rslt = least_squares(err_func, pars_init, verbose=1 if verbose else 0)

    s_mat, p_0, k, r_matrix, t_vec = x2pars(rslt.x, cfg)

    print('Final camera matrix is')
    print(s_mat)

    if p_0 is None:
        p_0 = np.array([c_x, c_y])
    print(f'Principal point is: ({p_0[0]}, {p_0[1]})')

    print(f'Distotion vector is {k}')

    err_list = []
    for _i in range(len(h_matrix)):
        rep_pts = reproject(homo_wp, s_mat, r_matrix[_i], t_vec[_i], p_0, k, img_size)
        err_list.append(rep_error(img_points[_i], rep_pts))
    err_list = np.array(err_list)

    err = np.mean(err_list)
    print(f'After nonlinear fit, reprojection error is {err}+-{np.std(err_list)}')

    return {
        'matrix': s_mat,
        'error': err,
        'init_matrix': s_0,
        'init_error': err0,
        'principal_point': p_0, 
        'distortion': k
    }, quit_now

def parse_args():
    par = argparse.ArgumentParser()
    par.add_argument("images", metavar='calibration images',
                   help="path to calibration images")
    par.add_argument("-c", "--config", dest="config", metavar='',
                   help="configuration file", required=True)
    par.add_argument("-o", "--out", dest="save", metavar='',
                   help="save results to this pickel file", required=False)

    if len(sys.argv) == 1:
        sys.argv.append("-h")
    return par.parse_args()

def main(args):
    s_names = sorted(find_images(args.images))
    if len(s_names) < 1:
        print('No images')
        sys.exit(1)

    cfg = CalibrationConfig()
    cfg.read_config(args.config)
    cfg.world_points = cfg.auto_points()

    results = []
    n_round = cfg.bootstrap_rounds
    n_img = int(round(cfg.bootstrap_ratio * len(s_names)))

    signal.signal(signal.SIGINT, signal_handler)

    quit_now = False
    for _ii in range(n_round):
        if CALL_OFF or quit_now:
            break
        print(f'Round {_ii}')
        idx = np.random.choice(len(s_names), n_img, replace=False)
        rslt, quit_now = calibrate_cam([s_names[i] for i in idx], cfg)
        results.append(rslt)
        print()

    mean, std = stat_dict(results)

    print()
    print(f'Re-projection error is: {mean["error"]} +- {std["error"]}')
    print(f'Camera Intrinsics are:\n {mean["matrix"]}  \n+-\n {std["matrix"]}')
    print(f'Initial re-projection error is: {mean["init_error"]} +- {std["init_error"]}')
    print(f'Initial camera intrinsics are:\n {mean["init_matrix"]}  \n+-\n {std["init_matrix"]}')
    print(f'Distortion vector is:\n {mean["distortion"]} \n+-\n {std["distortion"]}')
    print(f'Principal point is {mean["principal_point"]} +- {std["principal_point"]}')

    if args.save is not None:
        with open(args.save, "wb") as pk_f:
            pickle.dump({
                'matrix': mean['matrix'], 
                'distortion': mean['distortion'],
                'matrix_std': std['matrix'],
                'distortion_std': std['distortion'],
                'principal_point': mean['principal_point'],
                'principal_point_std': std['principal_point']
            }, pk_f)

if __name__ == '__main__':
    main(parse_args())
