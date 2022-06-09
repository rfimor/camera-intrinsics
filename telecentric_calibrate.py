#!/usr/bin/env python3

import numpy as np
import sys
import cv2
import signal
import pickle
import argparse
from skimage.io import imread
from scipy.optimize import least_squares

from utils import get_char, stat_dict, find_images
from utils.find_centers import find_centers
from utils.point_config import CalibrationConfig

call_off = False

def signal_handler(signal, frame):
    print("Break requested, exiting ...")
    global call_off
    call_off = True

signal.signal(signal.SIGINT, signal_handler)

def distotion_map(img_pts, img_size, p0, k):
    r_vec = np.empty(img_pts.shape[0])
    r_vec[:] = (img_pts[:, 0] - p0[0]) * (img_pts[:, 0] - p0[0]) / img_size[0] / img_size[0] +\
                 (img_pts[:, 1] - p0[1]) * (img_pts[:, 1] - p0[1]) / img_size[1] / img_size[1]

    delta = np.ones(len(r_vec))
    accu = r_vec.copy()

    for ii in range(len(k)):
        delta[:] = delta[:] + k[ii] * accu[:]
        accu[:] = accu[:] * r_vec[:]
    
    map_pts = np.empty(img_pts.shape)
    
    map_pts[:, 0] = (img_pts[:, 0] - p0[0]) * delta[:] + p0[0]
    map_pts[:, 1] = (img_pts[:, 1] - p0[1]) * delta[:] + p0[1]
    
    return map_pts

def reproject(wld_pts, S, R, t, p0, k, img_size):
    H = np.eye(3, dtype=float)
    H[:2, :2] = np.matmul(S, R)
    H[:2, 2] = np.matmul(S, t)
    img_pts = np.matmul(wld_pts, H.T)[:,:2]

    if p0 is None:
        p0 = [img_size[0] / 2.0, img_size[1] / 2.0]

    return distotion_map(img_pts, img_size, p0, k)

def pars2x(S, p0, k, R_mat, t_vec, cfg):
    pars = [S[0,0], S[0,1], S[1,1]]
    if not cfg.model.fix_center:
        pars.extend(p0)
    for ii in range(len(k)):
        if not cfg.model.fix_k[ii]: 
            pars.append(k[ii])
    for ii in range(len(R_mat)):
        R = R_mat[ii]
        t = t_vec[ii]
        pars.extend([R[0,0], R[0,1], R[1,0], R[1,1], t[0], t[1]])
    return np.array(pars)
    
def x2pars(pars, cfg):
    S = np.zeros((2,2), dtype=float)
    p0 = None
    k = []
    
    S[0,0], S[0,1], S[1,1] = pars[0:3]
    cnt = 3
    
    if not cfg.model.fix_center:
        p0 = [pars[3], pars[4]]
        cnt = 5

    for ii in range(len(cfg.model.fix_k)):
        if not cfg.model.fix_k[ii]:
            k.append(pars[cnt+ii])
            cnt += 1
    
    R_mat = []
    t_vec = []
    for ii in range(cnt, len(pars), 6):
        R = np.zeros((2,2), dtype=float)
        t = np.zeros(2, dtype=float)
        R[0,0], R[0,1], R[1,0], R[1,1], t[0], t[1] = pars[ii:ii+6]
        R_mat.append(R)
        t_vec.append(t)

    return S, p0, k, R_mat, t_vec
    
def reproject_error(wld_pts, img_pts, pars, img_size, cfg):
    S, p0, k, R_mat, t_vec = x2pars(pars, cfg)
    err = []
    for ii in range(len(img_pts)):
        rep_pts = reproject(wld_pts, S, R_mat[ii], t_vec[ii], p0, k, img_size)
        err.append(img_pts[ii] - rep_pts)
    return np.array(err).ravel()
    
def rep_error(ctrs, rep_ctrs):
    diff = rep_ctrs - ctrs
    return np.sqrt(np.mean(diff * diff))

def calibrate_cam(image_files, cfg):
    global call_off
    
    np.set_printoptions(precision=3)

    img_points = []
    world_points = []
    H_matrix = []

    img_size = None

    quit_now = False
    ctrs = None
    verbose = cfg.verbose

    err_list = []
                
    for img_name in image_files:
        if quit_now:
            break
        
        if cfg.verbose:
            print('Processing ' + img_name)
        img = imread(img_name)
                    
        if img_size is None:
            img_size = img.shape[:2][::-1]
            width, height = img_size
            cx = width / 2.0
            cy = height / 2.0
            img_ctr = np.array((cx, cy))
        else:
            if img.shape[:2][::-1] != img_size:
                raise Exception('Imcompatible image size')
        
        order = 0
        def work_func():
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
         
        ctrs, homo_wp, mtx, err, num = work_func()
      
        c = 'n'
        acpt = True
        quit_now = False
        
        if call_off: break
        
        if cfg.show_images:
            while(True):
                print('Command: "r" to retry, "a" to accept, "e" to exclude, "q" to stop')
                c = get_char().lower()
        
                if c == 'r':
                    print('Retry ...')
                    order += 1
                    ctrs, homo_wp, mtx, err, num = work_func()
                elif c == 'a':
                    break
                elif c == 'e':
                    acpt = False
                    break
                elif c == 'q':
                    quit_now = True
                    acpt =False
                    break
        else:
            acpt = False
            for tt in range(1, cfg.num_trials):
                order = tt
                if call_off:
                    break
                elif (not ctrs is None) and err < cfg.reprojection_error_threshold:
                    acpt = True
                    break
                else:
                    if ctrs is None:
                        print('No points found. Retry ...')
                    else:
                        print('Error {} is too large. Retry ...'.format(err))
                    ctrs, homo_wp, mtx, err, num = work_func()                    
    
        if acpt and not ctrs is None:
            img_points.append(ctrs)
            world_points.append(cfg.auto_points(num[1], num[0]))
            if verbose: 
                print('Homography matrix:')
                print(mtx)
                print("Reprojection error is: {}".format(err))
            H_matrix.append(mtx)
            err_list.append(err)
             
        if verbose: print(' ')

    if len(img_points) < 4:
        print('Not enough images.')
        sys.exit(0)
    else:
        print('{} images processed out of {} in total'.format(len(img_points), len(image_files)))
        if verbose:
            print('Image size is {}'.format(img_size))
    
    if verbose: print('Start calibration:')
    
    G = []
    for ii in range(len(H_matrix)):
        H = H_matrix[ii]
        d = np.linalg.det(H)
        G.append([H[0,0]*H[0,0] + H[0,1]*H[0,1], 
                2.0*(H[0,0]*H[1,0] + H[0,1]*H[1,1]),
                H[1,1]*H[1,1] + H[1,0]*H[1,0],
                d*d])

    G = np.array(G)
    b = np.sum(np.matmul(np.linalg.inv(np.matmul(G.T, G)), G.T), axis = 1)
    #print(G)
    #print(b)
    #print(np.matmul(G, b))
    
    B = np.linalg.inv(np.array([[b[0], b[1]], [b[1], b[2]]]))
    #print(B)
    #print(b[0]*b[2] - b[1]*b[1])
    #print(b[3])
    
    S = np.linalg.cholesky(B).T
    S_inv = np.linalg.inv(S)
            
    if verbose:
        print('Initial camera matrix is')
        print(S)

    S0 = S.copy()

    R_matrix = list(map(lambda H: np.matmul(S_inv, H[:2,:2]), H_matrix))
    t_vec = list(map(lambda H: np.matmul(S_inv, H[:2,2]), H_matrix))
    
    err_list = np.array(err_list)
    err0 = np.mean(err_list)
    if verbose:
        print('Before nonlinear fit, reprojection error is {}+-{}'.format(err0, np.std(err_list)))
    
    pars_init = pars2x(S, img_ctr, [0], R_matrix, t_vec, cfg)
    err_func = lambda x: reproject_error(homo_wp, img_points, x, img_size, cfg)
    
    if verbose:
        print('Start least square fit ...')
    rslt = least_squares(err_func, pars_init, verbose=1 if verbose else 0)

    S, p0, k, R_matrix, t_vec = x2pars(rslt.x, cfg)
    
    print('Final camera matrix is')
    print(S)
        
    if p0 is None: p0 = np.array([cx, cy])
    print('Principal point is: ({}, {})'.format(p0[0], p0[1]))
        
    print('Distotion vector is {}'.format(k))

    err_list = []
    for ii in range(len(H_matrix)):
        rep_pts = reproject(homo_wp, S, R_matrix[ii], t_vec[ii], p0, k, img_size)
        err_list.append(rep_error(img_points[ii], rep_pts))
    err_list = np.array(err_list)

    err = np.mean(err_list)
    print('After nonlinear fit, reprojection error is {}+-{}'.format(err, np.std(err_list)))

    return {'matrix': S, 
            'error': err,
            'init_matrix': S0,
            'init_error': err0,
            'principal_point': p0, 
            'distortion': k
            }, quit_now

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("images", metavar='calibration images', help="path to calibration images")
    p.add_argument("-c", "--config", dest="config", metavar='', help="configuration file", required=True)
    p.add_argument("-o", "--out", dest="save", metavar='', help="save results to this pickel file", required=False)

    if len(sys.argv) == 1:
        sys.argv.append("-h")
    args = p.parse_args()

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
    for ii in range(n_round):
        if call_off or quit_now:
            break
        print('Round {}'.format(ii))
        idx = np.random.choice(len(s_names), n_img, replace=False)
        r, quit_now = calibrate_cam([s_names[i] for i in idx], cfg)
        results.append(r)
        print()
    
    mean, std = stat_dict(results)

    print(' ')
    print('Re-projection error is: {} +- {}'.format(mean['error'], std['error']))
    print('Camera Intrinsics are:\n {}  \n+-\n {}'.format(mean['matrix'], std['matrix']))
    print('Initial re-projection error is: {} +- {}'.format(mean['init_error'], std['init_error']))
    print('Initial camera intrinsics are:\n {}  \n+-\n {}'.format(mean['init_matrix'], std['init_matrix']))
    print('Distortion vector is:\n {} \n+-\n {}'.format(mean['distortion'], std['distortion']))
    print('Principal point is {} +- {}'.format(mean['principal_point'], std['principal_point']))
   
    if args.save is not None:
        with open(args.save, "wb") as f:
            pickle.dump({
                'matrix': mean['matrix'], 
                'distortion': mean['distortion'],
                'matrix_std': std['matrix'],
                'distortion_std': std['distortion'],
                'principal_point': mean['principal_point'],
                'principal_point_std': std['principal_point']
            }, f)

