#!/usr/bin/env python3

import sys
import pickle
import argparse
import numpy as np
import cv2
import signal
import atexit
from typing import Tuple, List
from skimage.io import imread
from skimage.color import rgba2rgb
from scipy.optimize import least_squares

from utils import get_char, stat_dict, find_images
from utils.find_centers import find_centers
from utils.point_config import CalibrationConfig

call_off = False
atexit.register(lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))

def signal_handler(signal, frame):
    print("Break requested, exiting ...")
    global call_off
    call_off = True

def distort(uv_vec, fx, fy, cx, cy, k1, k2, k3, p1, p2, model="pinhole"):
    if len(uv_vec.shape) == 1:
        uv_vec = uv_vec.reshape((1, uv_vec.shape[0]))
    r_vec = np.empty(shape=(uv_vec.shape[0], uv_vec.shape[1]), dtype=float)
    if model == "pinhole":
        for i in range(len(uv_vec)):
            x = (uv_vec[i,0] - cx) / fx
            y = (uv_vec[i,1] - cy) / fy
            x2 = x*x
            y2 = y*y
            xy = x*y
            r2 = x2 + y2
            r4 = r2 * r2
            r6 = r4 * r2

            xDistort = x * (1 + k1*r2 + k2*r4 + k3*r6)
            yDistort = y * (1 + k1*r2 + k2*r4 + k3*r6)

            xDistort = xDistort + (2 * p1*xy + p2 * (r2 + 2*x2))
            yDistort = yDistort + (p1 * (r2 + 2*y2) + 2*p2*xy)
            r_vec[i,0] = xDistort * fx + cx
            r_vec[i,1] = yDistort * fy + cy
    elif model == "spherical":
        x = (uv_vec[:,0] - cx) / fx
        y = (uv_vec[:,1] - cy) / fy
        r = np.sqrt(x*x + y*y + 1)
        d = r*k1 + 1
        r_vec[:,0] = (x*fx) / d + cx
        r_vec[:,1] = (y*fy) / d + cy
    else:
        raise Exception("Distortion with {} model is not supported".format(model))

    return r_vec[0] if len(r_vec) == 1 else r_vec

def undistort(uv_vec, fx, fy, cx, cy, k1, k2, k3, p1, p2, model="pinhole"):
    if len(uv_vec.shape) == 1:
        uv_vec = uv_vec.reshape((1, uv_vec.shape[0]))
    r_vec = np.empty(shape=(uv_vec.shape[0], uv_vec.shape[1]), dtype=float)

    x = (uv_vec[:,0] - cx) / fx
    y = (uv_vec[:,1] - cy) / fy
    d2 = x*x + y*y

    if model == 'pinhole':
        # radial distortion ONLY
        rd = np.sqrt(d2)
        for i, d in enumerate(rd):
            rt = np.roots([k3, 0, k2, 0, k1, 0, 1, -d])
            r = np.real(rt[np.abs(np.imag(rt)) < 1e-6])
            
            if len(r) == 0:
                raise Exception('Failed to find a solution to undistorted radius')
            elif len(r) > 1:
                fr = r[np.argmin(np.abs(r-d))] / d
            else:
                fr = r[0] / d
            x[i] *= fr
            y[i] *= fr
        
        r_vec[:,0] = x*fx + cx
        r_vec[:,1] = y*fy + cy

    elif model == "spherical":
        w = (k1 + np.sqrt(1+(1-k1*k1)*d2)) / (1+d2)
        z = w-k1
        r_vec[:,0] = w*x*(fx/z) + cx
        r_vec[:,1] = w*y*(fy/z) + cy

    else:
        raise Exception("Undistortion with {} model is not supported".format(model))
    
    return r_vec[0] if len(r_vec) == 1 else r_vec

def _pinhole2spherical_helper(distortion:float,
                            image_points:np.ndarray, 
                            orig_points:np.ndarray, 
                            focal_lengths:Tuple[float, float],
                            image_center:Tuple[float, float]): 
    dp = distort(orig_points, 
                *focal_lengths, 
                *image_center, 
                distortion, 0, 0, 0, 0, 
                model="spherical")
    return (dp - image_points).ravel()

def pinhole2spherical(image_points:np.ndarray, 
                    orig_points:np.array,
                    focal_lengths:Tuple[float, float], 
                    image_center:Tuple[float, float],
                    pinhole_distortion_params:List[float]) -> float:
    ret = least_squares(fun=lambda x: _pinhole2spherical_helper(x,
                                                                image_points, 
                                                                orig_points, 
                                                                focal_lengths, 
                                                                image_center),
                        x0 = -pinhole_distortion_params[0]*2)
    return ret.x


def calibrate_cam(image_files, cfg):
    global call_off

    img_points = []
    world_points = []

    img_size = None

    quit_now = False
    i_name = 0
    ctrs = None
    verbose = cfg.verbose
    
    for img_name in image_files:
        if call_off or quit_now:
            break

        i_name += 1
        
        if verbose: print('Processing ' + img_name)
        img = imread(img_name)

        if len(img.shape) == 3 and img.shape[2] == 4:
            img = rgba2rgb(img)
            
        if img_size is None:
            img_size = img.shape
        else:
            if img.shape != img_size:
                raise Exception('Incompatible image size')

        def work_func():
            try:
                ctrs, num = find_centers(img, cfg)
            except Exception as ex:
                print(f'Error occuredred when processing {img_name}')
                print(ex)
                ctrs = None
                num = 0
            return ctrs, num


        ctrs, num = work_func()
        
        c = 'n'
        acpt = True
        quit_now = False

        while cfg.show_images or ctrs is None:
            print('Command: "r" to retry, "a" to accept, "e" to exclude, "q" to stop')
            c = get_char().lower()
        
            if c == 'r':
                ctrs, num = work_func()
            elif c == 'a':
                break
            elif c == 'e':
                acpt = False
                break
            elif c == 'q':
                quit_now = True
                break
        
        if acpt and not ctrs is None:
            img_points.append(ctrs)
            world_points.append(cfg.auto_points(num[1], num[0]))
            
        if verbose:
            print(' ')

    if len(img_points) < 2:
        print('Not enough images.')
        sys.exit(0)
    
    if verbose:
        print('Start calibration:')

    height, width = img_size[:2]
        
    if cfg.model.pixel_pitch is not None and cfg.model.focal_length_estimate is not None and cfg.model.initial_guess:
        px_size = cfg.model.pixel_pitch
        cx = width / 2.0
        cy = height / 2.0
        est_f = cfg.model.focal_length_estimate
        if isinstance(est_f, float) or isinstance(est_f, int):
            est_f = [est_f] * 2
        est_f = (est_f[0] / px_size, est_f[1] / px_size)
        mtx = np.array([[est_f[0], 0, cx], [0, est_f[1], cy], [0, 0, 1]], 
                    dtype = np.float32)
    else:
        mtx = None
                
    if cfg.model.use_rational_model:
        flags = cv2.CALIB_RATIONAL_MODEL + \
                cv2.CALIB_USE_INTRINSIC_GUESS # + cv2.CALIB_SAME_FOCAL_LENGTH + cv2.CALIB_FIX_PRINCIPAL_POINT
    else:
        flags = cv2.CALIB_USE_INTRINSIC_GUESS
        
    if not cfg.model.use_tangential_model:
        flags += cv2.CALIB_ZERO_TANGENT_DIST

    if cfg.model.fix_center:
        flags += cv2.CALIB_FIX_PRINCIPAL_POINT
        
    if cfg.model.fix_k[0]:
        flags += cv2.CALIB_FIX_K1
    if cfg.model.fix_k[1]:
        flags += cv2.CALIB_FIX_K2
    if cfg.model.fix_k[2]:
        flags += cv2.CALIB_FIX_K3
    if cfg.model.fix_k[3]:
        flags += cv2.CALIB_FIX_K4
    if cfg.model.fix_k[4]:
        flags += cv2.CALIB_FIX_K5
    if cfg.model.fix_k[5]:
        flags += cv2.CALIB_FIX_K6
    
    if not cfg.model.initial_guess:
        mtx = cv2.initCameraMatrix2D(world_points, img_points, (width, height))
        if verbose: 
            print('Initial matrix:')
            print(mtx)
    
    ret, mtx, dist_vec, rvecs, tvecs = cv2.calibrateCamera(world_points, 
                                                        img_points, 
                                                        (width, height), 
                                                        mtx, 
                                                        None,
                                                        flags = flags)

    fovx, fovy, f, pp, ap = cv2.calibrationMatrixValues(mtx, (width, height), px_size * width, px_size * height)
    
    if verbose:
        print(' ')
        print('Re-projection error is: {}'.format(ret))
        print('Camera Intrinsics are:\n {}'.format(mtx))
        print('Distortion vector is:\n {}'.format(dist_vec))


        print('H-FOV is {} degrees'.format(fovx))
        print('V-FOV is {} degrees'.format(fovy))
        print('Principal point is {}'.format(pp))
        print('Focal length is {}'.format(ap))
        print('Aspect ratio (fy/fx) is {}'.format(ap))

    all_image_points = []
    for ii in img_points: 
        all_image_points.extend(ii)
    all_image_points = np.array(all_image_points)
    focal_lengths = (mtx[0,0], mtx[1,1])
    image_center = (mtx[0,2], mtx[1,2])
    pinhole_distortion_params = [*dist_vec[0][:2], dist_vec[0][4], *dist_vec[0][2:4]]

    orig_pixels = undistort(all_image_points, 
                            *focal_lengths, 
                            *image_center, 
                            *pinhole_distortion_params, 
                            model="pinhole")
    ks = pinhole2spherical(image_points=all_image_points,
                            orig_points=orig_pixels,
                            focal_lengths=focal_lengths,
                            image_center=image_center,
                            pinhole_distortion_params=pinhole_distortion_params)    
    dp = distort(orig_pixels, 
                mtx[0,0], mtx[1,1], 
                mtx[0,2], mtx[1,2], 
                ks, 0, 0, 0, 0, 
                model="spherical")
    diff = all_image_points - dp
    sph_err = np.mean(np.sqrt(diff[:,0]**2 + diff[:,1]**2))

    return {'error': ret, 
            'matrix': mtx, 
            'distortion': dist_vec,
            'HFOV': fovx,
            'VFOV': fovy,
            'focal_length': f,
            'principal_point': pp,
            'aspect_ratio': ap,
            'spherical_distortion': ks,
            'spherical_err': sph_err}, quit_now

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
    
    mean, std = stat_dict(results)
    
    print(' ')
    print('Re-projection error is: {} +- {}'.format(mean['error'], std['error']))
    print('Camera Intrinsics are:\n {}  \n+-\n {}'.format(mean['matrix'], std['matrix']))
    print('Distortion vector is:\n {} \n+-\n {}'.format(mean['distortion'], std['distortion']))


    print('H-FOV is {} +- {} degrees'.format(mean['HFOV'], std['HFOV']))
    print('V-FOV is {} +- {} degrees'.format(mean['VFOV'], std['VFOV']))
    print('Principal point is {} +- {}'.format(mean['principal_point'], std['principal_point']))
    print('Focal length is {} +- {}'.format(mean['focal_length'], std['focal_length']))
    print('Aspect ratio (fy/fx) is {} +- {}'.format(mean['aspect_ratio'], std['aspect_ratio']))
    print('Spherical k1 is {} +- {}'.format(mean['spherical_distortion'], std['spherical_distortion']))
    print('spherical reprojection error is {} +- {}'.format(mean['spherical_err'], std['spherical_err']))

    if args.save is not None:
        with open(args.save, "wb") as f:
            pickle.dump({
                'matrix': mean['matrix'], 
                'distortion': mean['distortion'],
                'matrix_std': std['matrix'],
                'distortion_std': std['distortion']
            }, f)













