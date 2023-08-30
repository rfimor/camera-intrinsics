#!/usr/bin/env python3

import sys
import pickle
import argparse
from typing import Tuple, List
import signal
import atexit

import numpy as np
import cv2
from skimage.io import imread
from skimage.color import rgba2rgb
from scipy.optimize import least_squares

from utils import get_char, stat_dict, find_images
from utils.find_centers import find_centers
from utils.point_config import CalibrationConfig

CALL_OFF = False
atexit.register(lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))

def signal_handler(sig, frame):
    '''
        User cancel
    '''
    global CALL_OFF
    print("Break requested, exiting ...")
    CALL_OFF = True

def distort(
    uv_vec : np.ndarray,
    f_x, f_y,
    c_x, c_y,
    k_1, k_2, k_3,
    p_1, p_2,
    model : str = "pinhole"
):
    '''
        compute pixel values after distortion
    '''
    if len(uv_vec.shape) == 1:
        uv_vec = uv_vec.reshape((1, uv_vec.shape[0]))
    r_vec = np.empty(shape=(uv_vec.shape[0], uv_vec.shape[1]), dtype=float)
    if model == "pinhole":
        for i in range(len(uv_vec)):
            _x = (uv_vec[i,0] - c_x) / f_x
            _y = (uv_vec[i,1] - c_y) / f_y
            x_2 = _x*_x
            y_2 = _y*_y
            x_y = _x*_y
            r_2 = x_2 + y_2
            r_4 = r_2 * r_2
            r_6 = r_4 * r_2

            x_dist = _x * (1 + k_1*r_2 + k_2*r_4 + k_3*r_6)
            y_dist = _y * (1 + k_1*r_2 + k_2*r_4 + k_3*r_6)

            x_dist += (2 * p_1*x_y + p_2 * (r_2 + 2*x_2))
            y_dist += (p_1 * (r_2 + 2*y_2) + 2*p_2*x_y)
            r_vec[i,0] = x_dist * f_x + c_x
            r_vec[i,1] = y_dist * f_y + c_y
    elif model == "spherical":
        _x = (uv_vec[:,0] - c_x) / f_x
        _y = (uv_vec[:,1] - c_y) / f_y
        _r = np.sqrt(_x*_x + _y*_y + 1)
        _d = _r*k_1 + 1
        r_vec[:,0] = (_x*f_x) / _d + c_x
        r_vec[:,1] = (_y*f_y) / _d + c_y
    else:
        raise ValueError(f"Distortion with {model} model is not supported")

    return r_vec[0] if len(r_vec) == 1 else r_vec

def undistort(
    uv_vec : np.ndarray,
    f_x, f_y,
    c_x, c_y,
    k_1, k_2, k_3,
    p_1, p_2,
    model : str = "pinhole"
):
    '''
        computer pixel values after undistortion
    '''
    if len(uv_vec.shape) == 1:
        uv_vec = uv_vec.reshape((1, uv_vec.shape[0]))
    r_vec = np.empty(shape=(uv_vec.shape[0], uv_vec.shape[1]), dtype=float)

    _x = (uv_vec[:,0] - c_x) / f_x
    _y = (uv_vec[:,1] - c_y) / f_y
    d_2 = _x*_x + _y*_y

    if model == 'pinhole':
        # radial distortion ONLY
        r_d = np.sqrt(d_2)
        for i, _d in enumerate(r_d):
            r_t = np.roots([k_3, 0, k_2, 0, k_1, 0, 1, -_d])
            _r = np.real(r_t[np.abs(np.imag(r_t)) < 1e-6])

            if len(_r) == 0:
                raise RuntimeError('Failed to find a solution to undistorted radius')
            elif len(_r) > 1:
                f_r = _r[np.argmin(np.abs(_r-_d))] / _d
            else:
                f_r = _r[0] / _d
            _x[i] *= f_r
            _y[i] *= f_r

        r_vec[:,0] = _x*f_x + c_x
        r_vec[:,1] = _y*f_y + c_y

    elif model == "spherical":
        _w = (k_1 + np.sqrt(1+(1-k_1*k_1)*d_2)) / (1+d_2)
        _z = _w-k_1
        r_vec[:,0] = _w*_x*(f_x/_z) + c_x
        r_vec[:,1] = _w*_y*(f_y/_z) + c_y

    else:
        raise ValueError(f"Undistortion with {model} model is not supported")

    return r_vec[0] if len(r_vec) == 1 else r_vec

def _pinhole2spherical_helper(
    distortion : float,
    image_points : np.ndarray,
    orig_points : np.ndarray,
    focal_lengths : Tuple[float, float],
    image_center:Tuple[float, float]
) -> np.ndarray:
    d_p = distort(
        orig_points,
        *focal_lengths,
        *image_center,
        distortion,
        0, 0, 0, 0,
        model="spherical"
    )
    return (d_p - image_points).ravel()

def pinhole2spherical(
    image_points : np.ndarray,
    orig_points : np.array,
    focal_lengths : Tuple[float, float],
    image_center : Tuple[float, float],
    pinhole_distortion_params : List[float]
) -> float:
    '''
        given pinhole-model parameters,
        find the best fitting spherical model parameters 
    '''
    ret = least_squares(
            fun=lambda x: _pinhole2spherical_helper(
                x,
                image_points,
                orig_points,
                focal_lengths,
                image_center
            ),
            x0 = -pinhole_distortion_params[0]*2
        )
    return ret.x


def calibrate_cam(image_files, cfg):
    '''
        calibration process
    '''
    img_points = []
    world_points = []

    img_size = None

    quit_now = False
    i_name = 0
    ctrs = None
    verbose = cfg.verbose

    def work_func(img, img_name):
        try:
            ctrs, num = find_centers(img, cfg)
        except Exception as ex:
            print(f'Error occuredred when processing {img_name}: {ex}')
            ctrs = None
            num = 0
        return ctrs, num

    for img_name in image_files:
        if CALL_OFF or quit_now:
            break

        i_name += 1

        if verbose:
            print('Processing', img_name)
        img = imread(img_name)

        if len(img.shape) == 3 and img.shape[2] == 4:
            img = rgba2rgb(img)

        if img_size is None:
            img_size = img.shape
        else:
            if img.shape != img_size:
                raise ValueError('Incompatible image size')

        ctrs, num = work_func(img, img_name)

        cmd = 'n'
        acpt = True
        quit_now = False

        while cfg.show_images or ctrs is None:
            print('Command: "r" to retry, "a" to accept, "e" to exclude, "q" to stop')
            cmd = get_char().lower()

            if cmd == 'r':
                ctrs, num = work_func(img, img_name)
            elif cmd == 'a':
                break
            elif cmd == 'e':
                acpt = False
                break
            elif cmd == 'q':
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

    if cfg.model.pixel_pitch is not None and \
        cfg.model.focal_length_estimate is not None and \
        cfg.model.initial_guess:
        px_size = cfg.model.pixel_pitch
        c_x = width / 2.0
        c_y = height / 2.0
        est_f = cfg.model.focal_length_estimate
        if isinstance(est_f, float) or isinstance(est_f, int):
            est_f = [est_f] * 2
        est_f = (est_f[0] / px_size, est_f[1] / px_size)
        mtx = np.array([
                [est_f[0], 0, c_x],
                [0, est_f[1], c_y],
                [0, 0, 1]
                ], dtype = np.float32
            )
    else:
        mtx = None

    if cfg.model.use_rational_model:
        flags = cv2.CALIB_RATIONAL_MODEL + \
                cv2.CALIB_USE_INTRINSIC_GUESS
        # + cv2.CALIB_SAME_FOCAL_LENGTH + cv2.CALIB_FIX_PRINCIPAL_POINT
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

    ret, mtx, dist_vec, _, _ = cv2.calibrateCamera(
        world_points,
        img_points,
        (width, height),
        mtx,
        None,
        flags = flags
    )

    fovx, fovy, focal, p_pt, astp = cv2.calibrationMatrixValues(
        mtx,
        (width, height),
        px_size * width,
        px_size * height
    )

    if verbose:
        print(' ')
        print(f'Re-projection error is: {ret}')
        print(f'Camera Intrinsics are:\n {mtx}')
        print(f'Distortion vector is:\n {dist_vec}')


        print(f'H-FOV is {fovx} degrees')
        print(f'V-FOV is {fovy} degrees')
        print(f'Principal point is {p_pt}')
        print(f'Focal length is {focal}')
        print(f'Aspect ratio (fy/fx) is {astp}')

    all_image_points = []
    for i in img_points:
        all_image_points.extend(i)
    all_image_points = np.array(all_image_points)
    focal_lengths = (mtx[0,0], mtx[1,1])
    image_center = (mtx[0,2], mtx[1,2])
    pinhole_distortion_params = [
        *dist_vec[0][:2],
        dist_vec[0][4],
        *dist_vec[0][2:4]
    ]

    orig_pixels = undistort(
        all_image_points,
        *focal_lengths,
        *image_center,
        *pinhole_distortion_params,
        model="pinhole"
    )
    k_s = pinhole2spherical(
        image_points=all_image_points,
        orig_points=orig_pixels,
        focal_lengths=focal_lengths,
        image_center=image_center,
        pinhole_distortion_params=pinhole_distortion_params
    )
    d_p = distort(
        orig_pixels,
        mtx[0,0], mtx[1,1],
        mtx[0,2], mtx[1,2],
        k_s, 0, 0, 0, 0,
        model="spherical"
    )
    diff = all_image_points - d_p
    sph_err = np.mean(np.sqrt(diff[:,0]**2 + diff[:,1]**2))

    return {'error': ret,
            'matrix': mtx, 
            'distortion': dist_vec,
            'HFOV': fovx,
            'VFOV': fovy,
            'focal_length': focal,
            'principal_point': p_pt,
            'aspect_ratio': astp,
            'spherical_distortion': k_s,
            'spherical_err': sph_err}, quit_now

def parse_args():
    '''
        parse arguments
    '''
    _p = argparse.ArgumentParser()
    _p.add_argument("images", metavar='calibration images',
                    help="path to calibration images")
    _p.add_argument("-c", "--config", dest="config", metavar='',
                    help="configuration file", required=True)
    _p.add_argument("-o", "--out", dest="save", metavar='',
                    help="save results to this pickel file", required=False)
    _p.add_argument("--show", dest="show", action='store_true',
                    help="show visualization", required=False)

    if len(sys.argv) == 1:
        sys.argv.append("-h")
    return _p.parse_args()

def main(args):
    '''
        main function
    '''
    s_names = sorted(find_images(args.images))
    if len(s_names) < 1:
        print('No images')
        sys.exit(1)

    cfg = CalibrationConfig()
    cfg.read_config(args.config)
    cfg.world_points = cfg.auto_points()
    cfg.show_images = args.show

    results = []
    n_round = cfg.bootstrap_rounds
    n_img = int(round(cfg.bootstrap_ratio * len(s_names)))

    signal.signal(signal.SIGINT, signal_handler)

    quit_now = False
    for i in range(n_round):
        if CALL_OFF or quit_now:
            break
        print(f'Round {i}')
        idx = np.random.choice(len(s_names), n_img, replace=False)
        _r, quit_now = calibrate_cam([s_names[i] for i in idx], cfg)
        results.append(_r)

    mean, std = stat_dict(results)

    print(' ')
    print(f"Re-projection error is: {mean['error']} +- {std['error']}")
    print(f"Camera Intrinsics are:\n {mean['matrix']}  \n+-\n {std['matrix']}")
    print(f"Distortion vector is:\n {mean['distortion']} \n+-\n {std['distortion']}")


    print(f"H-FOV is {mean['HFOV']} +- {std['HFOV']} degrees")
    print(f"V-FOV is {mean['VFOV']} +- {std['VFOV']} degrees")
    print(f"Principal point is {mean['principal_point']} +- {std['principal_point']}")
    print(f"Focal length is {mean['focal_length']} +- {std['focal_length']}")
    print(f"Aspect ratio (fy/fx) is {mean['aspect_ratio']} +- {std['aspect_ratio']}")
    print(f"Spherical k1 is {mean['spherical_distortion']} +- {std['spherical_distortion']}")
    print(f"spherical reprojection error is {mean['spherical_err']} +- {std['spherical_err']}")

    if args.save is not None:
        with open(args.save, "wb") as _f:
            pickle.dump({
                'matrix': mean['matrix'], 
                'distortion': mean['distortion'],
                'matrix_std': std['matrix'],
                'distortion_std': std['distortion']
            },
            _f
        )

if __name__ == '__main__':
    main(parse_args())
