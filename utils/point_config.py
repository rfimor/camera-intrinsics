import json
from easydict import EasyDict
import numpy as np

AREA_THRES = None
ECC_THRES = None
AVG_AREA_THRES = None
OPEN_KER_RATIO = None
CLOSE_KER_RATIO = None
REL_DIST_THRES = 0.33
ANGLE_THRES = 0.15
REPRO_ERR_THRES = 1.0

class CalibrationConfig(object):
    def __init__(self):
        # circle detection
        _det = EasyDict()
        _det.radius_etimate = 1

        # 3: RGB sum
        # 0: R only
        # 1: G only
        # 2: B only
        _det.channel_selection = "all"

        _det.threshold = None
        _det.adaptive_threshold = False
        _det.inverse_threshold = False

        _det.open_kernel = OPEN_KER_RATIO
        _det.close_kernel = CLOSE_KER_RATIO

        _det.area_thresholds = AREA_THRES
        _det.eccentricity_threshold = ECC_THRES
        _det.average_area_threshold = AVG_AREA_THRES

        self.circle_detection = _det

        # pattern detection
        _det = EasyDict()
        _det.world_points = None

        _det.skip_header = [0, 0]
        _det.num_points = [1, 1]
        _det.spacing = [1,1]

        # location of zero in the WCS
        # top-left
        # top-right
        # bottom-left
        # bottom-right
        _det.zero_location = "top-left"

        _det.allow_less_circles = False
        _det.relative_distance_threshold = REL_DIST_THRES
        _det.angle_threshold = ANGLE_THRES
        self.pattern_detection = _det

        # camera model
        _model = EasyDict()
        _model.initial_guess = True
        _model.use_rational_model = False

        _model.use_tangential_model = False
        _model.fix_k = [False, ] * 6
        _model.fix_center = False
        _model.magnification = None

        _model.focal_length_estimate = None
        _model.pixel_pitch = None

        self.model = _model

        self.num_trials = 3
        self.show_images = False
        self.verbose = True   
        self.bootstrap_ratio = 1
        self.bootstrap_rounds = 1
        self.reprojection_error_threshold = REPRO_ERR_THRES

    def read_config(self, json_path):
        with open(json_path) as f:
            j = json.load(f)
        for k, v in j.items():
            if isinstance(v, dict):
                v0 = getattr(self, k)
                v0.update(v)
                setattr(self, k, v0)
            else:
                setattr(self, k, v)

    def auto_points(self, nx=None, ny=None):
        rnx, rny = self.pattern_detection.num_points

        if nx is None:
            nx = rnx
        if ny is None:
            ny = rny

        sx, sy = self.pattern_detection.spacing
        
        world_points = np.zeros((ny * nx, 3), np.float32)

        #clockwise rotating the lattice
        if self.pattern_detection.zero_location == "top-left":
            world_points[:,:2] = np.mgrid[0:nx*sx:sx, 0:ny*sy:sy].T.reshape(-1,2)
        elif self.pattern_detection.zero_location == "top-right":
            world_points[:,:2] = np.mgrid[(rny-1)*sy:(rny-ny-1)*sy:-sy, 0:nx*sx:sx].T.reshape(-1,2)[:,::-1]
        elif self.pattern_detection.zero_location == "bottom-left":
            world_points[:,:2] = np.mgrid[(rnx-1)*sx:(rnx-nx-1)*sx:-sx, (rny-1)*sy:(rny-ny-1)*sy:-sy].T.reshape(-1,2)
        elif self.pattern_detection.zero_location == "bottom-right":
            world_points[:,:2] = np.mgrid[0:ny*sy:sy, (rnx-1)*sx:(rnx-nx-1)*sx:-sx].T.reshape(-1,2)[:,::-1]
        else:
            raise Exception(f'Zero-location {self.pattern_detection.zero_location} is incorrect')
                                    
        return world_points
