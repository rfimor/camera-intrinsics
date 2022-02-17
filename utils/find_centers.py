from pydoc import cli
import numpy as np
import cv2
from matplotlib import pyplot as plt
from typing import Sequence

PAD_WID = 0

def normalize_image(f_img, channel_selection="all"):
    channel = 3
    channel_selection = channel_selection.lower()
    if channel_selection == "red":
        channel = 0
    elif channel_selection == "green":
        channel = 1
    elif channel_selection == "blue":
        channel = 2

    img = f_img.copy().astype(float)

    if len(img.shape) == 3 and img.shape[2] > 1:
        if channel == 3:
            img = np.sum(img, axis=2)
        else:
            img = img[:, :, channel]

    img -= np.min(img)
    img /= np.max(img)
    img *= 255
    img = img.astype(np.uint8)

    return img

def threshold_image(img, detect_conf):
    '''
        img should be an uint8 image
    '''
    est_radius = detect_conf.radius_lower_estimate

    is_adaptive = detect_conf.adaptive_threshold
    is_inv = detect_conf.inverse_threshold
    threshold = detect_conf.threshold
    if threshold is not None and threshold < 1:
        threshold *= np.max(img)
    
    if threshold is None:
        bin_opt = cv2.THRESH_BINARY_INV if is_inv else cv2.THRESH_BINARY
        if is_adaptive:
            img = cv2.adaptiveThreshold(img, 255,  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, bin_opt, est_radius*8+1, 2)
        else:
            _, img = cv2.threshold(img, 0, 255,  bin_opt + cv2.THRESH_OTSU)
    else:
        img[img<threshold] = 255 if is_inv else 0 
        img[img>=threshold] = 0 if is_inv else 255

    return img


def find_ellipses(clist, detect_conf):
    area = detect_conf.radius_lower_estimate ** 2
    ellipses = []
    #tmp_img = np.zeros(img.shape, dtype=np.uint8)

    for cont in clist:
        try:
            center, diameter, angle = cv2.fitEllipse(cont)
        except:
            continue

        if detect_conf.area_thresholds is not None:
            this_area = diameter[0]*diameter[1]
            if not (this_area > detect_conf.area_thresholds[0] * area and this_area < detect_conf.area_thresholds[1] * area):
                continue
        
        if detect_conf.eccentricity_threshold is not None:
            if diameter[0]/diameter[1] > detect_conf.eccentricity_threshold:
                continue
            if diameter[1]/diameter[0] > detect_conf.eccentricity_threshold:
                continue 
        
        ellipses.append({'center': center,
                        'diameter': diameter,
                        'angle': angle})

    if detect_conf.average_area_threshold is not None:
        avg_area = np.mean(np.array([d['diameter'][0]*d['diameter'][1] for d in ellipses]))
        ellipses = list(filter(
                        lambda d: d['diameter'][0]*d['diameter'][1] > avg_area / detect_conf.average_area_threshold and\
                                    d['diameter'][0]*d['diameter'][1] < avg_area * detect_conf.average_area_threshold, 
                        ellipses))

    return ellipses

def find_coutours(img, detect_conf):
    open_k = int(detect_conf.open_kernel * detect_conf.radius_lower_estimate) if detect_conf.open_kernel is not None else 2
    close_k = int(detect_conf.close_kernel * detect_conf.radius_lower_estimate) if detect_conf.close_kernel is not None else 2

    if open_k < 1:
        open_k = 1
    if close_k < 1:
        close_k = 1
    
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((open_k,open_k), dtype = np.uint8))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((close_k,close_k), dtype = np.uint8))

    # pad to find all circles
    if PAD_WID>0:
        pad_img = np.pad(img, 
                        ((PAD_WID, PAD_WID), (PAD_WID, PAD_WID)), 
                        mode='constant', 
                        constant_values=255 if detect_conf.inverse_threshold else 0)
    else:
        pad_img = img

    #plt.imshow(pad_img, cmap='gray')
    #plt.show()

    #tmp_img, clist, hierachy = cv2.findContours(pad_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    clist, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    return pad_img, clist

def find_ellipses_direct(u_img, detect_conf):
    _, clist = find_coutours(u_img, detect_conf)
    ellipses = find_ellipses(clist, detect_conf)
    for x in ellipses:
        x['center'] = np.array(x['center']) - np.array([PAD_WID, PAD_WID])
    
    #ctrs = list(map(lambda x: tuple(np.array(x['center'])), ellipses))
    #print(ctrs)
    #x, y = zip(*ctrs)
    #plt.plot(x, y, '-o')
    #plt.show()

    return ellipses

def find_bright_ellipses(f_img, ellipses, n_cluster, detect_conf):
    u_img = normalize_image(f_img, detect_conf.circle_detection.channel_selection)
    img = threshold_image(u_img.copy(), detect_conf.circle_detection)

    tmp_img = np.zeros(img.shape, dtype=np.uint8)
    i_list = []
    for ii in range(len(ellipses)):
        e = ellipses[ii]
        e_tmp = cv2.ellipse(tmp_img, (e['center'], e['diameter'], e['angle']), 1, -1)
        i_tmp = e_tmp * u_img
        i_list.append((np.sum(i_tmp) / np.sum(i_tmp > 0), e))

    if len(i_list) == 0:
        return []

    i_list = sorted(i_list, key = lambda x: -x[0])

    rslt = []
    for ii in range(n_cluster):
        rslt.append(i_list[ii][1])
    
    return rslt

def find_distances(points:Sequence[np.ndarray], num_neighbors:int=None) -> np.ndarray:
    mat = np.zeros(shape=(len(points), len(points)), dtype=float)
    pt_arry = np.array(points)

    for i in range(len(points)-1):
        y = pt_arry[i+1:, :] - pt_arry[i, :]
        mat[i,i+1:] = np.sqrt(y[:,0]**2 + y[:,1]**2)
        mat[i+1:,i] = mat[i,i+1:]
    
    if num_neighbors is not None:
        for i in range(len(points)):
            mat[i, [np.argsort(mat[i,:])[num_neighbors+1:]]] = np.inf
    
    return mat

def find_neighbors(source_index:int, 
                    distance_mat:np.ndarray,
                    num_neighbors:int=None):
    L = distance_mat[source_index, :] < np.inf
    nidx = np.arange(len(L))[L]
    idx = [n for n in nidx if n != source_index]
    
    if num_neighbors is None or num_neighbors>=len(idx):
        ret_idx = idx
    else:
        d = distance_mat[source_index, idx]
        ret_idx = [idx[n] for n in np.argsort(d)[:num_neighbors]]
    
    return ret_idx, distance_mat[source_index, ret_idx]

def find_next_point(current_point:np.ndarray, 
                    prev_point:np.ndarray, 
                    candidates:Sequence[np.ndarray], 
                    relative_distance_threshold:float,
                    distances:Sequence[float]=None):
    disp = current_point - prev_point
    nxt = current_point + disp
    cand = np.array(candidates)
    d = cand - nxt
    d = np.sqrt(d[:,0]**2 + d[:,1]**2)
    ii = np.argmin(d)

    maxd = np.linalg.norm(disp) if distances is None else distances[ii]
    if d[ii] < relative_distance_threshold * maxd:
        return ii
    else:
        return None

def find_next_neighbor(points:Sequence[np.ndarray],
                        current_index:int, 
                        prev_index:int, 
                        distance_mat:np.ndarray,
                        relative_distance_threshold:float,
                        num_neighbors:int=None):
    idx, dist = find_neighbors(current_index, distance_mat, num_neighbors)
    nxt = find_next_point(current_point=points[current_index], 
                        prev_point=points[prev_index],
                        candidates=[points[n] for n in idx],
                        relative_distance_threshold=relative_distance_threshold,
                        distances=dist)
    return None if nxt is None else idx[nxt]

def pair_neighbors(points:Sequence[np.ndarray], 
                    source_index:int, 
                    distance_mat:np.ndarray, 
                    relative_distance_threshold:float):
    idx, _ = find_neighbors(source_index, distance_mat)
    curr = points[source_index]

    pairs = []
    while len(idx)>1:
        nxt = find_next_point(current_point=curr, 
                            prev_point=points[idx[0]], 
                            candidates=[points[n] for n in idx[1:]], 
                            relative_distance_threshold=relative_distance_threshold, 
                            distances=[distance_mat[source_index, n] for n in idx[1:]])
        if nxt is not None:
            pairs.append((idx[0], idx[nxt+1]))
            idx.remove(idx[nxt+1])
        idx.remove(idx[0])
    
    return pairs

def _pick_topleft(points:Sequence[np.ndarray], index1:int, index2:int, visited:Sequence[int]=[]):
    if index1 in visited:
        if index2 not in visited: 
            return index2
        else:
            return None
    elif index2 in visited: 
        return index1
    else:
        x1, y1 = points[index1]
        x2, y2 = points[index2]
        if y1<y2 and x1<x2:
            return index1
        elif y1>y2 and x1>x2:
            return index2
        elif abs(x1-x2) <= abs(y1-y2):
            return index1 if y1<y2 else index2
        else:
            return index1 if x1<x2 else index2

def find_topleft(points:Sequence[np.ndarray], 
                    distance_mat:np.ndarray, 
                    relative_distance_threshold:float,
                    source_index:int=0):
    curr = source_index
    visited = [curr,]
    for _ in range(len(points)):
        p = pair_neighbors(points=points, 
                            source_index=curr,
                            distance_mat=distance_mat,
                            relative_distance_threshold=relative_distance_threshold)
        if len(p)==0:
            return curr
        else:
            nxt = None
            for n1, n2 in p:
                n = _pick_topleft(points=points,
                                index1=n1,
                                index2=n2,
                                visited=visited)
                if n is not None:
                    nxt = n if nxt is None else _pick_topleft(points, nxt, n)
            if nxt is None:
                return None
            else:
                curr = nxt            
                visited.append(curr)

    return None

def find_upper_edge(points:Sequence[np.ndarray], 
                    topleft_index:int, 
                    distance_mat:np.ndarray, 
                    relative_distance_threshold:float,
                    angle_threshold:float):
    nb, dist = find_neighbors(topleft_index, distance_mat, 3)
    angles = [np.arctan2(*((points[n]-points[topleft_index])[::-1])) for n in nb]
    ang_idx = np.argsort(angles)

    all_angles = [ang_idx[0]]
    for ii in range(1, len(ang_idx)):
        if angles[ang_idx[ii]] - angles[ang_idx[0]] < angle_threshold:
            all_angles.append(ang_idx[ii])

    if len(all_angles)==1:
        nxt = nb[ang_idx[0]]
    else:
        flt = [dist[n] for n in all_angles]
        nxt = nb[all_angles[np.argmin(flt)]]
    
    line = [topleft_index, nxt]
    while True:
        tail = find_next_neighbor(points=points,
                                current_index=line[-1],
                                prev_index=line[-2],
                                distance_mat=distance_mat,
                                relative_distance_threshold=relative_distance_threshold)
        if tail is None:
            return line        
        line.append(tail)

def find_centers(f_img, config):
    nx, ny = config.pattern_detection.num_points

    img = normalize_image(f_img, config.circle_detection.channel_selection)
    img = threshold_image(img, config.circle_detection)

    ellipses = find_ellipses_direct(img, config.circle_detection)
    ctrs = list(map(lambda x: np.array(x['center']), ellipses))
    dist_mat = find_distances(ctrs, num_neighbors=8)

    lines = []
    for _ in range(ny):
        ctr_mean = np.mean(ctrs, axis=0)
        ctr_dist = [np.linalg.norm(ctrs[ii]-ctr_mean) for ii in range(len(ctrs))]
        close_idx = np.argsort(ctr_dist)
        closest_idx = close_idx[:min(3, len(close_idx))]
        start_idx = closest_idx[np.argmin([ctrs[ii][1] for ii in closest_idx])]

        corner = find_topleft(points=ctrs,
                            distance_mat=dist_mat,
                            relative_distance_threshold=config.pattern_detection.relative_distance_threshold,
                            source_index=start_idx)
        line = find_upper_edge(points=ctrs, 
                            topleft_index=corner, 
                            distance_mat=dist_mat, 
                            relative_distance_threshold=config.pattern_detection.relative_distance_threshold,
                            angle_threshold=config.pattern_detection.angle_threshold)
        lines.append([ctrs[n] for n in line])

        if len(line)<nx:
            if not config.pattern_detection.allow_less_circles:
                raise Exception('Error: Not enough circles in the x direction at line {}. {} found'.format(len(lines), len(line)))
            else:
                print('Warning: Not enough circles in the x direction at line {}. {} found'.format(len(lines), len(line)))
    
        rest = [n for n in range(len(ctrs)) if n not in line]
        if len(rest) == 0: break
        ctrs = [ctrs[n] for n in rest]
        dist_mat = dist_mat[rest, :][:, rest]

    if len(lines) < ny:
        if not config.pattern_detection.allow_less_circles:
            raise Exception('Error: Not enough circles in the y direction. {} found'.format(len(lines)))
        else:
            print('Warning: Not enough circles in the y direction. {} found'.format(len(lines)))

    minx = np.min([len(l) for l in lines])
    minx = min(minx, nx)
    pts = []
    for l in lines:
        pts.extend(l[:minx])
    
    if config.show_images:
        x, y = zip(*pts)
        plt.plot(x, y, '-o')
        plt.show()

    return np.array(pts, dtype = np.float32), (len(lines), minx)
