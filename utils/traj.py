import sys
sys.path.append('..')
import numpy as np
import random
import math

from config import Config
from utils import tool_funcs
from utils.rdp import rdp, rdp_with_time_indices
from utils.cellspace import CellSpace
from utils.tool_funcs import truncated_rand


def straight(src, time_indices=None):
    return src

def simplify(src, time_indices=None):
    # src: [[lon, lat], [lon, lat], ...]
    if time_indices is None:
        return rdp(src, epsilon = Config.traj_simp_dist)
    else:
        return rdp_with_time_indices(src, time_indices, epsilon = Config.traj_simp_dist)

def time_shift(time_indices=None):
    time_shift_val = random.randint(Config.traj_time_shift_min, Config.traj_time_shift_max)
    if time_indices is not None:
        return [(t + time_shift_val) if t + time_shift_val <= Config.traj_max_time else Config.traj_max_time for t in time_indices]
    else:
        return None

def simplify_by_time(src, time_indices):
    new_src = []
    new_time_indices = []
    for i in range(len(src)):
        if i == 0:
            new_src.append(src[i])
            new_time_indices.append(time_indices[i])
        else:
            if int((time_indices[i] - new_time_indices[-1]))/1e9/60 > 10:
                new_src.append(src[i])
                new_time_indices.append(time_indices[i])
    new_src = np.array(new_src)
    new_time_indices = np.array(new_time_indices)
    return new_src, new_time_indices


def shift(src, time_indices=None):
    if time_indices is not None:
        # time_shifted = time_shift(time_indices)
        return [[p[0] + truncated_rand(), p[1] + truncated_rand()] for p in src], time_indices
    else:
        return [[p[0] + truncated_rand(), p[1] + truncated_rand()] for p in src]

def mask(src, time_indices=None):
    l = len(src)
    arr = np.array(src)
    # print(len(src), len(time_indices))
    if time_indices is not None:
        time_arr = np.array(time_indices)
        mask_idx_1 = np.random.choice(l, int(l * Config.traj_mask_ratio), replace=False)
        mask_idx = []
        for i in mask_idx_1:
            if i<len(time_indices)-1 and int((time_indices[i+1] - time_indices[i]))/1e9/60 < 60:
                mask_idx.append(i)
        return np.delete(arr, mask_idx, 0).tolist(), np.delete(time_arr, mask_idx, 0).tolist()
    else:
        mask_idx = np.random.choice(l, int(l * Config.traj_mask_ratio), replace=False)
        return np.delete(arr, mask_idx, 0).tolist()

def subset(src, time_indices=None):
    l = len(src)
    max_start_idx = l - int(l * Config.traj_subset_ratio)
    start_idx = random.randint(0, max_start_idx)
    end_idx = start_idx + int(l * Config.traj_subset_ratio)
    if time_indices is None:
        return src[start_idx: end_idx]
    else:
        return src[start_idx: end_idx], time_indices[start_idx: end_idx]


def shift_mask(src,time_indices):
    src, time_indices = shift(src, time_indices)
    src, time_indices = mask(src, time_indices)
    return src, time_indices

def simplify_shift(src, time_indices):
    src, time_indices = simplify(src, time_indices)
    src, time_indices = shift(src, time_indices)

    return src, time_indices
def get_aug_fn(name: str):
    return {'straight': straight, 'simplify': simplify, 'shift': shift,
            'mask': mask, 'subset': subset, 'shift_mask': shift_mask, 'simplify_shift': simplify_shift, "simplify_by_time": simplify_by_time  }.get(name, None)

def coalesce(coords,cell_ids, timestamps):
    start = 0
    new_coords = []
    new_cell_ids = []
    delta_t = []
    for i in range(1,len(cell_ids)):
        if cell_ids[i] != cell_ids[start]:
            new_coords.append(coords[start])
            new_cell_ids.append(cell_ids[start])
            dt2 = timestamps[i]
            dt1 = timestamps[start]
            # print(dt2,dt1,int(dt2-dt1)/1e9/60)
            delta_t.append(int((dt2 - dt1))/1e9/60+1)
            start = i
    new_coords.append(coords[start])
    new_cell_ids.append(cell_ids[start])
    dt2 = timestamps[-1]
    dt1 = timestamps[start]
    delta_t.append(int((dt2 - dt1))/1e9/60+1)
    return new_coords, new_cell_ids, delta_t    

# pair-wise conversion -- structural features and spatial feasures
def merc2cell2(coords, timestamps, cs: CellSpace):
    # convert and remove consecutive duplicates
    cellids = [cs.get_cellid_by_point(x,y) for x,y in coords]

    new_coords, new_cell_ids, delta_t = coalesce(coords, cellids, timestamps)
    # don't execute this if you want to keep the consecutive duplicate points. 
    # tgt = [v for i, v in enumerate(tgt) if i == 0 or v[0] != tgt[i-1][0]]

    return new_cell_ids, new_coords, delta_t


def generate_spatial_features(src, cs: CellSpace):
    # src = [length, 2]
    tgt = []
    lens = []
    for p1, p2 in tool_funcs.pairwise(src):
        lens.append(tool_funcs.l2_distance(p1[0], p1[1], p2[0], p2[1]))

    for i in range(1, len(src) - 1):
        dist = (lens[i-1] + lens[i]) / 2
        dist = dist / (Config.trajcl_local_mask_sidelen / 1.414) # float_ceil(sqrt(2))

        radian = math.pi - math.atan2(src[i-1][0] - src[i][0],  src[i-1][1] - src[i][1]) \
                        + math.atan2(src[i+1][0] - src[i][0],  src[i+1][1] - src[i][1])
        radian = 1 - abs(radian) / math.pi

        x = (src[i][0] - cs.x_min) / (cs.x_max - cs.x_min)
        y = (src[i][1] - cs.y_min)/ (cs.y_max - cs.y_min)
        tgt.append( [x, y, dist, radian] )

    x = (src[0][0] - cs.x_min) / (cs.x_max - cs.x_min)
    y = (src[0][1] - cs.y_min)/ (cs.y_max - cs.y_min)
    tgt.insert(0, [x, y, 0.0, 0.0] )
    
    x = (src[-1][0] - cs.x_min) / (cs.x_max - cs.x_min)
    y = (src[-1][1] - cs.y_min)/ (cs.y_max - cs.y_min)
    tgt.append( [x, y, 0.0, 0.0] )
    # tgt = [length, 4]
    if len(src)==1:
        return [tgt[0]]
    return tgt


def traj_len(src):
    length = 0.0
    for p1, p2 in tool_funcs.pairwise(src):
        length += tool_funcs.l2_distance(p1[0], p1[1], p2[0], p2[1])
    return length

