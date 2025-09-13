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

def shift(src, time_indices=None):
    if time_indices is not None:
        time_shifted = time_shift(time_indices)
        return [[p[0] + truncated_rand(), p[1] + truncated_rand()] for p in src], time_shifted
    else:
        return [[p[0] + truncated_rand(), p[1] + truncated_rand()] for p in src]


def mask(src, time_indices=None):
    l = len(src)
    arr = np.array(src)
    if time_indices is not None:
        time_arr = np.array(time_indices)
        mask_idx = np.random.choice(l, int(l * Config.traj_mask_ratio), replace=False)
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


def get_aug_fn(name: str):
    return {'straight': straight, 'simplify': simplify, 'shift': shift,
            'mask': mask, 'subset': subset}.get(name, None)


# pair-wise conversion -- structural features and spatial feasures
def merc2cell2(src, cs):
    # convert and remove consecutive duplicates
    tgt = [ (cs.get_parent_child_cellid(*p)) for p in src]
    # don't execute this if you want to keep the consecutive duplicate points. 
    # tgt = [v for i, v in enumerate(tgt) if i == 0 or v[0] != tgt[i-1][0]]
    tgt_parent, tgt_child = zip(*tgt)
    return tgt_parent, tgt_child, src

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
    return tgt


def traj_len(src):
    length = 0.0
    for p1, p2 in tool_funcs.pairwise(src):
        length += tool_funcs.l2_distance(p1[0], p1[1], p2[0], p2[1])
    return length

