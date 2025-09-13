import sys
sys.path.append('..')
from config import Config
from utils import tool_funcs
from utils.cellspace import CellSpace
from utils.tool_funcs import lonlat2meters
from model.node2vec_ import train_node2vec
import pickle
import torch

def init_cellspace():
    # 1. create cellspase
    # 2. initialize cell embeddings (create graph, train, and dump to file)

    x_min, y_min = lonlat2meters(Config.min_lon, Config.min_lat)
    x_max, y_max = lonlat2meters(Config.max_lon, Config.max_lat)
    x_min -= Config.cellspace_buffer
    y_min -= Config.cellspace_buffer
    x_max += Config.cellspace_buffer
    y_max += Config.cellspace_buffer

    cell_size = int(Config.cell_size)
    cs = CellSpace(cell_size, cell_size, x_min, y_min, x_max, y_max)
    print("Initialized Cell Space!")
    print("Cell space size: ", cs.size())
    with open(Config.dataset_cell_file, 'wb') as fh:
        pickle.dump(cs, fh, protocol = pickle.HIGHEST_PROTOCOL)
    print("Dumpted Cell space pickle.")
    _, edge_index = cs.all_neighbour_cell_pairs_permutated_optmized()
    edge_index = torch.tensor(edge_index, dtype = torch.long, device = Config.device).T
    print("Started training node2vec")
    train_node2vec(edge_index)
    print("Done!")
    return


Config.dataset = "generic_v3"
Config.post_value_updates()

init_cellspace()