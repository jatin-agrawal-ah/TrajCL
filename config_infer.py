import os
import random
import torch
import numpy

class InfernceConfig:
    def __init__(self):
        self.debug = True
        self.dumpfile_uniqueid = ''
        self.seed = 2000
        # self.device = torch.device("cpu")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.checkpoint_file = "/home/sagemaker-user/TrajCL/exp/usa_neg_sampling_smallqueue/ep1_batch100000"

        #===========TrajCL=============
        self.min_lon = -125 # -124.73306
        self.min_lat = 25 # 25.11833
        self.max_lon = -66 #-66.94978
        self.max_lat = 50 #49.38447
        self.cell_size = 50000
        self.neg_sampling = True
        self.batch_size = 512
        self.cell_embedding_dim = 256
        self.seq_embedding_dim = 256
        self.trajcl_local_mask_sidelen = self.cell_size * 11
        
        self.trans_attention_head = 4
        self.trans_attention_dropout = 0.1
        self.trans_attention_layer = 2
        self.trans_pos_encoder_dropout = 0.1
        self.trans_hidden_dim = 2048

        self.traj_time_shift_min = 0
        self.traj_time_shift_max = 20
        self.traj_max_time = 143
        self.traj_simp_dist = 250
        self.traj_shift_dist = 500
        self.traj_mask_ratio = 0.5
        self.traj_add_ratio = 0.3
        self.traj_subset_ratio = 0.7 # preserved ratio
        self.traj_large_time_shift_min = 70
        self.traj_large_time_shift_max = 90

        self.test_exp1_lcss_edr_epsilon = 0.25 # normalized

        self.dataset_cell_file_parent = "/home/sagemaker-user/TrajCL/data/usa_large_cell_cell50000_cellspace.pkl"
        self.dataset_cell_file_child = "/home/sagemaker-user/TrajCL/data/usa_small_cell_cell250_cellspace.pkl"
        self.dataset_embs_file_parent = "/home/sagemaker-user/TrajCL/data/usa_large_cell_cell50000_embdim256_embs.pkl"
        self.dataset_embs_file_child = "/home/sagemaker-user/TrajCL/data/usa_small_cell_cell250_embdim256_embs.pkl"

    # @classmethod
    # def to_str(cls): # __str__, self
    #     dic = cls.__dict__.copy()
    #     lst = list(filter( \
    #                     lambda p: (not p[0].startswith('__')) and type(p[1]) != classmethod, \
    #                     dic.items() \
    #                     ))
    #     return '\n'.join([str(k) + ' = ' + str(v) for k, v in lst])
