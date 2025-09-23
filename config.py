import os
import random
import torch
import numpy

def set_seed(seed = -1):
    if seed == -1:
        return
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Config:
    
    debug = True
    dumpfile_uniqueid = ''
    seed = 2000
    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root_dir = os.path.abspath(__file__)[:-10] # dont use os.getcwd()
    checkpoint_dir = root_dir + '/exp/usa_neg_sampling_smallqueue_512/'
    save_steps = 10000
    
    dataset = 'porto'
    dataset_prefix = ''
    dataset_file = ''
    dataset_cell_file = ''
    dataset_embs_file = ''

    min_lon = 0.0
    min_lat = 0.0
    max_lon = 0.0
    max_lat = 0.0
    max_traj_len = 500
    min_traj_len = 20
    cell_size = 100.0
    cellspace_buffer = 500.0
    max_len_meters = 100000

    #===========TrajCL=============
    neg_sampling = True
    trajcl_batch_size =128
    cell_embedding_dim = 256
    seq_embedding_dim = 256
    moco_proj_dim =  seq_embedding_dim // 2
    moco_nqueue = 32
    moco_temperature = 0.05

    trajcl_training_epochs = 20
    trajcl_training_bad_patience = 5
    trajcl_training_lr = 0.001
    trajcl_training_lr_degrade_gamma = 0.5
    trajcl_training_lr_degrade_step = 5
    trajcl_aug1 = 'mask'
    trajcl_aug2 = 'subset'
    trajcl_aug3 = "simplify"
    trajcl_aug4 = "shift"
    trajcl_aug5 = "shift_mask"
    trajcl_aug6 = "simplify_shift"
    trajcl_pos_aug_list = [ 'mask', 'simplify', 'shift','shift_mask', 'simplify_by_time']
    trajcl_neg_aug_list = [ 'jumble', 'translate']
    trajcl_local_mask_sidelen = cell_size * 11
    
    trans_attention_head = 4
    trans_attention_dropout = 0.1
    trans_attention_layer = 2
    trans_pos_encoder_dropout = 0.1
    trans_hidden_dim = 2048

    traj_time_shift_min = 0
    traj_time_shift_max = 20
    traj_max_time = 143
    traj_simp_dist = 250
    traj_shift_dist = 500
    traj_mask_ratio = 0.5
    traj_add_ratio = 0.3
    traj_subset_ratio = 0.7 # preserved ratio
    traj_large_time_shift_min = 70
    traj_large_time_shift_max = 90

    test_exp1_lcss_edr_epsilon = 0.25 # normalized


    # #===========trajsimi=============
    # trajsimi_encoder_name = 'TrajCL'
    # trajsimi_encoder_mode = 'finetune_all'
    # trajsimi_measure_fn_name = 'edwp'

    # trajsimi_batch_size = 128
    # trajsimi_epoch = 30
    # trajsimi_training_bad_patience = 10
    # trajsimi_learning_rate = 0.0001
    # trajsimi_learning_weight_decay = 0.0001
    # trajsimi_finetune_lr_rescale = 0.5


    @classmethod
    def update(cls, dic: dict):
        for k, v in dic.items():
            if k in cls.__dict__:
                assert type(getattr(Config, k)) == type(v)
            setattr(Config, k, v)
        cls.post_value_updates()


    @classmethod
    def post_value_updates(cls):
        if 'porto' == cls.dataset:
            cls.dataset_prefix = 'porto_20200'
            cls.min_lon = -8.7005
            cls.min_lat = 41.1001
            cls.max_lon = -8.5192
            cls.max_lat = 41.2086
        elif 'nyc' == cls.dataset:
            cls.dataset_prefix = 'nyc'
            cls.min_lon = -74.2591
            cls.min_lat = 40.4774
            cls.max_lon = -73.7004
            cls.max_lat = 40.9176 
            cls.cell_size = 250
        elif 'generic_v2' == cls.dataset:
            cls.dataset_prefix = 'generic_v2'
            cls.min_lon = 40
            cls.min_lat = 40
            cls.max_lon = 40.8
            cls.max_lat = 40.8
            cls.cell_size = 250
        elif 'generic_v3' == cls.dataset:
            cls.dataset_prefix = 'generic_v3'
            cls.min_lon = 40
            cls.min_lat = 40
            cls.max_lon = 40.8
            cls.max_lat = 40.8
            cls.cell_size = 250
        elif 'generic_small' == cls.dataset:
            cls.dataset_prefix = 'generic_small'
            cls.min_lon = 0
            cls.min_lat = 0
            cls.max_lon = 0.6
            cls.max_lat = 0.6
            cls.cell_size = 250
        elif 'generic_small_v2' == cls.dataset:
            cls.dataset_prefix = 'generic_small_v2'
            cls.min_lon = 70
            cls.min_lat = 40
            cls.max_lon = 70.6
            cls.max_lat = 40.6
            cls.cell_size = 250
        elif 'generic_small_v3' == cls.dataset:
            cls.dataset_prefix = 'generic_small_v3'
            cls.min_lon = 40
            cls.min_lat = 40
            cls.max_lon = 40.7
            cls.max_lat = 40.7
            cls.cell_size = 250
        elif 'generic_large' == cls.dataset:
            cls.dataset_prefix = 'generic_large'
            cls.min_lon = 40
            cls.min_lat = 40
            cls.max_lon = 41.8
            cls.max_lat = 41.4
            cls.cell_size = 250
        elif 'usa_large_cell_512' == cls.dataset:
            cls.dataset_prefix = 'usa_large_cell_512'
            cls.min_lon = -125 # -124.73306
            cls.min_lat = 25 # 25.11833
            cls.max_lon = -66 #-66.94978
            cls.max_lat = 50 #49.38447
            cls.cell_size = 50000
        elif 'usa_small_cell_512' == cls.dataset:
            cls.dataset_prefix = 'usa_small_cell_512'
            cls.min_lon = 0
            cls.min_lat = 0 
            cls.max_lon = 0.45 
            cls.max_lat = 0.45 
            cls.cell_size = 250
        else:
            pass
        
        cls.dataset_file = cls.root_dir + '/data/' + cls.dataset_prefix
        # cls.dataset_cell_file_parent = "/home/sagemaker-user/TrajCL/data/usa_large_cell_512_cell50000_cellspace.pkl"
        # cls.dataset_cell_file_child = "/home/sagemaker-user/TrajCL/data/usa_small_cell_512_cell250_cellspace.pkl"
        # cls.dataset_embs_file_parent = "/home/sagemaker-user/TrajCL/data/usa_large_cell_512_cell50000_embdim512_embs.pkl"
        # cls.dataset_embs_file_child = "/home/sagemaker-user/TrajCL/data/usa_small_cell_512_cell250_embdim512_embs.pkl"
        cls.dataset_cell_file_parent = "/home/sagemaker-user/TrajCL/data/usa_large_cell_cell50000_cellspace.pkl"
        cls.dataset_cell_file_child = "/home/sagemaker-user/TrajCL/data/usa_small_cell_cell250_cellspace.pkl"
        cls.dataset_embs_file_parent = "/home/sagemaker-user/TrajCL/data/usa_large_cell_cell50000_embdim256_embs.pkl"
        cls.dataset_embs_file_child = "/home/sagemaker-user/TrajCL/data/usa_small_cell_cell250_embdim256_embs.pkl"
        cls.dataset_cell_file = cls.dataset_file + '_cell' + str(int(cls.cell_size)) + '_cellspace.pkl'
        cls.dataset_embs_file = cls.dataset_file + '_cell' + str(int(cls.cell_size)) + '_embdim' + str(cls.cell_embedding_dim) + '_embs.pkl'
        cls.parquet_data_dir = "/mnt/sagemaker-nvme/data/usa/v1"
        cls.parquet_data_dir_val = "/mnt/sagemaker-nvme/data/usa/val"
        set_seed(cls.seed)

        cls.moco_proj_dim =  cls.seq_embedding_dim // 2

    @classmethod
    def to_str(cls): # __str__, self
        dic = cls.__dict__.copy()
        lst = list(filter( \
                        lambda p: (not p[0].startswith('__')) and type(p[1]) != classmethod, \
                        dic.items() \
                        ))
        return '\n'.join([str(k) + ' = ' + str(v) for k, v in lst])
