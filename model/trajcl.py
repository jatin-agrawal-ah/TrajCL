import sys
sys.path.append('..')

import time
import logging
import pickle
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial

from config import Config
from model.moco import MoCo
from model.dual_attention import DualSTB, DualSTBWithTime, DualSTBTimeWeighted
from utils.data_loader import read_traj_dataset, read_spark_dataset
from utils.traj import *
from utils import tool_funcs
from utils.cellspace import HirearchicalCellSpace
import warnings
import os
warnings.filterwarnings("ignore", message=".*Support for mismatched src_key_padding_mask and mask is deprecated.*")


class TrajCL(nn.Module):

    def __init__(self):
        super(TrajCL, self).__init__()

        encoder_q = DualSTBTimeWeighted(Config.seq_embedding_dim, 
                                            Config.trans_hidden_dim, 
                                            Config.trans_attention_head, 
                                            Config.trans_attention_layer, 
                                            Config.trans_attention_dropout, 
                                            Config.trans_pos_encoder_dropout)
        encoder_k = DualSTBTimeWeighted(Config.seq_embedding_dim, 
                                            Config.trans_hidden_dim, 
                                            Config.trans_attention_head, 
                                            Config.trans_attention_layer, 
                                            Config.trans_attention_dropout, 
                                            Config.trans_pos_encoder_dropout)

        self.clmodel = MoCo(encoder_q, encoder_k, 
                        Config.seq_embedding_dim,
                        Config.moco_proj_dim, 
                        Config.moco_nqueue,
                        temperature = Config.moco_temperature)


    def forward(self, trajs1_emb, trajs1_emb_p, trajs1_len, time_deltas1, trajs2_emb, trajs2_emb_p, trajs2_len, time_deltas2):
        # create kwargs inputs for TransformerEncoder
        
        max_trajs1_len = trajs1_len.max().item() # in essense -- trajs1_len[0]
        max_trajs2_len = trajs2_len.max().item() # in essense -- trajs2_len[0]
        src_padding_mask1 = torch.arange(max_trajs1_len, device = Config.device)[None, :] >= trajs1_len[:, None]
        src_padding_mask2 = torch.arange(max_trajs2_len, device = Config.device)[None, :] >= trajs2_len[:, None]

        logits, targets = self.clmodel({'src': trajs1_emb, 'time_deltas': time_deltas1, 'attn_mask': None, 'src_padding_mask': src_padding_mask1, 'src_len': trajs1_len, 'srcspatial': trajs1_emb_p},
                {'src': trajs2_emb, 'time_deltas': time_deltas2, 'attn_mask': None, 'src_padding_mask': src_padding_mask2, 'src_len': trajs2_len, 'srcspatial': trajs2_emb_p})
        # logits, targets = self.clmodel({'src': trajs1_emb, 'attn_mask': None, 'src_padding_mask': src_padding_mask1, 'src_len': trajs1_len, 'srcspatial': trajs1_emb_p},
        #         {'src': trajs2_emb, 'attn_mask': None, 'src_padding_mask': src_padding_mask2, 'src_len': trajs2_len, 'srcspatial': trajs2_emb_p})
        return logits, targets


    def interpret(self, trajs1_emb, trajs1_emb_p, trajs1_len, time_deltas1):
        max_trajs1_len = trajs1_len.max().item() # trajs1_len[0]
        src_padding_mask1 = torch.arange(max_trajs1_len, device = Config.device)[None, :] >= trajs1_len[:, None]
        # traj_embs = self.clmodel.encoder_q(**{'src': trajs1_emb, 'time_indices': time_indices1, 'attn_mask': None, 'src_padding_mask': src_padding_mask1, 'src_len': trajs1_len, 'srcspatial': trajs1_emb_p})
        traj_embs = self.clmodel.encoder_q(**{'src': trajs1_emb, 'time_deltas': time_deltas1, 'attn_mask': None, 'src_padding_mask': src_padding_mask1, 'src_len': trajs1_len, 'srcspatial': trajs1_emb_p})
        return traj_embs


    def loss(self, logits, targets):
        return self.clmodel.loss(logits, targets)


    def load_checkpoint(self):
        checkpoint_file = '{}/{}_TrajCL_best{}.pt'.format(Config.checkpoint_dir, Config.dataset_prefix, Config.dumpfile_uniqueid)
        checkpoint = torch.load(checkpoint_file)
        self.load_state_dict(checkpoint['model_state_dict'])
        return self

# def get_min_max_xy(traj):
#     min_x = min([p[0] for p in traj])
#     max_x = max([p[0] for p in traj])
#     min_y = min([p[1] for p in traj])
#     max_y = max([p[1] for p in traj])
#     return min_x, max_x, min_y, max_y

# def calculate_dx_dy(k, target_min_x, target_min_y, min_x, max_x, min_y, max_y):
#     dx = target_min_x+ (k - max_x - min_x) / 2
#     dy = target_min_y+ (k - max_y - min_y) / 2
#     return dx, dy

# def transform_traj(traj, dx, dy):
#     new_traj = [[p[0] + dx, p[1] + dy] for p in traj]
#     return new_traj

def collate_and_augment(batch, cellspace, embs_parent, embs_child, aug_func_list):
    # trajs: list of [[lon, lat], [,], ...]

    # 1. augment the input traj in order to form 2 augmented traj views
    # 2. convert augmented trajs to the trajs based on mercator space by cells
    # 3. read cell embeddings and form batch tensors (sort, pad)
    trajs = [t['merc_seq'] for t in batch]
    time_indices = [t['timestamps'] for t in batch]
    
    trajs1, trajs2 = [], []
    time_indices1, time_indices2 = [], []
    for l,t in zip(trajs, time_indices):
        random_int = np.random.randint(0,len(aug_func_list))
        augfn1 = aug_func_list[random_int]
        l,t = l[:Config.max_traj_len], t[:Config.max_traj_len]
        new_l, new_t = augfn1(l, t)
        # new_l = transform_traj(new_l, dx, dy)

        trajs1.append(new_l)
        time_indices1.append(new_t)
        random_int = np.random.randint(0,len(aug_func_list))
        augfn2 = aug_func_list[random_int]
        new_l, new_t = augfn2(l, t)
        # new_l = transform_traj(new_l, dx, dy)
        trajs2.append(new_l)
        time_indices2.append(new_t)


    trajs1_cell_parent, trajs1_cell_child, trajs1_p, trajs1_timedelta = zip(*[merc2cell2(t, cellspace) for t in trajs1])
    trajs2_cell_parent, trajs2_cell_child, trajs2_p, trajs2_timedelta = zip(*[merc2cell2(t, cellspace) for t in trajs2])
    
    trajs1_emb_p = [torch.tensor(generate_spatial_features(t, cellspace)) for t in trajs1_p]
    trajs2_emb_p = [torch.tensor(generate_spatial_features(t, cellspace)) for t in trajs2_p]

    trajs1_emb_p = pad_sequence(trajs1_emb_p, batch_first = False).to(Config.device)
    trajs2_emb_p = pad_sequence(trajs2_emb_p, batch_first = False).to(Config.device)

    trajs1_emb_cell_parent = [embs_parent[list(t)] for t in trajs1_cell_parent]
    trajs2_emb_cell_parent = [embs_parent[list(t)] for t in trajs2_cell_parent]

    trajs1_emb_cell_child = [embs_child[list(t)] for t in trajs1_cell_child]
    trajs2_emb_cell_child = [embs_child[list(t)] for t in trajs2_cell_child]

    trajs1_emb_cell = [a + b for a, b in zip(trajs1_emb_cell_parent, trajs1_emb_cell_child)] # add parent and child embeddings.
    trajs2_emb_cell = [a + b for a, b in zip(trajs2_emb_cell_parent, trajs2_emb_cell_child)]

    trajs1_emb_cell = pad_sequence(trajs1_emb_cell, batch_first = False).to(Config.device) # [seq_len, batch_size, emb_dim]
    trajs2_emb_cell = pad_sequence(trajs2_emb_cell, batch_first = False).to(Config.device) # [seq_len, batch_size, emb_dim]

    trajs1_len = torch.tensor(list(map(len, trajs1_cell_parent)), dtype = torch.long, device = Config.device)
    trajs2_len = torch.tensor(list(map(len, trajs2_cell_parent)), dtype = torch.long, device = Config.device)

    time_deltas1 = pad_sequence([torch.log(torch.tensor(t)) for t in trajs1_timedelta], batch_first=False, padding_value=0).to(Config.device) # [seq_len, batch_size]
    time_deltas2 = pad_sequence([torch.log(torch.tensor(t)) for t in trajs2_timedelta], batch_first=False, padding_value=0).to(Config.device)

    # return: two padded tensors and their lengths
    return trajs1_emb_cell.float(), trajs1_emb_p.float(), trajs1_len, time_deltas1, trajs2_emb_cell.float(), trajs2_emb_p.float(), trajs2_len, time_deltas2


def collate_for_test(batch, cellspace, embs):
    # trajs: list of [[lon, lat], [,], ...]

    # behavior is similar to collate_and_augment, but no augmentation
    trajs = [t['merc_seq'] for t in batch]
    time_indices = [t['time_indices'] for t in batch]

    trajs2_cell, trajs2_p = zip(*[merc2cell2(t, cellspace) for t in trajs])
    trajs2_emb_p = [torch.tensor(generate_spatial_features(t, cellspace)) for t in trajs2_p]
    trajs2_emb_p = pad_sequence(trajs2_emb_p, batch_first = False).to(Config.device)

    trajs2_emb_cell = [embs[list(t)] for t in trajs2_cell]
    trajs2_emb_cell = pad_sequence(trajs2_emb_cell, batch_first = False).to(Config.device) # [seq_len, batch_size, emb_dim]

    trajs2_len = torch.tensor(list(map(len, trajs2_cell)), dtype = torch.long, device = Config.device)
    time_indices2 = pad_sequence([torch.tensor(t, dtype=torch.long) for t in time_indices], batch_first=False, padding_value=-1).to(Config.device) # [seq_len, batch_size]
    
    # return: padded tensor and their length
    return trajs2_emb_cell, trajs2_emb_p, trajs2_len, time_indices2



class TrajCLTrainer:

    def __init__(self, aug_list):
        super(TrajCLTrainer, self).__init__()
        aug_fn_list = [get_aug_fn(str_aug) for str_aug in aug_list]
        self.embs_parent = pickle.load(open(Config.dataset_embs_file_parent, 'rb')).to('cpu').detach() # tensor
        self.embs_child = pickle.load(open(Config.dataset_embs_file_child, 'rb')).to('cpu').detach() # tensor
        self.cellspace_parent = pickle.load(open(Config.dataset_cell_file_parent, 'rb'))
        self.cellspace_child = pickle.load(open(Config.dataset_cell_file_child, 'rb'))
        self.hier_cellspace = HirearchicalCellSpace(self.cellspace_parent, self.cellspace_child)
        
        train_dataset =  read_spark_dataset(Config.parquet_data_dir)
        self.train_dataloader = DataLoader(train_dataset, 
                                            batch_size = Config.trajcl_batch_size, 
                                            shuffle = False, 
                                            num_workers = 0, 
                                            drop_last = True, 
                                            collate_fn = partial(collate_and_augment, cellspace = self.hier_cellspace, embs_parent = self.embs_parent, embs_child = self.embs_child, aug_func_list = aug_fn_list) )

        self.model = TrajCL().to(Config.device)
        if os.path.exists(Config.checkpoint_dir)==False:
            os.makedirs(Config.checkpoint_dir)
        self.checkpoint_file = '{}/{}_TrajCL_best{}.pt'.format(Config.checkpoint_dir, Config.dataset_prefix, Config.dumpfile_uniqueid)


    def train(self):
        training_starttime = time.time()
        training_gpu_usage = training_ram_usage = 0.0
        logging.info("[Training] START! timestamp={:.0f}".format(training_starttime))
        torch.autograd.set_detect_anomaly(True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr = Config.trajcl_training_lr, weight_decay = 0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = Config.trajcl_training_lr_degrade_step, gamma = Config.trajcl_training_lr_degrade_gamma)

        best_loss_train = 100000
        best_epoch = 0
        bad_counter = 0
        bad_patience = Config.trajcl_training_bad_patience

        for i_ep in range(Config.trajcl_training_epochs):
            _time_ep = time.time()
            loss_ep = []
            train_gpu = []
            train_ram = []

            self.model.train()

            _time_batch_start = time.time()
            for i_batch, batch in enumerate(self.train_dataloader):
                _time_batch = time.time()
                optimizer.zero_grad()

                trajs1_emb, trajs1_emb_p, trajs1_len, time_deltas1, trajs2_emb, trajs2_emb_p, trajs2_len, time_deltas2 = batch
                # print(trajs1_emb.dtype, trajs1_emb_p.dtype, trajs1_len.dtype, trajs2_emb.dtype, trajs2_emb_p.dtype, trajs2_len.dtype )
                model_rtn = self.model(trajs1_emb, trajs1_emb_p, trajs1_len, time_deltas1, trajs2_emb, trajs2_emb_p, trajs2_len, time_deltas2)
                loss = self.model.loss(*model_rtn)

                loss.backward()
                optimizer.step()
                loss_ep.append(loss.item())
                train_gpu.append(tool_funcs.GPUInfo.mem()[0])
                train_ram.append(tool_funcs.RAMInfo.mem())
                # print(i_batch)
                if i_batch % 100 == 0 and i_batch:
                    logging.debug("[Training] ep-batch={}-{}, loss={:.3f}, @={:.3f}, gpu={}, ram={}" \
                            .format(i_ep, i_batch, loss.item(), time.time() - _time_batch_start,
                                    tool_funcs.GPUInfo.mem(), tool_funcs.RAMInfo.mem()))

            scheduler.step() # decay before optimizer when pytorch < 1.1

            loss_ep_avg = tool_funcs.mean(loss_ep)
            logging.info("[Training] ep={}: avg_loss={:.3f}, @={:.3f}/{:.3f}, gpu={}, ram={}" \
                    .format(i_ep, loss_ep_avg, time.time() - _time_ep, time.time() - training_starttime,
                    tool_funcs.GPUInfo.mem(), tool_funcs.RAMInfo.mem()))
            
            training_gpu_usage = tool_funcs.mean(train_gpu)
            training_ram_usage = tool_funcs.mean(train_ram)

            # early stopping
            if loss_ep_avg < best_loss_train:
                best_epoch = i_ep
                best_loss_train = loss_ep_avg
                bad_counter = 0
                self.save_checkpoint()
            else:
                bad_counter += 1

            if bad_counter == bad_patience or (i_ep + 1) == Config.trajcl_training_epochs:
                logging.info("[Training] END! @={}, best_epoch={}, best_loss_train={:.6f}" \
                            .format(time.time()-training_starttime, best_epoch, best_loss_train))
                break
        
        return {'enc_train_time': time.time()-training_starttime, \
                'enc_train_gpu': training_gpu_usage, \
                'enc_train_ram': training_ram_usage}


    @torch.no_grad()
    def test(self):
        # 1. read best model
        # 2. read trajs from file, then -> embeddings
        # 3. run testing
        # n. varying db size, downsampling rates, and distort rates
        
        logging.info('[Test]start.')
        self.load_checkpoint()
        self.model.eval()

        # varying db size
        with open(Config.dataset_file + '_newsimi_raw.pkl', 'rb') as fh:
            q_lst, db_lst = pickle.load(fh)
            querys, databases = self.test_merc_seq_to_embs(q_lst, db_lst)
            dists = torch.cdist(querys, databases, p = 1) # [1000, 100000]
            targets = torch.diag(dists) # [1000]
            results = []
            for n_db in range(20000,100001,20000):
                rank = torch.sum(torch.le(dists[:,0:n_db].T, targets)).item() / len(q_lst)
                results.append(rank)
            logging.info('[EXPFlag]task=newsimi,encoder=TrajCL,varying=dbsize,r1={:.3f},r2={:.3f},r3={:.3f},r4={:.3f},r5={:.3f}' \
                                            .format(*results))

        # varying downsampling; varying distort
        for vt in ['downsampling', 'distort']:
            results = []
            for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
                with open(Config.dataset_file + '_newsimi_' + vt + '_' + str(rate) + '.pkl', 'rb') as fh:
                    q_lst, db_lst = pickle.load(fh)
                    querys, databases = self.test_merc_seq_to_embs(q_lst, db_lst)
                    dists = torch.cdist(querys, databases, p = 1) # [1000, 100000]
                    targets = torch.diag(dists) # [1000]
                    result = torch.sum(torch.le(dists.T, targets)).item() / len(q_lst)
                    results.append(result)
            logging.info('[EXPFlag]task=newsimi,encoder=TrajCL,varying={},r1={:.3f},r2={:.3f},r3={:.3f},r4={:.3f},r5={:.3f}' \
                                          .format(vt, *results))
        return


    @torch.no_grad()
    def test_merc_seq_to_embs(self, q_lst, db_lst):        
        querys = []
        databases = []
        num_query = len(q_lst) # 1000
        num_database = len(db_lst) # 100000
        batch_size = num_query

        for i in range(num_database // batch_size):
            if i == 0:
                trajs1_emb, trajs1_emb_p, trajs1_len, time_indices1 \
                        = collate_for_test(q_lst, self.cellspace, self.embs)
                trajs1_emb = self.model.interpret(trajs1_emb, trajs1_emb_p, trajs1_len, time_indices1)
                querys.append(trajs1_emb)

            trajs2_emb, trajs2_emb_p, trajs2_len, time_indices2 \
                    = collate_for_test(db_lst[i*batch_size : (i+1)*batch_size], self.cellspace, self.embs)
            trajs2_emb = self.model.interpret(trajs2_emb, trajs2_emb_p, trajs2_len, time_indices2)
            databases.append(trajs2_emb)

        querys = torch.cat(querys) # tensor; traj embeddings
        databases = torch.cat(databases)
        return querys, databases
    

    def dump_embeddings(self):
        return


    def save_checkpoint(self):
        torch.save({'model_state_dict': self.model.state_dict()},
                    self.checkpoint_file)
        return
    

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(Config.device)
        
        return



