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
from tqdm import tqdm
import numpy as np
import torch.amp
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
                        temperature = Config.moco_temperature,
                        neg_sampling = Config.neg_sampling)

    def forward(self, trajs1_emb, trajs1_emb_p, trajs1_len, time_deltas1, trajs2_emb, trajs2_emb_p, trajs2_len, time_deltas2, neg_trajs_emb=None, neg_trajs_emb_p=None, neg_trajs_len=None, neg_time_deltas=None):
        # create kwargs inputs for TransformerEncoder
        
        max_trajs1_len = trajs1_len.max().item() # in essense -- trajs1_len[0]
        max_trajs2_len = trajs2_len.max().item() # in essense -- trajs2_len[0]
        max_neg_trajs_len = neg_trajs_len.max().item() if neg_trajs_len is not None else 0

        src_padding_mask1 = torch.arange(max_trajs1_len, device = Config.device)[None, :] >= trajs1_len[:, None]
        src_padding_mask2 = torch.arange(max_trajs2_len, device = Config.device)[None, :] >= trajs2_len[:, None]
        src_padding_mask_neg = torch.arange(max_neg_trajs_len, device = Config.device)[None, :] >= neg_trajs_len[:, None] if neg_trajs_len is not None else None

        logits, targets = self.clmodel({'src': trajs1_emb, 'time_deltas': time_deltas1, 'attn_mask': None, 'src_padding_mask': src_padding_mask1, 'src_len': trajs1_len, 'srcspatial': trajs1_emb_p},
                {'src': trajs2_emb, 'time_deltas': time_deltas2, 'attn_mask': None, 'src_padding_mask': src_padding_mask2, 'src_len': trajs2_len, 'srcspatial': trajs2_emb_p}, 
                {'src': neg_trajs_emb, 'time_deltas': neg_time_deltas, 'attn_mask': None, 'src_padding_mask': src_padding_mask_neg, 'src_len': neg_trajs_len, 'srcspatial': neg_trajs_emb_p} if neg_trajs_emb is not None else None)
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
    
def collate_and_augment(batch, cellspace, embs_parent, embs_child, pos_aug_list, neg_aug_list):
    # trajs: list of [[lon, lat], [,], ...]

    # 1. augment the input traj in order to form 2 augmented traj views
    # 2. convert augmented trajs to the trajs based on mercator space by cells
    # 3. read cell embeddings and form batch tensors (sort, pad)
    trajs = [t['merc_seq'][:Config.max_traj_len] for t in batch]
    time_indices = [t['timestamps'][:Config.max_traj_len] for t in batch]
    
    trajs1, trajs2 = [], []
    time_indices1, time_indices2 = [], []
    for l,t in zip(trajs, time_indices):
        random_int = np.random.randint(0,len(pos_aug_list))
        augfn1 = pos_aug_list[random_int]
        l,t = l[:Config.max_traj_len], t[:Config.max_traj_len]
        new_l, new_t = augfn1(l, t)
        # new_l = transform_traj(new_l, dx, dy)

        trajs1.append(new_l)
        time_indices1.append(new_t)
        random_int = np.random.randint(0,len(pos_aug_list))
        augfn2 = pos_aug_list[random_int]
        new_l, new_t = augfn2(l, t)
        # new_l = transform_traj(new_l, dx, dy)
        trajs2.append(new_l)
        time_indices2.append(new_t)
    
    neg_traj, neg_time_indices = [], []
    for l,t in zip(trajs, time_indices):
        neg_aug_fn_1 = np.random.randint(0, len(neg_aug_list))  # randomly choose one of the two augmentation pairs
        new_l, new_t = neg_aug_list[neg_aug_fn_1](l, t)
        neg_traj.append(new_l)
        neg_time_indices.append(new_t)

    trajs1_cell_parent, trajs1_cell_child, trajs1_p, trajs1_timedelta = zip(*[merc2cell2(l,t, cellspace) for l,t in zip(trajs1, time_indices1)])
    trajs2_cell_parent, trajs2_cell_child, trajs2_p, trajs2_timedelta = zip(*[merc2cell2(l,t, cellspace) for l,t in zip(trajs2, time_indices2)])
    neg_trajs_cell_parent, neg_trajs_cell_child, neg_trajs_p, neg_trajs_timedelta = zip(*[merc2cell2(l,t, cellspace) for l,t in zip(neg_traj, neg_time_indices)])
    
    trajs1_emb_p = [torch.tensor(generate_spatial_features(t, cellspace)) for t in trajs1_p]
    trajs2_emb_p = [torch.tensor(generate_spatial_features(t, cellspace)) for t in trajs2_p]
    neg_trajs_emb_p = [torch.tensor(generate_spatial_features(t, cellspace)) for t in neg_trajs_p]

    trajs1_emb_p = pad_sequence(trajs1_emb_p, batch_first = False).to(Config.device)
    trajs2_emb_p = pad_sequence(trajs2_emb_p, batch_first = False).to(Config.device)
    neg_trajs_emb_p = pad_sequence(neg_trajs_emb_p, batch_first = False).to(Config.device)

    trajs1_emb_cell_parent = [embs_parent[list(t)] for t in trajs1_cell_parent]
    trajs2_emb_cell_parent = [embs_parent[list(t)] for t in trajs2_cell_parent]
    neg_trajs_emb_cell_parent = [embs_parent[list(t)] for t in neg_trajs_cell_parent]

    trajs1_emb_cell_child = [embs_child[list(t)] for t in trajs1_cell_child]
    trajs2_emb_cell_child = [embs_child[list(t)] for t in trajs2_cell_child]
    neg_trajs_emb_cell_child = [embs_child[list(t)] for t in neg_trajs_cell_child]

    trajs1_emb_cell = [a + b for a, b in zip(trajs1_emb_cell_parent, trajs1_emb_cell_child)] # add parent and child embeddings.
    trajs2_emb_cell = [a + b for a, b in zip(trajs2_emb_cell_parent, trajs2_emb_cell_child)]
    neg_trajs_emb_cell = [a + b for a, b in zip(neg_trajs_emb_cell_parent, neg_trajs_emb_cell_child)]

    trajs1_emb_cell = pad_sequence(trajs1_emb_cell, batch_first = False).to(Config.device) # [seq_len, batch_size, emb_dim]
    trajs2_emb_cell = pad_sequence(trajs2_emb_cell, batch_first = False).to(Config.device) # [seq_len, batch_size, emb_dim]
    neg_trajs_emb_cell = pad_sequence(neg_trajs_emb_cell, batch_first = False).to(Config.device) # [seq_len, batch_size, emb_dim]

    trajs1_len = torch.tensor(list(map(len, trajs1_cell_parent)), dtype = torch.long, device = Config.device)
    trajs2_len = torch.tensor(list(map(len, trajs2_cell_parent)), dtype = torch.long, device = Config.device)
    neg_trajs_len = torch.tensor(list(map(len, neg_trajs_cell_parent)), dtype = torch.long, device = Config.device)

    time_deltas1 = pad_sequence([torch.tensor(t) for t in trajs1_timedelta], batch_first=False, padding_value=0).to(Config.device) # [seq_len, batch_size]
    time_deltas2 = pad_sequence([torch.tensor(t) for t in trajs2_timedelta], batch_first=False, padding_value=0).to(Config.device)
    neg_time_deltas = pad_sequence([torch.tensor(t) for t in neg_trajs_timedelta], batch_first=False, padding_value=0).to(Config.device) # [seq_len, batch_size]

    # return: two padded tensors and their lengths
    return trajs1_emb_cell.float(), trajs1_emb_p.float(), trajs1_len, time_deltas1, trajs2_emb_cell.float(), trajs2_emb_p.float(), trajs2_len, time_deltas2, neg_trajs_emb_cell.float(), neg_trajs_emb_p.float(), neg_trajs_len, neg_time_deltas


def collate_for_test(batch, cellspace, embs_parent, embs_child, pos_aug_list, neg_aug_list):
    # trajs: list of [[lon, lat], [,], ...]

    trajs = [t['merc_seq'][:Config.max_traj_len] for t in batch]
    time_indices = [t['timestamps'][:Config.max_traj_len] for t in batch]

    pos_traj, pos_time_indices = [], []
    for l,t in zip(trajs, time_indices):
        augfn = np.random.randint(0,len(pos_aug_list))
        new_l, new_t = pos_aug_list[augfn](l, t)
        pos_traj.append(new_l)
        pos_time_indices.append(new_t)
    
    neg_traj, neg_time_indices = [], []
    for l,t in zip(trajs, time_indices):
        neg_aug_fn_1 = np.random.randint(0, len(neg_aug_list))  # randomly choose one of the two augmentation pairs
        new_l, new_t = neg_aug_list[neg_aug_fn_1](l, t)
        neg_traj.append(new_l)
        neg_time_indices.append(new_t)
    
    trajs_cell_parent, trajs_cell_child, trajs_p, trajs_timedelta = zip(*[merc2cell2(l,t, cellspace) for l,t in zip(trajs, time_indices)])
    pos_trajs_cell_parent, pos_trajs_cell_child, pos_trajs_p, pos_trajs_timedelta = zip(*[merc2cell2(l,t, cellspace) for l,t in zip(pos_traj, pos_time_indices)])
    neg_trajs_cell_parent, neg_trajs_cell_child, neg_trajs_p, neg_trajs_timedelta = zip(*[merc2cell2(l,t, cellspace) for l,t in zip(neg_traj, neg_time_indices)])

    trajs_emb_p = [torch.tensor(generate_spatial_features(t, cellspace)) for t in trajs_p]
    pos_trajs_emb_p = [torch.tensor(generate_spatial_features(t, cellspace)) for t in pos_trajs_p]
    neg_trajs_emb_p = [torch.tensor(generate_spatial_features(t, cellspace)) for t in neg_trajs_p]

    trajs_emb_p = pad_sequence(trajs_emb_p, batch_first = False).to(Config.device)
    pos_trajs_emb_p = pad_sequence(pos_trajs_emb_p, batch_first = False).to(Config.device)
    neg_trajs_emb_p = pad_sequence(neg_trajs_emb_p, batch_first = False).to(Config.device)

    trajs_emb_cell_parent = [embs_parent[list(t)] for t in trajs_cell_parent]
    pos_trajs_emb_cell_parent = [embs_parent[list(t)] for t in pos_trajs_cell_parent]
    neg_trajs_emb_cell_parent = [embs_parent[list(t)] for t in neg_trajs_cell_parent]   
    trajs_emb_cell_child = [embs_child[list(t)] for t in trajs_cell_child]
    pos_trajs_emb_cell_child = [embs_child[list(t)] for t in pos_trajs_cell_child]
    neg_trajs_emb_cell_child = [embs_child[list(t)] for t in neg_trajs_cell_child]

    trajs_emb_cell = [a + b for a, b in zip(trajs_emb_cell_parent, trajs_emb_cell_child)] # add parent and child embeddings.
    pos_trajs_emb_cell = [a + b for a, b in zip(pos_trajs_emb_cell_parent, pos_trajs_emb_cell_child)]
    neg_trajs_emb_cell = [a + b for a, b in zip(neg_trajs_emb_cell_parent, neg_trajs_emb_cell_child)]

    trajs_emb_cell = pad_sequence(trajs_emb_cell, batch_first = False).to(Config.device) # [seq_len, batch_size, emb_dim]
    pos_trajs_emb_cell = pad_sequence(pos_trajs_emb_cell, batch_first = False).to(Config.device) # [seq_len, batch_size, emb_dim]
    neg_trajs_emb_cell = pad_sequence(neg_trajs_emb_cell, batch_first = False).to(Config.device) # [seq_len, batch_size, emb_dim]

    trajs_len = torch.tensor(list(map(len, trajs_cell_parent)), dtype = torch.long, device = Config.device)
    pos_trajs_len = torch.tensor(list(map(len, pos_trajs_cell_parent)), dtype = torch.long, device = Config.device)
    neg_trajs_len = torch.tensor(list(map(len, neg_trajs_cell_parent)), dtype = torch.long, device = Config.device)

    time_deltas = pad_sequence([torch.tensor(t) for t in trajs_timedelta], batch_first=False, padding_value=0).to(Config.device) # [seq_len, batch_size]
    pos_time_deltas = pad_sequence([torch.tensor(t) for t in pos_trajs_timedelta], batch_first=False, padding_value=0).to(Config.device)
    neg_time_deltas = pad_sequence([torch.tensor(t) for t in neg_trajs_timedelta], batch_first=False, padding_value=0).to(Config.device) # [seq_len, batch_size]

    return trajs_emb_cell.float(), trajs_emb_p.float(), trajs_len, time_deltas, pos_trajs_emb_cell.float(), pos_trajs_emb_p.float(), pos_trajs_len, pos_time_deltas, neg_trajs_emb_cell.float(), neg_trajs_emb_p.float(), neg_trajs_len, neg_time_deltas



class TrajCLTrainer:

    def __init__(self, pos_aug_str_list, neg_aug_str_list):
        super(TrajCLTrainer, self).__init__()
        self.pos_aug_str_list = pos_aug_str_list
        self.neg_aug_str_list = neg_aug_str_list
        pos_aug_list = [get_aug_fn(name) for name in pos_aug_str_list]
        neg_aug_list = [get_aug_fn(name) for name in neg_aug_str_list]
        self.embs_parent = pickle.load(open(Config.dataset_embs_file_parent, 'rb')).to('cpu').detach() # tensor
        self.embs_child = pickle.load(open(Config.dataset_embs_file_child, 'rb')).to('cpu').detach() # tensor
        self.cellspace_parent = pickle.load(open(Config.dataset_cell_file_parent, 'rb'))
        self.cellspace_child = pickle.load(open(Config.dataset_cell_file_child, 'rb'))
        self.hier_cellspace = HirearchicalCellSpace(self.cellspace_parent, self.cellspace_child)
        
        train_dataset =  read_spark_dataset(Config.parquet_data_dir)
        test_dataset = read_spark_dataset(Config.parquet_data_dir_val)
        self.train_dataloader = DataLoader(train_dataset, 
                                            batch_size = Config.trajcl_batch_size, 
                                            shuffle = False, 
                                            num_workers = 0, 
                                            drop_last = True, 
                                            collate_fn = partial(collate_and_augment, cellspace = self.hier_cellspace, embs_parent = self.embs_parent, embs_child = self.embs_child, pos_aug_list = pos_aug_list, neg_aug_list = neg_aug_list) )
        self.test_dataloader =  DataLoader(test_dataset, 
                                    batch_size = Config.trajcl_batch_size, 
                                    shuffle = False, 
                                    num_workers = 0, 
                                    drop_last = False, 
                                    collate_fn =  partial(collate_for_test, cellspace = self.hier_cellspace, embs_parent = self.embs_parent, embs_child = self.embs_child, pos_aug_list = pos_aug_list, neg_aug_list = neg_aug_list) )
        self.model = TrajCL().to(Config.device)
        if os.path.exists(Config.checkpoint_dir)==False:
            os.makedirs(Config.checkpoint_dir)
        self.checkpoint_dir = Config.checkpoint_dir
        # self.checkpoint_file = '{}/{}_TrajCL_best{}.pt'.format(Config.checkpoint_dir, Config.dataset_prefix, Config.dumpfile_uniqueid)


    def train(self):
        training_starttime = time.time()
        training_gpu_usage = training_ram_usage = 0.0
        logging.info("[Training] START! timestamp={:.0f}".format(training_starttime))
        torch.autograd.set_detect_anomaly(True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr = Config.trajcl_training_lr, weight_decay = 0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = Config.trajcl_training_lr_degrade_step, gamma = Config.trajcl_training_lr_degrade_gamma)
        if Config.fp_16:
            scaler = torch.amp.GradScaler()

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

                if Config.fp_16:
                    with torch.amp.autocast("cuda", dtype = torch.float16):
                        trajs1_emb, trajs1_emb_p, trajs1_len, time_deltas1, trajs2_emb, trajs2_emb_p, trajs2_len, time_deltas2, neg_trajs_emb, neg_trajs_emb_p, neg_trajs_len, neg_time_deltas = batch
                        # print(trajs1_emb.dtype, trajs1_emb_p.dtype, trajs1_len.dtype, trajs2_emb.dtype, trajs2_emb_p.dtype, trajs2_len.dtype )
                        model_rtn = self.model(trajs1_emb, trajs1_emb_p, trajs1_len, time_deltas1, trajs2_emb, trajs2_emb_p, trajs2_len, time_deltas2, neg_trajs_emb, neg_trajs_emb_p, neg_trajs_len, neg_time_deltas)
                        loss = self.model.loss(*model_rtn)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    trajs1_emb, trajs1_emb_p, trajs1_len, time_deltas1, trajs2_emb, trajs2_emb_p, trajs2_len, time_deltas2, neg_trajs_emb, neg_trajs_emb_p, neg_trajs_len, neg_time_deltas = batch
                    # print(trajs1_emb.dtype, trajs1_emb_p.dtype, trajs1_len.dtype, trajs2_emb.dtype, trajs2_emb_p.dtype, trajs2_len.dtype )
                    model_rtn = self.model(trajs1_emb, trajs1_emb_p, trajs1_len, time_deltas1, trajs2_emb, trajs2_emb_p, trajs2_len, time_deltas2, neg_trajs_emb, neg_trajs_emb_p, neg_trajs_len, neg_time_deltas)
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
                if i_batch % Config.save_steps == 0 and i_batch!=0:
                    self.test()
                    self.save_checkpoint("ep{}_batch{}".format(i_ep, i_batch))
                    self.model.train()

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
                self.save_checkpoint("ep{}_final".format(i_ep))
            else:
                bad_counter += 1

            if bad_counter == bad_patience or (i_ep + 1) == Config.trajcl_training_epochs:
                logging.info("[Training] END! @={}, best_epoch={}, best_loss_train={:.6f}" \
                            .format(time.time()-training_starttime, best_epoch, best_loss_train))
                break
        
        return {'enc_train_time': time.time()-training_starttime, \
                'enc_train_gpu': training_gpu_usage, \
                'enc_train_ram': training_ram_usage}


    def save_checkpoint(self, checkpoint_name):
        checkpoint_file = os.path.join(self.checkpoint_dir, checkpoint_name)
        torch.save({'model_state_dict': self.model.state_dict()},
                    checkpoint_file)
        return
    

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(Config.device)
        
        return

    def test(self):
        self.model.eval()
        test_starttime = time.time()
        test_gpu_usage = test_ram_usage = 0.0
        logging.info("[Testing] START! timestamp={:.0f}".format(test_starttime))
        test_gpu = []
        test_ram = []

        trajs_emb_list = []
        pos_trajs_emb_list = []
        neg_trajs_emb_list = []
        with torch.no_grad():
            for i_batch, batch in tqdm(enumerate(self.test_dataloader)):
                _time_batch = time.time()
                if Config.fp_16:
                    with torch.amp.autocast("cuda", dtype = torch.float16):
                        trajs_emb, trajs_emb_p, trajs_len, time_deltas, pos_trajs_emb, pos_trajs_emb_p, pos_trajs_len, pos_time_deltas, neg_trajs_emb, neg_trajs_emb_p, neg_trajs_len, neg_time_deltas = batch
                        trajs_embs = self.model.interpret(trajs_emb, trajs_emb_p, trajs_len, time_deltas)
                        pos_trajs_embs = self.model.interpret(pos_trajs_emb, pos_trajs_emb_p, pos_trajs_len, pos_time_deltas)
                        neg_trajs_embs = self.model.interpret(neg_trajs_emb, neg_trajs_emb_p, neg_trajs_len, neg_time_deltas)
                else:
                    trajs_emb, trajs_emb_p, trajs_len, time_deltas, pos_trajs_emb, pos_trajs_emb_p, pos_trajs_len, pos_time_deltas, neg_trajs_emb, neg_trajs_emb_p, neg_trajs_len, neg_time_deltas = batch
                    trajs_embs = self.model.interpret(trajs_emb, trajs_emb_p, trajs_len, time_deltas)
                    pos_trajs_embs = self.model.interpret(pos_trajs_emb, pos_trajs_emb_p, pos_trajs_len, pos_time_deltas)
                    neg_trajs_embs = self.model.interpret(neg_trajs_emb, neg_trajs_emb_p, neg_trajs_len, neg_time_deltas)
                trajs_emb_list.append(trajs_embs.cpu())
                pos_trajs_emb_list.append(pos_trajs_embs.cpu())
                neg_trajs_emb_list.append(neg_trajs_embs.cpu())

        trajs_emb_all = torch.cat(trajs_emb_list, dim = 0)
        pos_trajs_emb_all = torch.cat(pos_trajs_emb_list, dim = 0)
        neg_trajs_emb_all = torch.cat(neg_trajs_emb_list, dim = 0)

        # find cosine similarity between trajs_emb_all and pos_trajs_emb_all, neg_trajs_emb_all
        pos_sim = torch.sum(trajs_emb_all * pos_trajs_emb_all, dim = 1) / (torch.norm(trajs_emb_all, dim = 1) * torch.norm(pos_trajs_emb_all, dim = 1) + 1e-8)
        neg_sim = torch.sum(trajs_emb_all * neg_trajs_emb_all, dim = 1) / (torch.norm(trajs_emb_all, dim = 1) * torch.norm(neg_trajs_emb_all, dim = 1) + 1e-8)

        # calculate accuracy, for pos_sim > 0.85, neg_sim < 0.85, accuracy = (TP + TN) / (P + N)
        pos_correct = torch.sum((pos_sim > 0.85).float()).item()
        neg_correct = torch.sum((neg_sim < 0.85).float()).item()
        total = pos_sim.shape[0]
        accuracy = (pos_correct + neg_correct) / (2 * total)

        # get median of pos_sim and neg_sim
        pos_median = torch.median(pos_sim).item()
        neg_median = torch.median(neg_sim).item()

        logging.info("[Testing] END! @={}, accuracy={:.4f}, pos_median={:.4f}, neg_median={:.4f}".format(time.time()-test_starttime, accuracy, pos_median, neg_median))

