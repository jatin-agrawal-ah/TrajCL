from config_infer import *
from model.dual_attention import DualSTBTimeWeighted
from utils.traj import *
import pickle
from utils.cellspace import *
from torch.nn.utils.rnn import pad_sequence
import pickle
import warnings
warnings.filterwarnings("ignore")
import torch
cfg = InfernceConfig()
# print(cfg.to_str())

print(cfg.checkpoint_file)
encoder_q = DualSTBTimeWeighted(cfg.seq_embedding_dim, 
                                            cfg.trans_hidden_dim, 
                                            cfg.trans_attention_head, 
                                            cfg.trans_attention_layer, 
                                            cfg.trans_attention_dropout, 
                                            cfg.trans_pos_encoder_dropout)

encoder_q = encoder_q.to(cfg.device)
device = cfg.device

print(encoder_q)

# load model from checkpoint
checkpoint = torch.load(cfg.checkpoint_file, map_location=cfg.device)['model_state_dict']
encoder_q_keys = [k for k in list(checkpoint.keys()) if 'encoder_q' in k]

new_checkpoint = {}
for k in encoder_q_keys:
    new_k = k.replace('clmodel.encoder_q.', '')
    new_checkpoint[new_k] = checkpoint[k]

encoder_q.load_state_dict(new_checkpoint)
encoder_q.eval()
print("Model loaded from checkpoint.")

embs_parent = pickle.load(open(cfg.dataset_embs_file_parent, 'rb')).to('cpu').detach() # tensor
embs_child = pickle.load(open(cfg.dataset_embs_file_child, 'rb')).to('cpu').detach() # tensor
cellspace_parent = pickle.load(open(cfg.dataset_cell_file_parent, 'rb'))
cellspace_child = pickle.load(open(cfg.dataset_cell_file_child, 'rb'))
hier_cellspace = HirearchicalCellSpace(cellspace_parent, cellspace_child)

def model_forward(trajs1_emb, trajs1_emb_p, trajs1_len, time_deltas1):
    max_trajs1_len = trajs1_len.max().item() # trajs1_len[0]
    src_padding_mask1 = torch.arange(max_trajs1_len, device = Config.device)[None, :] >= trajs1_len[:, None]
    # traj_embs = self.clmodel.encoder_q(**{'src': trajs1_emb, 'time_indices': time_indices1, 'attn_mask': None, 'src_padding_mask': src_padding_mask1, 'src_len': trajs1_len, 'srcspatial': trajs1_emb_p})
    traj_embs = encoder_q(**{'src': trajs1_emb, 'time_deltas': time_deltas1, 'attn_mask': None, 'src_padding_mask': src_padding_mask1, 'src_len': trajs1_len, 'srcspatial': trajs1_emb_p})
    return traj_embs

def infer_batch(traj, time_indices):
    traj_cell_parent, traj_cell_child, traj_p, traj_timedelta = zip(*[merc2cell2(l,t, hier_cellspace) for l,t in zip(traj, time_indices)])
    # print(traj_cell)
    traj_emb_p = [torch.tensor(generate_spatial_features(t, hier_cellspace)) for t in traj_p]
    traj_emb_p = pad_sequence(traj_emb_p, batch_first = False).to(device)
    traj_emb_cell_parent = [embs_parent[list(t)] for t in traj_cell_parent]
    traj_emb_cell_child = [embs_child[list(t)] for t in traj_cell_child]
    traj_emb_cell = [a + b for a, b in zip(traj_emb_cell_parent, traj_emb_cell_child)]
    traj_emb_cell = pad_sequence(traj_emb_cell, batch_first = False).to(device)
    traj_len = torch.tensor(list(map(len, traj_p)), dtype = torch.long, device = device)
    traj_timedelta = pad_sequence([torch.log(torch.tensor(t)) for t in traj_timedelta], batch_first=False, padding_value=0).to(Config.device)
    # print(traj_emb_cell, traj_emb_p, traj_len)
    traj_embs = model_forward(traj_emb_cell.float(), traj_emb_p.float(), traj_len, traj_timedelta)
    return traj_embs, traj_cell_parent, traj_cell_child , traj_p, traj_timedelta

batch_size = cfg.batch_size
def infer(traj, time_indices):
    if len(traj)> batch_size:
        traj_embs = []
        for i in range(0, len(traj), batch_size):
            traj_batch = traj[i:i+batch_size]
            time_indices_batch = time_indices[i:i+batch_size] 
            traj_embs.append(infer_batch(traj_batch, time_indices_batch))
        return torch.cat(traj_embs, dim=0)
    else:
        return infer_batch(traj, time_indices)


def get_data_for_userid(userid):
    df_user = pd.concat([df[df['userid']==userid] for df in df_list]).reset_index(drop=True)
    return df_user

def get_data_for_employername(df_user, employername):
    df_emp = df_user[df_user['employername']==employername].reset_index(drop=True)
    return df_emp


def get_data_for_partition(df_emp, partition):
    df_part = df_emp[df_emp['partition_id']==partition].reset_index(drop=True)
    df_part = df_part[~df_part['weekday'].isin([5, 6])].reset_index(drop=True) # filter out weekends
    df_part = df_part[df_part['pck_amt']>0].reset_index(drop=True) # filter out zero paycheck amount
    return df_part


def get_traj_and_time_data(df_part):
    traj = df_part['merc_seq'].values
    time_indices = df_part['timestamps'].values
    return traj, time_indices


def apply_dbscan(embs, target_min_similarity=0.85):
    from sklearn.cluster import DBSCAN     # require >= 0.9 cosine similarity
    eps = 1.0 - target_min_similarity    # cosine distance threshold
    n_embs = embs.shape[0]
    db = DBSCAN(eps=eps, min_samples=max(int(n_embs*0.3), 5), metric="cosine", n_jobs=-1).fit(embs)
    return db

if __name__ == "__main__":
    from glob import glob
    import pandas as pd
    parquet_files = glob("/home/sagemaker-user/TrajCL/data/nyc/test/*.parquet")
    total_count = 0
    df_list = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        df_list.append(df)
        total_count += len(df)


    print(f"Total records in repartitioned files: {total_count}")

    unique_userids = pd.concat([df['userid'] for df in df_list]).unique()
    print(f"Total unique userids: {len(unique_userids)}")

    unique_employernames = pd.concat([df['employername'] for df in df_list]).unique()
    print(f"Total unique employernames: {len(unique_employernames)}")

    from tqdm import tqdm
    import torch.nn as nn
    output_dict = {}

    # 0: moving, 1: static, 2: not enough data
    def get_gt_and_pred_label(userid):
        count = 0
        user_data = get_data_for_userid(userid)
        employers = user_data['employername'].unique()
        for employer in employers:
            emp_data = get_data_for_employername(user_data, employer)
            partitions = emp_data['partition_id'].unique()
            for partition in partitions:
                # print(userid, employer, partition)
                part_data = get_data_for_partition(emp_data, partition)
                if len(part_data)<15:
                    continue
                else:
                    # print(userid, employer, partition)
                    # print(len(part_data))
                    traj, time_indices = get_traj_and_time_data(part_data)
                    # print(traj)
                    embs, traj_cell_parent, traj_cell_child , traj_p, traj_timedelta = infer(traj, time_indices)
                    embs = nn.functional.normalize(embs, dim=1)
                    db_scan = apply_dbscan(embs.detach().cpu().numpy(), target_min_similarity=0.95)
                    labels = db_scan.labels_                      # shape: (n_samples,)
                    cluster_ids = [c for c in np.unique(labels) if c != -1]
                    if len(cluster_ids)>0:
                        pred_label = 1
                    else:
                        pred_label = 0
                    output_dict[(userid, employer, partition)] = pred_label

        return count

    count=0
    for i, userid in tqdm(enumerate(unique_userids)):
        count+=get_gt_and_pred_label(userid)
        if i>1000:
            break


    count_dict = {}
    for k,v in output_dict.items():
        if v in count_dict:
            count_dict[v]+=1
        else:
            count_dict[v]=1
    print(count_dict)