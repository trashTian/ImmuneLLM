

#train auto-encoder
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from Bio.Align import substitution_matrices
import matplotlib.pyplot as plt
import os
GPU_NUMBER = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
os.environ["NCCL_DEBUG"] = "INFO"
import sys
sys.path.append('.')
import numpy as np
import pandas as pd
import shutil
import argparse
from tqdm import tqdm
import pytorch_lightning as pl
pl.seed_everything(0)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from scripts.model_raw import TEIM
from utils.misc import load_config, calc_auc_aupr
from utils.dataset import load_data, SeqLevelDataset
import os

def GetBlosumMat(residues_list):
    n_residues = len(residues_list)  # the number of amino acids _ 'X'
    blosum62_mat = np.zeros([n_residues, n_residues])  # plus 1 for gap
    bl_dict = substitution_matrices.load('BLOSUM62')
    for pair, score in bl_dict.items():
        if (pair[0] not in residues_list) or (pair[1] not in residues_list):  # special residues not considered here
            continue
        idx_pair0 = residues_list.index(pair[0])  # index of residues
        idx_pair1 = residues_list.index(pair[1])
        blosum62_mat[idx_pair0, idx_pair1] = score
        blosum62_mat[idx_pair1, idx_pair0] = score
    return blosum62_mat

class Tokenizer:
    def __init__(self,):
        self.res_all = ['G', 'A', 'V', 'L', 'I', 'F', 'W', 'Y', 'D', 'N',
                     'E', 'K', 'Q', 'M', 'S', 'T', 'C', 'P', 'H', 'R'] #+ ['X'] #BJZOU
        self.tokens = ['-'] + self.res_all # '-' for padding encoding

    def tokenize(self, index): # int 2 str
        return self.tokens[index]

    def id(self, token): # str 2 int
        try:
            return self.tokens.index(token.upper())
        except ValueError:
            print('Error letter in the sequences:', token)
            if str.isalpha(token):
                return self.tokens.index('X')

    def tokenize_list(self, seq):
        return [self.tokenize(i) for i in seq]

    def id_list(self, seq):
        return [self.id(s) for s in seq]

    def embedding_mat(self):
        blosum62 = GetBlosumMat(self.res_all)
        mat = np.eye(len(self.tokens))
        mat[1:len(self.res_all) + 1, 1:len(self.res_all) + 1] = blosum62
        return mat

class AutoEncoder(nn.Module):
    def __init__(self, tokenizer, dim_hid, len_seq):
        super().__init__()
        embedding = tokenizer.embedding_mat()
        vocab_size, dim_emb = embedding.shape
        self.embedding_module = nn.Embedding.from_pretrained(torch.FloatTensor(embedding), padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Conv1d(dim_emb, dim_hid, 3, padding=1),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
            nn.Conv1d(dim_hid, dim_hid, 3, padding=1),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
        )

        self.seq2vec = nn.Sequential(
            nn.Flatten(),
            nn.Linear(len_seq * dim_hid, dim_hid),
            nn.ReLU()
        )
        self.vec2seq = nn.Sequential(
            nn.Linear(dim_hid, len_seq * dim_hid),
            nn.ReLU(),
            View(dim_hid, len_seq)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(dim_hid, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
            nn.ConvTranspose1d(dim_hid, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
        )
        self.out_layer = nn.Linear(dim_hid, vocab_size)

    def forward(self, inputs, latent_only=False):
        seq_emb = self.embedding_module(inputs)
        seq_enc = self.encoder(seq_emb.transpose(1, 2))
        vec = self.seq2vec(seq_enc)
        seq_repr = self.vec2seq(vec)
        seq_dec = self.decoder(seq_repr)
        out = self.out_layer(seq_dec.transpose(1, 2))
        if latent_only:
            return vec
        else:
            return out, seq_enc, vec



class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        shape = [input.shape[0]] + list(self.shape)
        return input.view(*shape)


def encoding_epi(seqs, max_len=15):
    encoding = np.zeros([len(seqs), max_len], dtype='long')
    for i, seq in enumerate(seqs):
        len_seq = len(seq)
#         print(tokenizer.id_list(seq))
        if (len_seq == 8) or (len_seq == 9):
            encoding[i, 3:len_seq+3] = tokenizer.id_list(seq)
        elif (len_seq == 10) or (len_seq == 11):
            encoding[i, 2:len_seq+2] = tokenizer.id_list(seq)
        elif (len_seq == 12) or (len_seq == 13):
            encoding[i, 1:len_seq+1] = tokenizer.id_list(seq)
        else:
            encoding[i, :len_seq] = tokenizer.id_list(seq)
    return encoding











class SeqLevelSystem(pl.LightningModule):
    def __init__(self, config, train_set, val_set):
        super().__init__()
        self.config = config
        self.lr = config.training.lr
        self.teim_seq = TEIM(config.model)
        self.train_set = train_set
        self.val_set = val_set
        
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.config.training.batch_size, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.config.training.batch_size, shuffle=False)

    def forward(self, x):
        return self.teim_seq(x)['seqlevel_out']
    
    def minimum_step(self, batch, device=None):
        # batch = batch.to(self.device)
        if device is None:
            cdr3, epi, labels = batch['cdr3'], batch['epi'], batch['labels']
        else:
            cdr3, epi, labels = batch['cdr3'].to(device), batch['epi'].to(device), batch['labels'].to(device)
        pred = self([cdr3, epi])
        loss = self.get_loss(pred, labels)
        return loss, labels, pred

    def training_step(self, batch, batch_idx):
        self.train()
        loss, labels, pred = self.minimum_step(batch)
        self.log('train/loss', loss)
        
        return {
            'loss':loss,
            'labels': labels,
            'pred': pred
        }

    def training_epoch_end(self, training_step_outputs):
        
        ## training metric
        loss, auc, aupr, auc_mean, aupr_mean = self.evaluate_model(self.train_dataloader())

        print('Train set: AUC={:.4}, AUPR={:.4}, AUC_AVG={:.4}, AUPR_AVG={:.4}'.format(auc, aupr, auc_mean, aupr_mean))
        self.log('lr', self.optimizers().state_dict()['param_groups'][0]['lr'])
        self.log_dict({
            'train/auc':auc,
            'train/aupr':aupr,
            'train/auc_avg':auc_mean,
            'train/aupr_avg':aupr_mean,
        }, prog_bar=False)

        

        ## validating metric
        loss, auc, aupr, auc_mean, aupr_mean = self.evaluate_model(self.val_dataloader())
        print('Valid', ' set: AUC={:.4}, AUPR={:.4}, AUC_AVG={:.4}, AUPR_AVG={:.4}'.format(auc, aupr, auc_mean, aupr_mean))
        self.log_dict({
            'valid/loss':loss,
            'valid/auc':auc,
            'valid/aupr':aupr,
            'valid/auc_avg':auc_mean,
            'valid/aupr_avg':aupr_mean,
        }, prog_bar=False)


    def evaluate_model(self, data_loader=None, ):
        self.eval()
        loss = 0
        y_true, y_pred = [], []
        epi_ids = []

        for i, batch in enumerate(data_loader):
            loss_this, y, y_hat = self.minimum_step(batch, self.device)
            loss += loss_this.item()
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(y_hat.detach().cpu().numpy().tolist())
            if 'epi_id' in batch:
                epi_ids.extend(batch['epi_id'].cpu().numpy().tolist())
        loss /= (i+1)
        auc, aupr = self.get_scores(y_true, y_pred)
        print(auc)
        print(aupr)
        ## per epi auc
        if len(epi_ids) > 0:
            ids_uni = np.unique(epi_ids, axis=0)
            print(ids_uni.shape)

            auc_sum = 0
            aupr_sum = 0
            cnt = 0
            for i, id_ in enumerate(ids_uni):
                index = np.array(epi_ids == id_)
                y_true_epi = np.array(y_true)[index]
                y_pred_epi = np.array(y_pred)[index]
                auc_epi, aupr_epi = self.get_scores(y_true_epi, y_pred_epi)
                if auc_epi is None:
                    continue
                auc_sum += auc_epi
                aupr_sum += aupr_epi
                cnt += 1
            auc_mean = auc_sum / cnt
            aupr_mean = aupr_sum / cnt
        else:
            auc_mean, aupr_mean = auc, aupr

        return loss, auc, aupr, auc_mean, aupr_mean


    def predict(self, data_loader=None):
        self.eval()
        cdr3_seqs, epi_seqs, y_true, y_pred = [], [], [], []
        epi_ids = []

        for i, batch in tqdm(enumerate(data_loader), desc='Predicting'):
            loss, y, y_hat = self.minimum_step(batch, self.device)
            cdr3_seqs.extend(batch['cdr3_seqs'])
            epi_seqs.extend(batch['epi_seqs'])
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(y_hat.detach().cpu().numpy().tolist())
            if 'epi_id' in batch.keys():
                epi_ids.extend(batch['epi_id'].cpu().numpy().tolist())

        if len(epi_ids) > 0:
            return cdr3_seqs, epi_seqs, y_true, np.reshape(y_pred, -1), epi_ids
        else:
            return cdr3_seqs, epi_seqs, y_true, np.reshape(y_pred, -1)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def get_loss(self, pred, labels):
        loss = F.binary_cross_entropy(pred.view(-1), labels.float(), weight=None, reduction='mean')
        return loss

    def get_scores(self, y_true, y_pred):
        if len(np.unique(y_true)) == 1:
            return None, None
        else:
            return calc_auc_aupr(y_true, y_pred)

def Model_retraining(trainfile_path, testfile_path, train_name, test_name,save_model_path, result_path):
    class Config:
        def __init__(self, path, file_list):
            self.path = path
            self.file_list = file_list

    config_path = 'configs/seqlevel_all.yml'
    config = load_config(config_path)  

    train_set = SeqLevelDataset(Config(path=trainfile_path, file_list=[train_name])) 
    val_set = SeqLevelDataset(Config(path=testfile_path, file_list=[test_name])) 

    model = SeqLevelSystem(config, train_set, val_set)
    checkpoint = ModelCheckpoint(monitor='valid/auc_avg', save_last=True, mode='max', save_top_k=1)
    earlystop = EarlyStopping(monitor='valid/auc_avg', patience=15, mode='max')
    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        gpus=1,
        callbacks=[checkpoint, earlystop],
        default_root_dir=os.path.join(os.getcwd(), 'logs', config.name)
    )

    trainer.fit(model)
    #model_save_path = os.path.join(save_model_path, 'model.pth')
    torch.save(model.state_dict(), save_model_path)
    shutil.copy2(config_path, os.path.join(trainer.log_dir, os.path.basename(config_path)))

    print('Predicting and saving results...')
    results = model.predict(model.val_dataloader())
    columns = ['CDR3B', 'Epitope', 'y_true', 'y_prob']
    if len(results) == 4:
        pd.DataFrame(zip(*results), columns=['CDR3B', 'Epitope', 'y_true', 'y_prob']).to_csv(result_path+'probability.csv', index=False)
    else:
        aa= pd.DataFrame(zip(*results), columns=['CDR3B', 'Epitope', 'y_true', 'y_prob', 'epi_id'])
        aa=aa[['CDR3B', 'Epitope', 'y_true', 'y_prob']]
        aa['y_pred'] = aa['y_prob'].apply(lambda x: 1 if x >= 0.5 else 0)
        aa.to_csv(result_path+'probability.csv', index=False)
    print('Done')

import pandas as pd
def fix_name(path,save_path):
    data=pd.read_csv(path)
    data=data[['CDR3B','Epitope','Affinity']]
    data.rename(columns={'CDR3B': 'cdr3', 'Epitope': 'epi', 'Affinity': 'y_true'}, inplace=True)
    data.to_csv(save_path) 


tokenizer = Tokenizer()
model_args = dict(
    tokenizer = tokenizer,
    dim_hid = 32,
    len_seq = 15,
)

batch_size = 512
epochs = 25
learning_rate = 2.e-4

all_data = []
for i in range(1,6):
    input_path = f'../data/retrain/ce_seen/as/{i}_1_1train.csv'
    df = pd.read_csv(input_path)
    all_data.append(df)
combined=pd.concat(all_data,ignore_index=True)
epitopes = combined['Epitope'].values.tolist()
encoded_epitopes = encoding_epi(epitopes)

inputs = torch.tensor(encoded_epitopes, dtype=torch.long)
targets = inputs.clone()  
dataset = TensorDataset(inputs, targets)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = AutoEncoder(**model_args)
criterion = nn.CrossEntropyLoss(ignore_index=0)  
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
losses=[]
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_inputs, batch_targets in data_loader:
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        optimizer.zero_grad()
        outputs, _, _ = model(batch_inputs)
        outputs = outputs.view(-1, outputs.size(-1))  
        batch_targets = batch_targets.view(-1)  
        
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(data_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
torch.save(model.state_dict(), "./ckpt/epi_ae_new.ckpt")
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), losses, marker='o', label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Trend')
plt.legend()
plt.grid()
plt.show()

config_path = 'configs/seqlevel_all.yml'
config = load_config(config_path)




path='../data/train.csv'
save_path='../data/train_TEIM.csv'
fix_name(path,save_path)


train_name='train_TEIM'
test_name='test_TEIM'
trainfile_path ="../data/"
testfile_path="../data/"
save_model_path="../Retraining_model/Retraining_model.pth"
result_path="../result_path/Retraining_model_prediction"
Model_retraining(trainfile_path,testfile_path,train_name,test_name,save_model_path,result_path) 