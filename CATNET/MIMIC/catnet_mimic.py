import os
import math
import random
import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb
from tqdm import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, hamming_loss, recall_score, f1_score, accuracy_score
from dotenv import load_dotenv
import mlflow

load_dotenv()

MIMIC_PATH = "/home/ara/Pictures/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"

mlflow.pytorch.autolog()
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/oladipoeyiara@gmail.com/catnet_mimic")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


classes = [
    "Chronic Ischemic Heart Disease (I25)",
    "Atrial Fibrillation (I48)",
    "Heart Failure (I50)",
    "Hypertensive Heart Diseases (I11-I16)",
    "Acute Myocardial Infarction (I21)",
    "Valve Disorders (I07,I08,I34,I35)",
    "Other"
]

batch_size = 64
epochs = 20
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(classes)
all_results = []

def denoise(data):
    # wavelet transform
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # Threshold denoising
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # Inverse wavelet transform to obtain the denoised signal
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata

def load_mimic_metadata(csv_path, classes):
    df = pd.read_csv(csv_path)
    
    if 'file_name' in df.columns:
        df['file_name'] = df['file_name'].str.replace(
            'mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/', 
            '', 
            regex=False
        )
    
    df = df.dropna(subset=classes)
    
    return df


class ECGDataset(Dataset):
    def __init__(
        self,
        df,
        indices,
        mimic_path,
        classes,
        denoise_fn=None,
        sampling_rate=500
    ):
        self.df = df.reset_index(drop=True)
        self.indices = list(indices)
        self.mimic_path = mimic_path
        self.classes = classes
        self.denoise_fn = denoise_fn
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.indices)


    def __getitem__(self, idx):
        df_idx = self.indices[idx]
        row = self.df.iloc[df_idx]

        record_path = os.path.join(self.mimic_path, row["file_name"])
        record = wfdb.rdrecord(record_path)

        ecg = record.p_signal.T  # (n_leads, n_samples)

        if self.denoise_fn is not None:
            ecg = np.stack([self.denoise_fn(lead) for lead in ecg], axis=0)

        ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
        ecg = torch.tensor(ecg.T, dtype=torch.float32)

        labels = torch.tensor(
            row[self.classes].values.astype(np.float32),
            dtype=torch.float32
        )

        
        return {
            "signal": ecg,   # (5000, 12)
            "label": labels # (C,)
        }
  
def make_loader(df, indices, batch_size, shuffle):
    ds = ECGDataset(
        df=df,
        indices=indices,
        mimic_path=MIMIC_PATH,
        classes=classes,
        denoise_fn=denoise,
        sampling_rate=500,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

# -------------------------
# Channel Attention (CBAM)
# -------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        hidden = max(1, channels // ratio)
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True)
        )

    def forward(self, x):
        # x: (B, C, L)
        b, c, l = x.shape

        # Global average pooling and max pooling across temporal dimension -> (B, C)
        avg_pool = F.adaptive_avg_pool1d(x, 1).view(b, c)   # (B, C)
        max_pool = F.adaptive_max_pool1d(x, 1).view(b, c)   # (B, C)

        # Shared MLP applied to both
        avg_out = self.mlp(avg_pool)  # (B, C)
        max_out = self.mlp(max_pool)  # (B, C)

        # Combine and sigmoid
        out = torch.sigmoid(avg_out + max_out).unsqueeze(-1)  # (B, C, 1)

        # Scale input
        return x * out  # broadcasting over length

# -------------------------
# Spatial Attention (CBAM)
# -------------------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels=2, out_channels=1,
                              kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        # x: (B, C, L)
        # channel-wise avg and max -> (B, 1, L)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)

        cat = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2, L)
        attn = torch.sigmoid(self.conv(cat))  # (B, 1, L)
        return x * attn  # broadcast over channels

# -------------------------
# Transformer Encoder Block
# -------------------------
class TransformerEncoderBlock(nn.Module):
    """
    Input: (B, L, D)  (batch-first)
    Output: (B, L, D)
    """
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.2):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads,
                                         dropout=dropout_rate, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(inplace=True),
            nn.Linear(dff, d_model)
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # x: (B, L, D)
        attn_out, _ = self.mha(x, x, x,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        attn_out = self.dropout1(attn_out)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        ffn_out = self.dropout2(ffn_out)
        x = self.norm2(x + ffn_out)
        return x  # (B, L, D)

# -------------------------
# Positional Encoding (sinusoidal)
# -------------------------
def sinusoidal_positional_encoding(seq_len, d_model, device=None, dtype=None):
    """
    Returns tensor shape (1, seq_len, d_model) for broadcasting over batch.
    """
    if device is None:
        device = torch.device('cpu')
    pe = torch.zeros(seq_len, d_model, device=device, dtype=dtype)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float)
                         * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, seq_len, d_model)

# -------------------------
# Full Model
# -------------------------
class ECGModel(nn.Module):
    def __init__(self, sequence_length=5000, num_channels=12,
                 d_model=128, num_heads=4, dff=128, dropout_rate=0.2,
                 num_classes=11, apply_softmax=False):
        super().__init__()
        self.apply_softmax = apply_softmax

        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=21, padding=(21-1)//2)
        self.bn1 = nn.GroupNorm(num_groups=8, num_channels=32)
        self.ca1 = ChannelAttention(32, ratio=8)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  

        self.conv2 = nn.Conv1d(32, 32, kernel_size=23, padding=(23-1)//2)
        self.bn2 = nn.GroupNorm(num_groups=8, num_channels=32)
        self.ca2 = ChannelAttention(32, ratio=8)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=25, padding=(25-1)//2)
        self.bn3 = nn.GroupNorm(num_groups=8, num_channels=64)
        self.ca3 = ChannelAttention(64, ratio=8)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv4 = nn.Conv1d(64, 128, kernel_size=27, padding=(27-1)//2)
        self.bn4 = nn.GroupNorm(num_groups=8, num_channels=128)
        self.ca4 = ChannelAttention(128, ratio=8)

        # Project final conv channels to d_model if necessary (makes transformer flexible)
        if 128 != d_model:
            self.project_to_d_model = nn.Conv1d(128, d_model, kernel_size=1)
        else:
            self.project_to_d_model = None

        # Transformer encoder
        self.transformer_block = TransformerEncoderBlock(d_model=d_model, num_heads=num_heads,
                                                         dff=dff, dropout_rate=dropout_rate)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(128, num_classes)

        self._init_weights()

        # store config for lazy head init
        self.d_model = d_model

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


    def _forward_features(self, x):
        x = x.permute(0, 2, 1)  # -> (B, C, L)
        # Conv → BN → ReLU → CA → Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.ca1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.ca2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)
        x = self.ca3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x, inplace=True)
        x = self.ca4(x)
        # x: (B, channels, L_final)

        # project to d_model if needed
        if self.project_to_d_model is not None:
            x = self.project_to_d_model(x)  # (B, d_model, L_final)

        # Transpose for transformer: (B, L, D)
        x = x.permute(0, 2, 1)

        seq_len = x.size(1)
        pe = sinusoidal_positional_encoding(seq_len, self.d_model, device=x.device, dtype=x.dtype)
        x = x + pe  # (B, L, D)

        # Transformer encoder block
        x = self.transformer_block(x)  # (B, L, D)
        return x

    def forward(self, x):
        device = x.device
        features = self._forward_features(x)  # (B, L, D)
        pooled = features.mean(dim=1)         # (B, D)
        pooled = self.dropout(pooled)
        logits = self.fc_out(pooled)

        if self.apply_softmax:
            return F.softmax(logits, dim=-1)
        return logits

import warnings
warnings.filterwarnings("ignore", module="pywt")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def multilabel_metrics(logits, targets, thresh=0.36):
    preds = (logits.sigmoid() > thresh).float().cpu().numpy()
    probs = logits.sigmoid().cpu().numpy()
    targets = targets.cpu().numpy()
    
    f1 = f1_score(targets, preds, average='samples', zero_division=0)
    hamming = hamming_loss(targets, preds)
    precision = precision_score(targets, preds, average='samples', zero_division=0)
    recall = recall_score(targets, preds, average='samples', zero_division=0)
    
    auc_scores = []
    for i in range(targets.shape[1]):
        if len(np.unique(targets[:, i])) > 1:
            try:
                class_auc = roc_auc_score(targets[:, i], probs[:, i])
                auc_scores.append(class_auc)
            except (ValueError, Exception):
                continue
    
    auc = np.mean(auc_scores) if len(auc_scores) > 0 else 0.0
    
    return f1, hamming, precision, recall, auc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_f1 = running_hamming = running_precision = running_recall = running_auc = 0.0
    steps = 0
    
    with torch.no_grad():
        for batch in loader:
            x = batch["signal"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            
            f1, hamming, precision, recall, auc = multilabel_metrics(logits, y)
            running_loss += loss.item()
            running_f1 += f1
            running_hamming += hamming
            running_precision += precision
            running_recall += recall
            running_auc += auc
            steps += 1
    
    return (running_loss/max(1,steps), running_f1/max(1,steps), running_hamming/max(1,steps), 
            running_precision/max(1,steps), running_recall/max(1,steps), running_auc/max(1,steps))

seeds = list(range(10))


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    for seed_idx, seed in enumerate(seeds, 1):
        with mlflow.start_run(run_name=f"Seed {seed}"):
            print(f"\n===== SEED {seed_idx}: {seed} =====")
            set_seed(seed)
            
            df = load_mimic_metadata('records.csv', classes)
            df = df.reset_index(drop=True)

            def label_freq(idxs):
                return Y[idxs].mean(axis=0)

            Y = df[classes].astype(int).values

            msss = MultilabelStratifiedShuffleSplit(
                n_splits=1,
                test_size=0.1,
                random_state=seed
            )

            trainval_idx, test_indices = next(msss.split(df.index, Y))

            msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
            train_rel_idx, val_rel_idx = next(msss2.split(trainval_idx, Y[trainval_idx]))

            train_indices = trainval_idx[train_rel_idx]
            val_indices = trainval_idx[val_rel_idx]

            train_loader = make_loader(df, train_indices, batch_size=batch_size, shuffle=True)
            val_loader   = make_loader(df, val_indices,   batch_size=batch_size, shuffle=False)
            test_loader  = make_loader(df, test_indices,  batch_size=batch_size, shuffle=False)

            # Initialize model
            model = ECGModel(sequence_length=5000, num_channels=12,
                            d_model=128, num_heads=4, dff=128, dropout_rate=0.3,
                            num_classes=num_classes, apply_softmax=False).to(device)

                
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
            criterion = torch.nn.BCEWithLogitsLoss()

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                steps_per_epoch=len(train_loader),
                epochs=epochs,
                pct_start=0.1  
            )

            train_acc_list, val_acc_list = [], []
            history = {"train_loss": [], "train_f1": [], "train_hamming": [], "train_precision": [], "train_recall": [], "train_auc": [],
                        "val_loss": [], "val_f1": [], "val_hamming": [], "val_precision": [], "val_recall": [], "val_auc": []}
                
            for epoch in range(1, epochs+1):
                model.train()
                running_loss = 0.0
                running_f1 = running_hamming = running_precision = running_recall = running_auc = 0.0
                steps = 0
                
                loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

                for batch in loop:
                    x = batch["signal"].to(device)
                    y = batch["label"].to(device)
                    
                    optimizer.zero_grad()
                    logits = model(x)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                    f1, hamming, precision, recall, auc = multilabel_metrics(logits.detach(), y)
                    running_loss += loss.item()
                    running_f1 += f1
                    running_hamming += hamming
                    running_precision += precision
                    running_recall += recall
                    running_auc += auc
                    steps += 1
                    
                    loop.set_postfix(loss=running_loss/steps, f1=running_f1/steps, auc=running_auc/steps)
                            
                history["train_loss"].append(running_loss/max(1,steps))
                history["train_f1"].append(running_f1/max(1,steps))
                history["train_hamming"].append(running_hamming/max(1,steps))
                history["train_precision"].append(running_precision/max(1,steps))
                history["train_recall"].append(running_recall/max(1,steps))
                history["train_auc"].append(running_auc/max(1,steps))
            
                vloss, vf1, vhamming, vprecision, vrecall, vauc = evaluate(model, val_loader, criterion, device)
                history["val_loss"].append(vloss)
                history["val_f1"].append(vf1)
                history["val_hamming"].append(vhamming)
                history["val_precision"].append(vprecision)
                history["val_recall"].append(vrecall)
                history["val_auc"].append(vauc)
                tqdm.write(f"Epoch {epoch} — train_loss {history['train_loss'][-1]:.4f} | val_loss {vloss:.4f} | val_f1 {vf1:.4f} | val_auc {vauc:.4f} | val_hamming {vhamming:.4f} | val_precision {vprecision:.4f} | val_recall {vrecall:.4f}")
               

        test_loss, test_f1, test_hamming, test_prec, test_rec, test_auc = evaluate(
                model,
                test_loader,
                criterion,
                device
        )

        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("test_hamming_loss", test_hamming)
        mlflow.log_metric("test_precision", test_prec)
        mlflow.log_metric("test_recall", test_rec)
        mlflow.log_metric("test_auc", test_auc)


        print("Test Metrics:")
        print(f"Loss: {test_loss:.4f}")
        print(f"F1: {test_f1:.4f}")
        print(f"AUC: {test_auc:.4f}")
        print(f"Hamming: {test_hamming:.4f}")
        print(f"Precision: {test_prec:.4f}")
        print(f"Recall: {test_rec:.4f}")