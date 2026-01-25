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
mlflow.set_experiment("/Users/oladipoeyiara@gmail.com/lightx3ecg_mimic")

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


# -------------------------
# utilities: denoising
# -------------------------
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
    
    # Keep only records that have all required label columns
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
        ecg = torch.tensor(ecg, dtype=torch.float32)

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

class LightSEModule(nn.Module):
    def __init__(self,
        in_channels,
        reduction = 16,
    ):
        super(LightSEModule, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.s_conv = DSConv1d(
            in_channels, in_channels//reduction,
            kernel_size = 1,
        )
        self.act_fn = nn.ReLU()
        self.e_conv = DSConv1d(
            in_channels//reduction, in_channels,
            kernel_size = 1,
        )

    def forward(self,
        input,
    ):
        attention_scores = self.pool(input)

        attention_scores = self.s_conv(attention_scores)
        attention_scores = self.act_fn(attention_scores)
        attention_scores = self.e_conv(attention_scores)

        return input*torch.sigmoid(attention_scores)

class DSConv1d(nn.Module):
    def __init__(self,
        in_channels, out_channels,
        kernel_size, padding = 0, stride = 1,
    ):
        super(DSConv1d, self).__init__()
        self.dw_conv = nn.Conv1d(
            in_channels, in_channels,
            kernel_size = kernel_size, padding = padding, stride = stride,
            groups = in_channels,
            bias = False,
        )
        self.pw_conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size = 1,
            bias = False,
        )

    def forward(self,
        input,
    ):
        output = self.dw_conv(input)
        output = self.pw_conv(output)

        return output

class LightSEResBlock(nn.Module):
    def __init__(self,
        in_channels,
        downsample = False,
    ):
        super(LightSEResBlock, self).__init__()
        if downsample:
            self.out_channels = in_channels*2
            self.conv_1 = DSConv1d(
                in_channels, self.out_channels,
                kernel_size = 7, padding = 3, stride = 2,
            )
            self.identity = nn.Sequential(
                DSConv1d(
                    in_channels, self.out_channels,
                    kernel_size = 1, padding = 0, stride = 2,
                ),
                nn.BatchNorm1d(self.out_channels),
            )
        else:
            self.out_channels = in_channels
            self.conv_1 = DSConv1d(
                in_channels, self.out_channels,
                kernel_size = 7, padding = 3, stride = 1,
            )
            self.identity = nn.Identity()
        self.conv_2 = DSConv1d(
            self.out_channels, self.out_channels,
            kernel_size = 7, padding = 3, stride = 1,
        )

        self.convs = nn.Sequential(
            self.conv_1,
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            self.conv_2,
            nn.BatchNorm1d(self.out_channels),
            LightSEModule(self.out_channels),
        )
        self.act_fn = nn.ReLU()

    def forward(self,
        input,
    ):
        output = self.convs(input) + self.identity(input)
        output = self.act_fn(output)

        return output

class LightSEResNet18(nn.Module):
    def __init__(self,
        base_channels = 64,
    ):
        super(LightSEResNet18, self).__init__()
        self.bblock = LightSEResBlock
        self.stem = nn.Sequential(
            nn.Conv1d(
                1, base_channels,
                kernel_size = 15, padding = 7, stride = 2,
            ),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size = 3, padding = 1, stride = 2,
            ),
        )
        self.stage_0 = nn.Sequential(
            self.bblock(base_channels),
            self.bblock(base_channels),
        )

        self.stage_1 = nn.Sequential(
            self.bblock(base_channels*1, downsample = True),
            self.bblock(base_channels*2),
        )
        self.stage_2 = nn.Sequential(
            self.bblock(base_channels*2, downsample = True),
            self.bblock(base_channels*4),
        )
        self.stage_3 = nn.Sequential(
            self.bblock(base_channels*4, downsample = True),
            self.bblock(base_channels*8),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self,
        input,
    ):
        output = self.stem(input)
        output = self.stage_0(output)

        output = self.stage_1(output)
        output = self.stage_2(output)
        output = self.stage_3(output)

        output = self.pool(output)

        return output


class LightX3ECG(nn.Module):
    def __init__(self,
        base_channels = 64,
        num_classes = 11,
    ):
        super(LightX3ECG, self).__init__()
        self.backbone_0 = LightSEResNet18(base_channels)
        self.backbone_1 = LightSEResNet18(base_channels)
        self.backbone_2 = LightSEResNet18(base_channels)
        self.lw_attention = nn.Sequential(
            nn.Linear(
                base_channels*24, base_channels*8,
            ),
            nn.BatchNorm1d(base_channels*8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(
                base_channels*8, 3,
            ),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(
                base_channels*8, num_classes,
            ),
        )

    def forward(self,
        input,
        return_attention_scores = False,
    ):
        features_0 = self.backbone_0(input[:, 0, :].unsqueeze(1)).squeeze(2)
        features_1 = self.backbone_1(input[:, 1, :].unsqueeze(1)).squeeze(2)
        features_2 = self.backbone_2(input[:, 2, :].unsqueeze(1)).squeeze(2)
        attention_scores = torch.sigmoid(
            self.lw_attention(
                torch.cat(
                [
                    features_0,
                    features_1,
                    features_2,
                ],
                dim = 1,
                )
            )
        )
        merged_features = torch.sum(
            torch.stack(
            [
                features_0,
                features_1,
                features_2,
            ],
            dim = 1,
            )*attention_scores.unsqueeze(-1),
            dim = 1,
        )

        output = self.classifier(merged_features)

        if not return_attention_scores:
            return output
        else:
            return output, attention_scores
         
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
            model = LightX3ECG(num_classes=num_classes).to(device)

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
               

        # Test evaluation after final epoch using updated function
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
