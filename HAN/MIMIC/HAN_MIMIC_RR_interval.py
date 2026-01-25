# %%
import torch
import os
import math
import random
from collections import defaultdict, Counter
from typing import List, Optional
import wfdb 
import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler, Subset
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.model_selection import train_test_split
from torchinfo import summary
from dotenv import load_dotenv
import mlflow
import mlflow.pytorch
import optuna
from optuna.exceptions import TrialPruned

load_dotenv()

# %%
MIMIC_PATH = "C:\\Users\\oladipea\\Downloads\\mimic\\mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"
SAMPLING_RATE = 500
SEED = 10
BATCH_SIZE = 16
DROP_LAST = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Setting up seeds for reproducibility
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# %%
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/oladipoeyiara@gmail.com/ecg_han_rr_intervals")

# %%
# Utilities: filtering and peak detection
def denoise(data):
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
    
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    rdata = np.nan_to_num(rdata, 0.0)
    return rdata


def pan_tompkins_detector(ecg_signal, fs):
    lowcut, highcut = 5.0, 15.0
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(1, [low, high], btype='band')
    filtered_ecg = filtfilt(b, a, ecg_signal)
    diff_ecg = np.diff(filtered_ecg)
    squared_ecg = diff_ecg ** 2
    window_size = int(0.150 * fs)
    mwa_ecg = np.convolve(squared_ecg, np.ones(window_size) / window_size, mode='same')
    peaks, _ = find_peaks(mwa_ecg, distance=int(0.6 * fs))
    return peaks


def multi_lead_fusion(detected_peaks, fs, fusion_window=0.1, min_leads=None):
    n_leads = len(detected_peaks)
    if min_leads is None:
        min_leads = int(np.ceil(n_leads / 2))

    all_peaks = [(p, lead) for lead, peaks in enumerate(detected_peaks) for p in peaks]
    all_peaks.sort(key=lambda x: x[0])

    fused_peaks = []
    i = 0

    while i < len(all_peaks):
        cluster = [all_peaks[i]]
        i += 1

        while i < len(all_peaks) and all_peaks[i][0] - cluster[-1][0] <= fusion_window * fs:
            cluster.append(all_peaks[i])
            i += 1

        unique_leads = {lead for _, lead in cluster}
        if len(unique_leads) >= min_leads:
            fused_peak = int(np.median([p for (p, _) in cluster]))
            fused_peaks.append(fused_peak)

    return np.array(sorted(fused_peaks))


def detect_r_peaks(ecg_signals, fs):
    detected_peaks = []
    for lead in ecg_signals:
        peaks = pan_tompkins_detector(lead, fs)
        detected_peaks.append(peaks)

    fused_r_peaks = multi_lead_fusion(detected_peaks, fs, fusion_window=0.1, min_leads=6)
    return fused_r_peaks


def extract_rr_beats_multi_lead(ecg_signals, fs, denoise_fn=None,
                                min_rr_ms=300, max_rr_ms=1500, min_beats=2):
    """
    Returns a list of beats, each beat is an array with shape (T_i, C) where C = n_leads.
    Variable length beats based on R-R intervals.
    """
    ecg = np.array(ecg_signals)  # (n_leads, n_samples)
    n_leads, n_samples = ecg.shape

    if denoise_fn is not None:
        ecg = np.array([denoise_fn(lead) for lead in ecg])

    # Detect fused R-peaks once across leads
    r_peaks = detect_r_peaks(ecg, fs)
    if len(r_peaks) < min_beats:
        return None  # not enough beats

    # RR in samples/ms
    rr_samples = np.diff(r_peaks)
    rr_ms = (rr_samples / fs) * 1000.0
    valid = (rr_ms >= min_rr_ms) & (rr_ms <= max_rr_ms)
    if valid.sum() == 0:
        return None

    beats = []
    for i in range(len(r_peaks) - 1):
        if not valid[i]:
            continue
        start = r_peaks[i]
        end = r_peaks[i+1]
        seg = ecg[:, start:end].T  # (T_i, C)
        if seg.shape[0] > 0:
            beats.append(seg.astype(np.float32))

    if len(beats) == 0:
        return None
    return beats  # list of variable-length (T_i, C)


# %%
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

# %%
from torch.utils.data import IterableDataset

class OnTheFlyBucketedBatchesRR(IterableDataset):
    """
    Modified to handle variable-length R-R interval beats with padding and masking.
    """
    def __init__(self, df, indices, mimic_path, classes, denoise_fn, 
                 batch_size=16, shuffle=True, drop_last=True,
                 min_leads=6, sampling_rate=SAMPLING_RATE,
                 min_rr_ms=300, max_rr_ms=1500, min_beats=2):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.indices = list(indices)
        self.mimic_path = mimic_path
        self.classes = classes
        self.denoise_fn = denoise_fn
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.min_leads = int(min_leads)
        self.sampling_rate = int(sampling_rate)
        self.min_rr_ms = min_rr_ms
        self.max_rr_ms = max_rr_ms
        self.min_beats = min_beats

    def _load_record(self, df_idx):
        row = self.df.iloc[df_idx]
        record_path = os.path.join(self.mimic_path, row['file_name'])

        record = wfdb.rdrecord(record_path)              
        ecg_signals = record.p_signal.T  # (n_leads, n_samples)
        ecg_signals = np.nan_to_num(ecg_signals, nan=0.0, posinf=0.0, neginf=0.0)

        # Extract variable-length R-R interval beats
        beats = extract_rr_beats_multi_lead(
            ecg_signals,
            fs=self.sampling_rate,
            denoise_fn=self.denoise_fn,
            min_rr_ms=self.min_rr_ms,
            max_rr_ms=self.max_rr_ms,
            min_beats=self.min_beats
        )
        
        if beats is None or len(beats) == 0:
            return None

        labels = row[self.classes].values.astype(np.float32)
        
        return {
            "beats": beats,  # list of (T_i, C) arrays
            "label": torch.as_tensor(labels, dtype=torch.float32),
            "num_beats": len(beats)
        }

    def _stack_batch_with_padding(self, items):
        """
        Pad variable-length beats and create masks.
        Returns:
            - signal: (B, S, T_max, C)
            - mask: (B, S, T_max) with 1 for real data, 0 for padding
            - labels: (B, num_classes)
        """
        B = len(items)
        S = items[0]["num_beats"]  # same S by bucketing
        C = items[0]["beats"][0].shape[1]  # number of leads (12)
        
        # Find max length across all beats in this batch
        T_max = max(beat.shape[0] for item in items for beat in item["beats"])
        
        # Initialize padded tensors
        signal = torch.zeros((B, S, T_max, C), dtype=torch.float32)
        mask = torch.zeros((B, S, T_max), dtype=torch.float32)
        labels = torch.stack([it["label"] for it in items], dim=0)  # (B, C)
        
        # Fill in actual data
        for bi, item in enumerate(items):
            for si, beat in enumerate(item["beats"]):
                T = beat.shape[0]
                beat_tensor = torch.from_numpy(beat) if isinstance(beat, np.ndarray) else beat
                signal[bi, si, :T, :] = beat_tensor
                mask[bi, si, :T] = 1.0
        
        return {"signal": signal, "mask": mask, "label": labels}

    def __iter__(self):
        # Shard indices across workers
        wi = torch.utils.data.get_worker_info()
        if wi is None:
            my_indices = list(self.indices)
        else:
            my_indices = self.indices[wi.id::wi.num_workers]

        if self.shuffle:
            rng = random.Random(SEED)
            rng.shuffle(my_indices)

        # Buckets keyed by num_beats (S)
        buckets = defaultdict(list)

        for df_idx in my_indices:
            try:
                item = self._load_record(df_idx)
            except Exception as e:
                print(f"Skip df_idx={df_idx} due to {e}")
                continue
            if item is None:
                continue

            S = item["num_beats"]
            buf = buckets[S]
            buf.append(item)

            if len(buf) >= self.batch_size:
                batch_items = buf[:self.batch_size]
                del buf[:self.batch_size]
                yield self._stack_batch_with_padding(batch_items)

        # Leftovers
        if not self.drop_last:
            for S, buf in list(buckets.items()):
                while len(buf) > 0:
                    take = min(self.batch_size, len(buf))
                    batch_items = buf[:take]
                    del buf[:take]
                    yield self._stack_batch_with_padding(batch_items)


# %%
classes = [
    "Chronic Ischemic Heart Disease (I25)",
    "Atrial Fibrillation (I48)",
    "Heart Failure (I50)",
    "Hypertensive Heart Diseases (I11-I16)",
    "Acute Myocardial Infarction (I21)",
    "Valve Disorders (I07,I08,I34,I35)",
    "Other"
]

df = load_mimic_metadata('records.csv', classes)

def make_bucketed_loader(df, indices, batch_size, shuffle, drop_last):
    ds_iter = OnTheFlyBucketedBatchesRR(
        df=df,
        indices=indices,
        mimic_path=MIMIC_PATH,
        classes=classes,
        denoise_fn=denoise,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        min_leads=6,
        sampling_rate=SAMPLING_RATE,
        min_rr_ms=300,
        max_rr_ms=1500,
        min_beats=2
    )
    
    return DataLoader(
        ds_iter,
        batch_size=None,       
        num_workers=4,           
        pin_memory=True
    )

# %%
class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        mid = max(1, channels // ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=True),
            nn.ReLU(),
            nn.Linear(mid, channels, bias=True)
        )
        
    def forward(self, x):
        avg_pool = torch.mean(x, dim=2)
        max_pool, _ = torch.max(x, dim=2)
        att = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).unsqueeze(2)
        return x * att


class SegmentAttention(nn.Module):
    def __init__(self, input_dim, units):
        super().__init__()
        self.linear = nn.Linear(input_dim, units, bias=True)
        self.u = nn.Parameter(torch.randn(units))
        
    def forward(self, inputs, mask=None):
        """
        inputs: (B, T, D)
        mask: (B, T) with 1=valid, 0=pad
        """
        v = torch.tanh(self.linear(inputs))
        vu = torch.matmul(v, self.u)
        if mask is not None:
            vu = vu.masked_fill(mask == 0, float('-inf'))
        alphas = F.softmax(vu, dim=1)
        alphas = torch.nan_to_num(alphas, nan=0.0)
        out = torch.sum(inputs * alphas.unsqueeze(-1), dim=1)
        return out, alphas


class TimeDistributedSegmentAttention(nn.Module):
    def __init__(self, input_dim, units):
        super().__init__()
        self.segment_attention = SegmentAttention(input_dim, units)
        
    def forward(self, inputs, mask=None):
        """
        inputs: (B, S, T, D)
        mask: (B, S, T)
        """
        B, S, T, D = inputs.shape
        flat = inputs.view(B * S, T, D)
        if mask is not None:
            mask_flat = mask.view(B * S, T)
        else:
            mask_flat = None
        outputs, alphas = self.segment_attention(flat, mask_flat)
        return outputs.view(B, S, D), alphas.view(B, S, T)


class HANWithAttention(nn.Module):
    def __init__(
        self,
        num_classes,
        conv_channels=128,
        segment_hidden=256,
        sequence_hidden=512,
        fc_hidden=2048,
        dropout=0.4,
    ):
        super().__init__()
        self.conv1d = nn.Conv1d(12, conv_channels, kernel_size=25, padding=12)
        self.channel_attention = ChannelAttention(conv_channels, ratio=8)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.lstm_segment = nn.LSTM(
            input_size=conv_channels,
            hidden_size=segment_hidden,
            batch_first=True
        )
        self.time_distributed_attention = TimeDistributedSegmentAttention(
            segment_hidden, segment_hidden
        )
        self.lstm_sequence = nn.LSTM(
            input_size=segment_hidden,
            hidden_size=sequence_hidden,
            batch_first=True
        )
        self.final_attention = SegmentAttention(sequence_hidden, sequence_hidden)
        self.fc = nn.Linear(sequence_hidden, fc_hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fc_hidden, num_classes)

    def forward(self, x, mask=None):
        logits, _, _ = self.forward_with_attention(x, mask)
        return logits

    def forward_with_attention(self, x, mask=None):
        """
        x: (B, S, T, C)
        mask: (B, S, T)
        """
        B, S, T, C = x.shape
        x = x.view(B * S, T, C).permute(0, 2, 1)

        conv = self.conv1d(x)
        att = self.channel_attention(conv)
        pooled = self.pool(att).permute(0, 2, 1)

        # Downsample mask
        if mask is not None:
            m = mask.view(B * S, 1, T)
            m2 = F.max_pool1d(m, kernel_size=3, stride=2, padding=1)
            m2 = (m2 > 0.0).float().squeeze(1)
        else:
            m2 = None

        seg_out, _ = self.lstm_segment(pooled)
        seg_out = seg_out.view(B, S, seg_out.shape[1], seg_out.shape[2])
        if m2 is not None:
            m2 = m2.view(B, S, -1)

        seg_feats, seg_alphas = self.time_distributed_attention(seg_out, mask=m2)
        seq_out, _ = self.lstm_sequence(seg_feats)
        final_vec, final_alphas = self.final_attention(seq_out)

        x = F.relu(self.fc(final_vec))
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits, final_alphas, seg_alphas


# %%
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score

criterion = nn.BCEWithLogitsLoss()

def multilabel_metrics(logits, targets, thresh=0.5):
    preds = (logits.sigmoid() > thresh).float().cpu().numpy()
    targets = targets.cpu().numpy()
    
    f1 = f1_score(targets, preds, average='samples', zero_division=0)
    hamming = hamming_loss(targets, preds)
    precision = precision_score(targets, preds, average='samples', zero_division=0)
    recall = recall_score(targets, preds, average='samples', zero_division=0)
    
    return f1, hamming, precision, recall

# %%
def evaluate(model, loader, criterion, device, threshold):
    model.eval()
    running_loss = 0.0
    running_f1 = running_hamming = running_precision = running_recall = 0.0
    steps = 0
    
    with torch.no_grad():
        for batch in loader:
            x = batch["signal"].to(device)
            mask = batch["mask"].to(device)
            y = batch["label"].to(device)
            logits = model(x, mask=mask)
            loss = criterion(logits, y)
            
            f1, hamming, precision, recall = multilabel_metrics(logits, y, threshold)
            running_loss += loss.item()
            running_f1 += f1
            running_hamming += hamming
            running_precision += precision
            running_recall += recall
            steps += 1
    
    return (running_loss/max(1,steps), running_f1/max(1,steps), running_hamming/max(1,steps), 
            running_precision/max(1,steps), running_recall/max(1,steps))

# %%
from tqdm import tqdm

def train(model, train_loader, val_loader, optimizer, criterion, device, epochs, scheduler, threshold):
    model.to(device)
    history = {
        "train_loss": [], "train_f1": [], "train_hamming": [], "train_precision": [], "train_recall": [],
        "val_loss": [],   "val_f1": [],   "val_hamming": [],   "val_precision": [],   "val_recall": []
    }
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_f1 = running_hamming = running_precision = running_recall = 0.0
        steps = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        
        for batch in loop:
            x = batch["signal"].to(device)
            mask = batch["mask"].to(device)
            y = batch["label"].to(device)
            
            optimizer.zero_grad()
            logits = model(x, mask=mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            f1, hamming, precision, recall = multilabel_metrics(logits.detach(), y, threshold)
            running_loss += loss.item()
            running_f1 += f1
            running_hamming += hamming
            running_precision += precision
            running_recall += recall
            steps += 1
            
            loop.set_postfix(
                loss=running_loss/steps,
                f1=running_f1/steps,
                hamming=running_hamming/steps
            )
                    
        epoch_train_loss = running_loss / max(1, steps)
        epoch_train_f1 = running_f1 / max(1, steps)
        epoch_train_hamming = running_hamming / max(1, steps)
        epoch_train_precision = running_precision / max(1, steps)
        epoch_train_recall = running_recall / max(1, steps)

        history["train_loss"].append(epoch_train_loss)
        history["train_f1"].append(epoch_train_f1)
        history["train_hamming"].append(epoch_train_hamming)
        history["train_precision"].append(epoch_train_precision)
        history["train_recall"].append(epoch_train_recall)
        
        if val_loader is not None:
            vloss, vf1, vhamming, vprecision, vrecall = evaluate(model, val_loader, criterion, device, threshold)
            history["val_loss"].append(vloss)
            history["val_f1"].append(vf1)
            history["val_hamming"].append(vhamming)
            history["val_precision"].append(vprecision)
            history["val_recall"].append(vrecall)

            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("train_f1", epoch_train_f1, step=epoch)
            mlflow.log_metric("val_loss", vloss, step=epoch)
            mlflow.log_metric("val_f1", vf1, step=epoch)
            mlflow.log_metric("val_hamming", vhamming, step=epoch)

            tqdm.write(
                f"Epoch {epoch} - "
                f"train_loss {epoch_train_loss:.4f} | val_loss {vloss:.4f} | "
                f"val_f1 {vf1:.4f} | val_hamming {vhamming:.4f}"
            )
        
        scheduler.step()
    
    return history


# %%
def objective(trial):
    set_global_seed(SEED)

    conv_channels = trial.suggest_categorical("conv_channels", [128, 256, 512])
    segment_hidden = trial.suggest_categorical("segment_hidden", [256, 384, 512])
    sequence_hidden = trial.suggest_categorical("sequence_hidden", [512, 768, 1024])
    fc_hidden = trial.suggest_categorical("fc_hidden", [1024, 2048])
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    lr = trial.suggest_float("lr", 1e-4, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    epochs = trial.suggest_int("epochs", 20, 40)
    threshold = trial.suggest_float("threshold", 0.2, 0.6)
    
    model = HANWithAttention(
        num_classes=len(classes),
        conv_channels=conv_channels,
        segment_hidden=segment_hidden,
        sequence_hidden=sequence_hidden,
        fc_hidden=fc_hidden,
        dropout=dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params({
            "conv_channels": conv_channels,
            "segment_hidden": segment_hidden,
            "sequence_hidden": sequence_hidden,
            "fc_hidden": fc_hidden,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "threshold": threshold
        })

        history = train(model, train_loader, val_loader, optimizer, criterion, 
                       DEVICE, epochs, scheduler, threshold)

        val_hamming_history = [h for h in history["val_hamming"] if h is not None]
        best_val_hamming = min(val_hamming_history)
        mlflow.log_metric("best_val_hamming", best_val_hamming)
        mlflow.pytorch.log_model(model, artifact_path="model")

    return best_val_hamming


# %%
import warnings
warnings.filterwarnings("ignore", module="pywt")

def main():
    set_global_seed(SEED)

    all_indices = list(range(len(df)))
    trainval_indices, test_indices = train_test_split(all_indices, test_size=0.1, random_state=42)
    train_indices, val_indices = train_test_split(trainval_indices, test_size=0.1, random_state=42)

    global train_loader, val_loader, test_loader
    train_loader = make_bucketed_loader(df, train_indices, batch_size=BATCH_SIZE, shuffle=True, drop_last=DROP_LAST)
    val_loader = make_bucketed_loader(df, val_indices, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader = make_bucketed_loader(df, test_indices, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    batch = next(iter(train_loader))
    print("Signal shape:", batch["signal"].shape)  # (B, S, T_max, C)
    print("Mask shape:", batch["mask"].shape)      # (B, S, T_max)
    print("Label shape:", batch["label"].shape)    # (B, num_classes)

    study = optuna.create_study(
        study_name="han_mimic_rr_intervals",
        direction="minimize"
    )

    study.optimize(objective, n_trials=20, show_progress_bar=True)

    print("Best Hamming:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
