# %%
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
from tqdm import tqdm
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score, roc_auc_score
from dotenv import load_dotenv
import mlflow

load_dotenv()

# %%
MIMIC_PATH = "C:\\Users\\oladipea\\Downloads\\mimic\\mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"
SAMPLING_RATE = 500
PRE_PEAK_SAMPLES = 99
POST_PEAK_SAMPLES = 201
SEED = 10
SEGMENT_LENGTH = PRE_PEAK_SAMPLES + POST_PEAK_SAMPLES  
BATCH_SIZE = 16
DROP_LAST = True
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

mlflow.pytorch.autolog()
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/oladipoeyiara@gmail.com/ecg_han_no_tuning")

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


def extract_segments_around_peaks(signal, r_peaks, pre_samples, post_samples):
    segments = []

    for peak in r_peaks:
        start = max(0, peak - pre_samples)
        end = min(len(signal), peak + post_samples)

        if end - start == pre_samples + post_samples:
            segment = signal[start:end]
            segments.append(segment)

    return segments

def extract_beats_multi_lead(ecg_signals, fs, pre_samples, post_samples, denoise_fn=None):
    ecg_signals = np.array(ecg_signals)

    if denoise_fn is not None:
        ecg_signals = np.array([denoise_fn(lead) for lead in ecg_signals])

    r_peaks = detect_r_peaks(ecg_signals, fs)

    if len(r_peaks) == 0:
        return None

    all_lead_segments = []
    for lead_idx, lead_signal in enumerate(ecg_signals):
        segments = extract_segments_around_peaks(lead_signal, r_peaks, pre_samples, post_samples)
        all_lead_segments.append(segments)

    min_segments = min(len(segments) for segments in all_lead_segments)

    if min_segments == 0:
        return None

    # Stack segments: (n_beats, segment_length, n_leads)
    beats_arr = np.stack([
        np.stack(segments[:min_segments], axis=0)
        for segments in all_lead_segments
    ], axis=-1)

    return beats_arr.astype(np.float32)

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

class OnTheFlyBucketedBatches(IterableDataset):
    def __init__(self, df, indices, mimic_path, pre_samples, post_samples,
                 classes, denoise_fn, batch_size=16, shuffle=True, drop_last=True,
                 min_leads=6, sampling_rate=SAMPLING_RATE):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.indices = list(indices)
        self.mimic_path = mimic_path
        self.pre_samples = pre_samples
        self.post_samples = post_samples
        self.classes = classes
        self.denoise_fn = denoise_fn
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.min_leads = int(min_leads)
        self.sampling_rate = int(sampling_rate)

    def _load_record(self, df_idx):
        row = self.df.iloc[df_idx]
        record_path = os.path.join(self.mimic_path, row['file_name'])

        record = wfdb.rdrecord(record_path)              
        ecg_signals = record.p_signal.T                  # (n_leads, n_samples)

        beats = extract_beats_multi_lead(
            ecg_signals,
            fs=self.sampling_rate,
            pre_samples=self.pre_samples,
            post_samples=self.post_samples,
            denoise_fn=self.denoise_fn
        )
        if beats is None or len(beats) == 0:
            return None

        labels = row[self.classes].values.astype(np.float32)
        
        return {
            "signal": torch.as_tensor(beats, dtype=torch.float32),  # (S, 300, 12)
            "label": torch.as_tensor(labels, dtype=torch.float32),  # (C,)
            "num_beats": beats.shape[0]
        }

    def _stack_batch(self, items):
        # all have same S by construction
        S = items[0]["num_beats"]
        signals = torch.stack([it["signal"] for it in items], dim=0)   # (B, S, 300, 12)
        labels  = torch.stack([it["label"]  for it in items], dim=0)   # (B, C)
        return {"signal": signals, "label": labels}

    def __iter__(self):
        # shard indices across workers (if any)
        wi = torch.utils.data.get_worker_info()
        if wi is None:
            my_indices = list(self.indices)
        else:
            my_indices = self.indices[wi.id::wi.num_workers]

        if self.shuffle:
            rng = random.Random(SEED)
            rng.shuffle(my_indices)

        # buckets keyed by true num_beats
        buckets = defaultdict(list)

        for df_idx in my_indices:
            try:
                item = self._load_record(df_idx)
            except Exception as e:
                # skip unreadable/bad record
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
                yield self._stack_batch(batch_items)

        # leftovers
        if not self.drop_last:
            for S, buf in list(buckets.items()):
                while len(buf) > 0:
                    take = min(self.batch_size, len(buf))
                    batch_items = buf[:take]
                    del buf[:take]
                    yield self._stack_batch(batch_items)


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

def make_bucketed_loader(df, indices, batch_size, shuffle, drop_last):
    ds_iter = OnTheFlyBucketedBatches(
        df=df,
        indices=indices,
        mimic_path=MIMIC_PATH,
        pre_samples=PRE_PEAK_SAMPLES,
        post_samples=POST_PEAK_SAMPLES,
        classes=classes,
        denoise_fn=denoise,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        min_leads=6,
        sampling_rate=SAMPLING_RATE
    )
    
    return DataLoader(
        ds_iter,
        batch_size=None,       
        num_workers=0,           
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
        
    def forward(self, inputs):
        v = torch.tanh(self.linear(inputs))   # (B,T,units)
        vu = torch.matmul(v, self.u)          # (B,T)
        alphas = F.softmax(vu, dim=1)
        out = torch.sum(inputs * alphas.unsqueeze(-1), dim=1)
        return out, alphas

class TimeDistributedSegmentAttention(nn.Module):
    def __init__(self, input_dim, units):
        super().__init__()
        self.segment_attention = SegmentAttention(input_dim, units)
        
    def forward(self, inputs):
        B,S,T,D = inputs.shape
        flat = inputs.view(B*S, T, D)
        out, alphas = self.segment_attention(flat)
        return out.view(B,S,D), alphas.view(B,S,T)

class HANWithAttention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1d = nn.Conv1d(12, 512, kernel_size=25, padding=12)
        self.channel_attention = ChannelAttention(512, ratio=8)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.lstm_segment = nn.LSTM(input_size=512, hidden_size=256, batch_first=True)
        self.time_distributed_attention = TimeDistributedSegmentAttention(256, 256)
        self.lstm_sequence = nn.LSTM(input_size=256, hidden_size=512, batch_first=True)
        self.final_attention = SegmentAttention(512, 512)

        self.fc = nn.Linear(512, 512)
        self.dropout = nn.Dropout(0.45)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        logits, _, _ = self.forward_with_attention(x)
        return logits

    def forward_with_attention(self, x):
        B,S,T,C = x.shape 
        x = x.view(B*S, T, C).permute(0, 2, 1)  # (B*S, C, T) - Each beat processed separately

        conv = self.conv1d(x)
        att  = self.channel_attention(conv)
        pooled = self.pool(att).permute(0, 2, 1)     # (B*S, T2, 128)

        seg_out, _ = self.lstm_segment(pooled)       # (B*S, T2, 256)
        seg_out = seg_out.view(B, S, seg_out.shape[1], seg_out.shape[2])  # (B,S,T2,256) - Each beat recombined

        seg_feats, seg_alphas = self.time_distributed_attention(seg_out)  # (B,S,256)
        seq_out, _ = self.lstm_sequence(seg_feats)                        # (B,S,512)
        final_vec, final_alphas = self.final_attention(seq_out)           # (B,512),(B,S)

        x = F.relu(self.fc(final_vec))
        x = self.dropout(x)
        logits = self.classifier(x)                                       # (B,C)
        return logits, final_alphas, seg_alphas

# %%
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
                # Skip this class if AUC can't be computed
                continue
    
    auc = np.mean(auc_scores) if len(auc_scores) > 0 else 0.0
    
    return f1, hamming, precision, recall, auc

def train(model, train_loader, val_loader, optimizer, criterion, device, epochs=5):
    model.to(device)
    history = {"train_loss": [], "train_f1": [], "train_hamming": [], "train_precision": [], "train_recall": [], "train_auc": [],
               "val_loss": [], "val_f1": [], "val_hamming": [], "val_precision": [], "val_recall": [], "val_auc": []}
    
    for epoch in range(1, epochs + 1):
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
        
        if val_loader is not None:
            vloss, vf1, vhamming, vprecision, vrecall, vauc = evaluate(model, val_loader, criterion, device)
            history["val_loss"].append(vloss)
            history["val_f1"].append(vf1)
            history["val_hamming"].append(vhamming)
            history["val_precision"].append(vprecision)
            history["val_recall"].append(vrecall)
            history["val_auc"].append(vauc)
            tqdm.write(f"Epoch {epoch} — train_loss {history['train_loss'][-1]:.4f} | val_loss {vloss:.4f} | val_f1 {vf1:.4f} | val_auc {vauc:.4f} | val_hamming {vhamming:.4f} | val_precision {vprecision:.4f} | val_recall {vrecall:.4f}")
        else:
            history["val_loss"].append(None)
            history["val_f1"].append(None)
            history["val_hamming"].append(None)
            history["val_precision"].append(None)
            history["val_recall"].append(None)
            history["val_auc"].append(None)
        
        scheduler.step()
    
    return history

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

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    for seed in range(10, 110, 10):
        with mlflow.start_run(run_name=f"Seed {seed}"):
            set_global_seed(seed)
            df = load_mimic_metadata('records.csv', classes)
            df = df.reset_index(drop=True)

            all_indices = list(range(len(df)))
            trainval_indices, test_indices = train_test_split(all_indices, test_size=0.1, random_state=seed)
            train_indices,  val_indices    = train_test_split(trainval_indices, test_size=0.1, random_state=seed)

            train_loader = make_bucketed_loader(df, train_indices, batch_size=BATCH_SIZE, shuffle=True,  drop_last=DROP_LAST)
            val_loader   = make_bucketed_loader(df, val_indices,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
            test_loader  = make_bucketed_loader(df, test_indices,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

            train_loader_iter = iter(train_loader)
            model = HANWithAttention(num_classes=len(classes)).to(DEVICE)

            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=3e-5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

            history = train(
                model,
                train_loader,
                val_loader,
                optimizer,
                criterion,
                DEVICE,
                epochs=EPOCHS
            )

            test_loss, test_f1, test_hamming, test_prec, test_rec, test_auc = evaluate(
                model,
                test_loader,
                criterion,
                DEVICE
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