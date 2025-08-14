import os
import urllib
import tarfile
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from melspecs import Melspecs
from hyperparameters import hp


class LJSpeechDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Tensors
        seq = torch.tensor(row['seq'], dtype=torch.long)
        seq_len = torch.tensor(row['seq_len'], dtype=torch.long)
        mel = torch.tensor(row['mel'], dtype=torch.float)
        gate = torch.tensor(row['gate'], dtype=torch.float)

        # Add SOS mel frame
        sos_frame = torch.zeros((1, hp.n_mels), dtype=mel.dtype)
        mel = torch.cat([sos_frame, mel], dim=0)

        # Add SOS gate (0 = do not stop)
        gate = torch.cat([torch.zeros(1, dtype=gate.dtype), gate])

        mel_len = torch.tensor(row['mel_len'] + 1, dtype=torch.long)
        return seq, seq_len, mel, mel_len, gate


def collate_fn(batch):
    seqs, seq_lens, mels, mel_lens, gates = zip(*batch)
    seqs = pad_sequence(seqs, batch_first=True, padding_value=hp.PAD)
    seq_lens = torch.stack(seq_lens)
    mels = pad_sequence(mels, batch_first=True)
    mel_lens = torch.stack(mel_lens)
    gates = pad_sequence(gates, batch_first=True)
    return seqs, seq_lens, mels, mel_lens, gates


def download_ljspeech(dest_dir):
    url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    os.makedirs(dest_dir, exist_ok=True)
    archive_path = os.path.join(dest_dir, "LJSpeech-1.1.tar.bz2")
    extract_path = os.path.join(dest_dir, "LJSpeech-1.1")

    if not os.path.isdir(extract_path):
        if not os.path.isfile(archive_path):
            print("Downloading LJSpeech dataset...")
            urllib.request.urlretrieve(url, archive_path)
        print("Extracting dataset...")
        with tarfile.open(archive_path, mode='r:bz2') as tar:
            tar.extractall(path=dest_dir)

    return extract_path


def build_dataset(ljspeech_dir, pickle_path):
    """Process LJSpeech into mel-spectrograms"""
    device = hp.device
    melspecs = Melspecs(hp).to(device)

    meta_csv = os.path.join(ljspeech_dir, "metadata.csv")
    meta = pd.read_table(meta_csv, sep='|', header=None,
                         names=['id', 'text', 'text_norm'], dtype=str)
    wavs_dir = os.path.join(ljspeech_dir, 'wavs')

    rows = []
    for row in tqdm(meta.itertuples(index=False), total=len(meta), desc="Processing audio"):
        wav_path = os.path.join(wavs_dir, f"{row.id}.wav")

        mel = melspecs.wav_to_mel(wav_path)
        mel = mel.squeeze(0).transpose(1, 0).cpu().numpy()
        mel_len = mel.shape[0]

        gate = np.zeros(mel_len, dtype=np.float32)
        if mel_len > 0:
            gate[-1] = 1.0

        text_raw = str(row.text).lower()
        seq = hp.text_to_sequence(text_raw)
        seq_len = len(seq)

        rows.append({
            'seq': seq,
            'seq_len': seq_len,
            'mel': mel,
            'mel_len': mel_len,
            'gate': gate
        })

    df = pd.DataFrame(rows)
    df.to_pickle(pickle_path)
    print(f"Saved dataset to {pickle_path}")
    return df


def prepare_dataset():
    lj_dir = os.path.join(hp.data_dir, "LJSpeech-1.1")
    pickle_path = os.path.join(lj_dir, "LJSpeech_df.pkl")

    if not os.path.exists(pickle_path):
        if not os.path.isdir(lj_dir):
            lj_dir = download_ljspeech(hp.data_dir)
        df = build_dataset(lj_dir, pickle_path)
    else:
        print(f"Loading cached dataset from {pickle_path}")
        df = pd.read_pickle(pickle_path)

    print(f"Dataset ready: {len(df)} samples")
    return df


def get_dataloaders(df, batch_size, split=(0.9, 0.1)):
    dataset = LJSpeechDataset(df)
    total = len(dataset)
    train_size = int(split[0] * total)
    val_size = total - train_size

    generator = torch.Generator().manual_seed(hp.seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, pin_memory=True, num_workers=0
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, pin_memory=True, num_workers=0
    )

    return train_loader, val_loader
