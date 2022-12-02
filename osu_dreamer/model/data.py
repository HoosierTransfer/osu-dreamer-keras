import os
import random

from pathlib import Path
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm

import numpy as np
import scipy.stats
import librosa


import torch
import torchaudio
from torch.utils.data import IterableDataset, DataLoader, random_split

import pytorch_lightning as pl

from osu_dreamer.osu.beatmap import Beatmap
from osu_dreamer.signal import (
    from_beatmap as signal_from_beatmap,
    timing_signal as beatmap_timing_signal,
)

# audio processing constants
N_FFT = 2048
HOP_LEN_S = 128. / 22050 # ~6 ms per frame
N_MELS = 64

A_DIM = 40

# check if using WSL
if os.system("uname -r | grep microsoft > /dev/null") == 0:
    def reclaim_memory():
        """
        free the vm page cache - see `https://devblogs.microsoft.com/commandline/memory-reclaim-in-the-windows-subsystem-for-linux-2/`
        
        add to /etc/sudoers:
        %sudo ALL=(ALL) NOPASSWD: /bin/tee /proc/sys/vm/drop_caches
        """
        os.system("echo 1 | sudo tee /proc/sys/vm/drop_caches > /dev/null")
else:
    def reclaim_memory():
        pass

def load_audio(audio_file):
    wave, sr = librosa.load(audio_file)

    # compute spectrogram
    spec: "A,L" = torchaudio.transforms.MFCC(
        sample_rate=sr, 
        n_mfcc=A_DIM,
        melkwargs=dict(
            normalized=True,
            n_fft=N_FFT,
            hop_length=int(HOP_LEN_S * sr),
            n_mels=N_MELS,
        ),
    )(torch.tensor(wave)).numpy()
    
    return spec, sr

def prepare_map(data_dir, map_file):
    try:
        bm = Beatmap(map_file, meta_only=True)
    except Exception as e:
        print(f"{map_file}: {e}")
        return

    af_dir = "_".join([bm.audio_filename.stem, *(s[1:] for s in bm.audio_filename.suffixes)])
    map_dir = data_dir / map_file.parent.name / af_dir
    
    spec_path =  map_dir / "spec.pt"
    timing_path = map_dir / "timing.pt"
    plp_path = map_dir / "plp.pt"
    map_path = map_dir / f"{map_file.stem}.map.pt"
    
    if map_path.exists():
        return
    
    try:
        bm.parse_map_data()
    except Exception as e:
        print(f"{map_file}: {e}")
        return

    if spec_path.exists():
        # determine audio sample rate
        sr = torchaudio.info(bm.audio_filename).sample_rate
        spec = np.load(spec_path)
    else:
        # load audio file
        try:
            spec, sr = load_audio(bm.audio_filename)
        except Exception as e:
            print(f"{bm.audio_filename}: {e}")
            return

        # save spectrogram
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        with open(spec_path, "wb") as f:
            np.save(f, spec, allow_pickle=False)
            
    frame_times = librosa.frames_to_time(
        np.arange(spec.shape[-1]),
        sr=sr, hop_length=int(HOP_LEN_S * sr), n_fft=N_FFT,
    ) * 1000
    
    # compute timing signal
    if not timing_path.exists():
        timing: "T,L" = beatmap_timing_signal(bm, frame_times)
        with open(timing_path, "wb") as f:
            np.save(f, timing, allow_pickle=False)
            
    # compute plp
    if not plp_path.exists():
        plp: "T,L" = librosa.beat.plp(
            onset_envelope=librosa.onset.onset_strength(
                S=spec, center=False,
            ),
            prior = scipy.stats.lognorm(loc=np.log(180), scale=180, s=1),
            # use 10s of audio to determine local bpm
            win_length=int(10. / HOP_LEN_S), 
        )[None]
        with open(plp_path, "wb") as f:
            np.save(f, plp, allow_pickle=False)

    # compute map signal
    x: "X,L" = signal_from_beatmap(bm, frame_times)
    with open(map_path, "wb") as f:
        np.save(f, x, allow_pickle=False)
    

class Data(pl.LightningDataModule):
    def __init__(
        self,
        
        seq_depth: int,
        sample_density: float,
        subseq_density: float,
        batch_size: int,
        num_workers: int,
        
        data_path: str = "./data",
        src_path: str = None,
        val_split: float = None,
        val_size: int = None,
    ):
        super().__init__()
        
        self.seq_len = 2 ** seq_depth
        
        self.sample_density = sample_density
        self.subseq_density = subseq_density
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        if (val_split is None) == (val_size is None):
            raise ValueError('exactly one of `val_split` or `val_size` must be specified')
        self.val_split = val_split
        self.val_size = val_size
        
        self.data_dir = Path(data_path)
        try:
            self.data_dir.mkdir()
        except FileExistsError:
            # data dir exists, check for samples
            try:
                next(self.data_dir.rglob("*.map.pt"))
                # data dir already has samples
                self.src_dir = None
                return
            except StopIteration:
                # data dir is empty, must prepare data
                pass
            
        if src_path is None:
            raise ValueError("`src_path` must be specified when `data_dir` does not exist or is empty")
        self.src_dir = Path(src_path)
        
        
    def prepare_data(self):
        if self.src_dir is None:
            return
        
        src_maps = list(self.src_dir.rglob("*.osu"))
        print(f"{len(src_maps)} osu! beatmaps found, processing...")
        with Pool(processes=self.num_workers) as p:
            for _ in tqdm(p.imap_unordered(partial(prepare_map, self.data_dir), src_maps), total=len(src_maps)):
                reclaim_memory()
            
    def setup(self, stage: str):
        full_set = list(self.data_dir.rglob("*.map.pt"))
        
        if self.val_size is not None:
            val_size = self.val_size
        else:
            val_size = int(len(full_set) * self.val_split)
            
        if val_size < 0:
            raise ValueError("`val_size` is negative")
            
        if val_size > len(full_set):
            raise ValueError(f"`val_size` ({val_size}) is greater than the number of samples ({len(full_set)})")
            
        train_size = len(full_set) - val_size
        print(f'train: {train_size} | val: {val_size}')
        train_split, val_split = random_split(full_set, [train_size, val_size])
        
        self.train_set = SubsequenceDataset(
            dataset=train_split,
            seq_len=self.seq_len,
            sample_density=self.sample_density,
            subseq_density=self.subseq_density,
        )
        self.val_set = FullSequenceDataset(dataset=val_split)
            
    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=1,
            num_workers=self.num_workers,
        )
    
    
class StreamPerSample(IterableDataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.dataset = kwargs.pop("dataset")
        self.sample_density = kwargs.pop("sample_density", 1.)
        
        if not 0 < self.sample_density <= 1:
            raise ValueError("sample density must be in (0, 1]:", self.sample_density)
            
        if len(kwargs):
            raise ValueError(f"unexpected kwargs: {kwargs}")
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  
            # single-process data loading, return the full iterator
            num_workers = 1
            worker_id = 0
            seed = torch.initial_seed()
        else:  # in a worker process
            # split workload
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            seed = worker_info.seed
        
        random.seed(seed)
        
        dataset = sorted(self.dataset)
        for i, sample in random.sample(list(enumerate(dataset)), int(len(dataset) * self.sample_density)):
            if i % num_workers != worker_id:
                continue
                
            try:
                for x in self.sample_stream(sample):
                    yield x
            finally:
                reclaim_memory()
                
                
def load_tensors_for_map(map_file):
    a: "A,L" = torch.tensor(np.load(map_file.parent / "spec.pt")).float()
    t: "T,L" = torch.tensor(np.load(map_file.parent / "timing.pt")).float()
    p: "T,L" = torch.tensor(np.load(map_file.parent / "plp.pt")).float()
    x: "X,L" = torch.tensor(np.load(map_file)).float()
    return a,t,p,x

class FullSequenceDataset(StreamPerSample):
    MAX_LEN = 60000
            
    def sample_stream(self, map_file):
        yield tuple([ 
            x[...,:self.MAX_LEN]
            # x[...,:self.MAX_LEN] if x.size(-1) > self.MAX_LEN else F.pad(x, (0, self.MAX_LEN - x.size(-1)))
            for x in load_tensors_for_map(map_file)
        ])     
        
class SubsequenceDataset(StreamPerSample):
    def __init__(self, **kwargs):
        self.seq_len = kwargs.pop("seq_len")
        self.subseq_density = kwargs.pop("subseq_density", 2)
        super().__init__(**kwargs)

    def sample_stream(self, map_file):
        tensors = load_tensors_for_map(map_file)
        L = tensors[0].size(-1)

        if self.seq_len >= L:
            return

        num_samples = int(L / self.seq_len * self.subseq_density)

        for idx in torch.randperm(L - self.seq_len)[:num_samples]:
            yield tuple([
                x[..., idx:idx+self.seq_len].clone()
                for x in tensors
            ])
