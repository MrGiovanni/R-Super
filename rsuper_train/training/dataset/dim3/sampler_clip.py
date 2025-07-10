#pedro

from __future__ import annotations
import math
import random
import torch
from torch.utils.data import BatchSampler
import copy
from typing import Dict, List
from typing import Optional
import os
from typing import Dict, List, Optional, Tuple
import yaml
from pathlib import Path
from pathlib import Path
from typing import List, Tuple
import pandas as pd

# ──────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────
def _scan_yaml(folder: Path) -> List[Tuple[str, str]]:
    """
    Collect (key, value) pairs from *_crop.yaml files located in *folder*.
      • key   = stem of the file (minus “_crop”), e.g. 'BDMAP_00012345'
      • value = first element of the YAML list
    Non‑BDMAP files are ignored.
    """
    pairs: List[Tuple[str, str]] = []

    for yf in folder.glob("*_crop.yaml"):
        if "BDMAP" not in yf.name:                       # skip rubbish
            continue
        try: 
            with yf.open() as f:
                data = yaml.safe_load(f) or []
            if isinstance(data, list) and data:              # we expect ['organ']
                pairs.append((yf.stem.replace("_crop", ""), str(data[0])))
        except Exception as e:
            print(f"Warning: could not read {yf}: {e}", flush=True)
            continue

    return pairs


# ──────────────────────────────────────────────────────────────
# "build_db"  ➜  returns a DataFrame instead of a file
# ──────────────────────────────────────────────────────────────
def build_df(yaml_folder: Path) -> pd.DataFrame:
    """
    Scan *yaml_folder* and return a DataFrame with two string columns:
        key   | value
        ------------- 
        BDMAP | organ
    If the folder is empty the returned DataFrame is empty as well.
    """
    print(f"Scanning {yaml_folder} for *_crop.yaml files...", flush=True)
    rows = _scan_yaml(Path(yaml_folder))
    df   = pd.DataFrame(rows, columns=["key", "value"], dtype="string")
    print(f"Found {len(rows)} *_crop.yaml files.", flush=True)
    return df.set_index("key", drop=False)   # keep 'key' both as index & col


# ──────────────────────────────────────────────────────────────
# Data‑frame‑based helpers (replacing LiveSQLiteKV methods)
# ──────────────────────────────────────────────────────────────
def df_get_all_keys(df: pd.DataFrame, *, exclude_random: bool = True) -> List[str]:
    """
    Return all BDMAP_IDs contained in *df*.
    If *exclude_random* == True records whose *value* equals 'random'
    are filtered out.
    """
    if exclude_random:
        return df.loc[df["value"] != "random", "key"].tolist()
    return df["key"].tolist()


def df_get(df: pd.DataFrame, key: str) -> Optional[str]:
    """
    Return the organ (value) for *key* or **None** if the key is absent.
    """
    try:
        return df.at[key, "value"]
    except KeyError:
        return None


def df_invert(
    df: pd.DataFrame, *, allowed_ids: List[str] | None = None
) -> Dict[str, List[str]]:
    """
    Build an inverse mapping   organ  ➜  [BDMAP_ID, …]

    If *allowed_ids* is not None the lists are filtered so that only
    those IDs are kept.
    """
    out: Dict[str, List[str]] = {}
    if allowed_ids is None:
        grouped = df.groupby("value")["key"]
        for organ, series in grouped:
            out[organ] = series.tolist()
    else:
        allowed = set(allowed_ids)
        for organ, series in df.groupby("value")["key"]:
            ids = [k for k in series.tolist() if k in allowed]
            if ids:
                out[organ] = ids
    return out

class one_organ_per_batch_sampler(BatchSampler):
    """
    A custom sampler that:
      - Divides the dataset into `cycle_length` chunks, each chunk up to `samples_per_epoch` in size.
      - Each "cycle" covers the entire dataset exactly once (with possible padding in the last chunk).
      - Shuffles the dataset *only once* per cycle (i.e., after every `cycle_length` epochs).
      - If using DDP, each rank sees a non-overlapping subset of the chunk, so the total 
        across all ranks is `samples_per_epoch` for each epoch in the cycle.
      - Due to the clip loss, we want each batch to focus on one organ at a time. So, we read a SQLite dataset,
        which explains what is the crop organ for each saved crop we have. In trianing we randomly select one 
        unseen sample, then we get its crop organ and use it to get the other batch elements. 
        We remove the batch elements from the unseen list, and we repeat the process.

    Usage:
      >>> sampler = ChunkedSamplerApproachA(
      ...     dataset_size=len(train_dataset),
      ...     samples_per_epoch=200,
      ...     shuffle=True,
      ...     seed=42,
      ...     rank=dist.get_rank() if ddp else 0,
      ...     world_size=dist.get_world_size() if ddp else 1
      ... )
      >>> train_loader = DataLoader(train_dataset, sampler=sampler, ...)
      
      for epoch in range(num_epochs):
          sampler.set_epoch(epoch)
          for batch in train_loader:
              ...
    """

    def __init__(
        self,
        dataset_size: int,
        samples_per_epoch: int,
        shuffle: bool = True,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        dataset = None,
        batch_size = None
    ):
        """
        Args:
            dataset_size (int): total number of items in the dataset.
            samples_per_epoch (int): how many samples define one "epoch subset".
            shuffle (bool): whether to shuffle once per cycle.
            seed (int): base seed for shuffling.
            rank (int): the rank of this process (0 if single GPU).
            world_size (int): total number of processes (1 if single GPU).
            batch_size (int): total size of each batch, NOT per-GPU (total batch size).
        """
        super().__init__(self, batch_size, drop_last=False)
        itens = dataset.img_list
        self.itens = [
            item[item.find('BDMAP_'):item.find('BDMAP_') + len('BDMAP_00001111')]
            for item in itens
        ]
        self.path = dataset.save_destination
        self.df = build_df(self.path)
        
        self.samples_per_epoch = samples_per_epoch
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        
        # Track the current epoch so we know which chunk to pick
        self.epoch = 0
        self.update_dataset(-1)
        
        #dict to convert bdmap_id into dataset idx
        self.id_to_idx = {bd: idx for idx, bd in enumerate(self.itens)}
        
        print('Sampler: samples_per_epoch:', self.samples_per_epoch)
        print('Sampler: cycle_length:', self.cycle_length)
        print('Sampler: rank:', self.rank)
        print('Sampler: world_size:', self.world_size)
        
        assert batch_size % world_size == 0, "batch_size must be divisible by world_size"
        assert batch_size >= world_size, "batch_size must be more than or equal to world_size"

    def set_epoch(self, epoch: int):
        """Called by the training loop at the start of each epoch."""
        self.epoch = epoch
    
    def update_dataset(self,cycle):
        self.cycle = cycle
        self.df = build_df(self.path)
        #which IDs are already in the DB (crops saved)
        saved_ids = df_get_all_keys(self.df , exclude_random=True)
        #raise ValueError(f'Number of saved IDs: {len(saved_ids)}')
        self.saved_itens = [i for i in self.itens if i in saved_ids]
        self.dataset_size = len(self.saved_itens) #we consider only the saved IDs
        
        

        # How many epochs it takes to go through the entire dataset exactly once
        # Each epoch draws 'samples_per_epoch' items (the chunk).
        # If there's leftover smaller than 'samples_per_epoch', that last chunk is padded.
        self.cycle_length = math.ceil(self.dataset_size / self.samples_per_epoch)
        
        
        
        if self.shuffle:
            g = torch.Generator()
            # combine base seed + self.cycle so each cycle is reproducibly unique
            g.manual_seed(self.seed + self.cycle)
            # shuffle in-place or create a shuffled list
            j = torch.randperm(self.dataset_size, generator=g).tolist()
            self.shuffled_indices = [self.saved_itens[i] for i in j]
        else:
            # if no shuffle, just keep in ascending order each cycle
            self.shuffled_indices = self.saved_itens
        

    def __iter__(self):
        """
        Generate the sample indices for this epoch on this rank only.
        """
        # 1) Figure out which cycle we're in
        new_cycle = self.epoch // self.cycle_length

        # 2) If we've moved to a new cycle, re-shuffle (once per cycle).
        if new_cycle != self.cycle:
            self.update_dataset(new_cycle)
            
        # 3) Within this cycle, pick which chunk we are in
        within_cycle = self.epoch % self.cycle_length
        start = within_cycle * self.samples_per_epoch
        end = start + self.samples_per_epoch

        # 4) Slice out that chunk
        chunk = self.shuffled_indices[start : min(end, self.dataset_size)]
        

        # 5) If this chunk is smaller than samples_per_epoch (the last chunk in the cycle),
        #    pad it so we always have exactly samples_per_epoch total.
        shortfall = self.samples_per_epoch - len(chunk)
        if shortfall > 0:
            # pad from the rest of this cycle's shuffled_indices
            # e.g. from the front portion [0:start] or the tail [end:].
            # We'll keep it simple: pad from anywhere else in the dataset:
            # (some prefer "randomly sample from leftover," up to you).
            # We'll do random.choices for consistent usage.
            # If you don't want randomness, you can just do chunk.extend(self.shuffled_indices[0:shortfall]).
            pool = self.shuffled_indices[:start] + self.shuffled_indices[end:]
            if len(pool) == 0:
                # fallback if entire dataset fits in chunk
                pool = self.shuffled_indices
            extra = random.choices(pool, k=shortfall)
            chunk.extend(extra)

        # Now chunk is exactly samples_per_epoch in length.

        #within this epoch chunk, we create our batches, with one organ per batch. And the same organ for all GPUs.
        
        #logic: for each batch, we select 1 index and get its crop organ. Then, we get the other batch elements as samples with this same organ.
        #Use use an unseen list to store the indices of the samples that have already been used in this epoch.
        #chunk is the epoch BDMAP IDs
        unseen_ids = copy.deepcopy(chunk)
        
        num_batches = math.ceil(self.samples_per_epoch / self.batch_size)
        for b in list(range(num_batches)):
            if b%100==0:
                self.df = build_df(self.path)
            if len(unseen_ids) < self.batch_size:
                print(f'CAREFUL: {len(unseen_ids)} samples left. Randomly choosing batch.',flush=True)
                batch = unseen_ids
                #pad until batch_size
                batch.extend(random.choices(chunk, k=self.batch_size - len(unseen_ids)))
                for i in batch:
                    if i in unseen_ids:
                        unseen_ids.remove(i)
            else:    
                #begin by getting the first index of the batch
                chosen_one = unseen_ids[0]
                batch = [chosen_one]
                #crop_organ for the chosen one
                crop_organ = df_get(self.df,chosen_one)
                #now let's read the sqlite database to get the organ of this index
                organ_dict = df_invert(self.df,allowed_ids=unseen_ids)
                #grab the first batch_size - 1 elements of the organ_dict[crop_organ] list 
                extended=False
                if len(organ_dict[crop_organ]) >= self.batch_size - 1:
                    candidates = list(set(organ_dict[crop_organ]))
                    if chosen_one in candidates:
                        #remove the chosen one from the candidates list
                        candidates.remove(chosen_one)
                    if len(candidates) >= self.batch_size - 1:
                        #get the first batch_size - 1 elements of the candidates list
                        batch.extend(candidates[:self.batch_size - 1])
                        extended=True
                if not extended:
                    #just get the next batch elements from the unseen_ids list
                    batch= unseen_ids[:self.batch_size]
                    print(f'CAREFUL: {crop_organ} has only {len(organ_dict[crop_organ])} samples. Randomly choosing batch.',flush=True)
                #remove the batch elements from the unseen_ids list
                for i in batch:
                    unseen_ids.remove(i)
                    
            #convert the BDMAP_IDs to indices
            batch = [self.id_to_idx[i] for i in batch]
            # now slice per-rank, round-robin style
            local_batch = batch[self.rank :: self.world_size]
            yield local_batch
            
            

    def __len__(self):
        """
        how many batches per epoch
        """
        return math.ceil(self.samples_per_epoch / self.batch_size)