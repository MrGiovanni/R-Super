#pedro
import math
import random
import torch
from torch.utils.data import Sampler

class ChunkedSampler(Sampler):
    """
    A custom sampler that:
      - Divides the dataset into `cycle_length` chunks, each chunk up to `samples_per_epoch` in size.
      - Each "cycle" covers the entire dataset exactly once (with possible padding in the last chunk).
      - Shuffles the dataset *only once* per cycle (i.e., after every `cycle_length` epochs).
      - If using DDP, each rank sees a non-overlapping subset of the chunk, so the total 
        across all ranks is `samples_per_epoch` for each epoch in the cycle.

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
        world_size: int = 1
    ):
        """
        Args:
            dataset_size (int): total number of items in the dataset.
            samples_per_epoch (int): how many samples define one "epoch subset".
            shuffle (bool): whether to shuffle once per cycle.
            seed (int): base seed for shuffling.
            rank (int): the rank of this process (0 if single GPU).
            world_size (int): total number of processes (1 if single GPU).
        """
        super().__init__(None)
        self.dataset_size = dataset_size
        self.samples_per_epoch = samples_per_epoch
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        

        # We'll store a single "shuffled_indices" for the current cycle
        self.shuffled_indices = list(range(self.dataset_size))
        
        # How many epochs it takes to go through the entire dataset exactly once
        # Each epoch draws 'samples_per_epoch' items (the chunk).
        # If there's leftover smaller than 'samples_per_epoch', that last chunk is padded.
        self.cycle_length = math.ceil(self.dataset_size / self.samples_per_epoch)

        # Track the current epoch so we know which chunk to pick
        self.epoch = 0
        # Track the current cycle so we know when to re-shuffle
        self.cycle = -1  # force an initial shuffle at the start
        
        print('Sampler: dataset_size:', self.dataset_size)
        print('Sampler: samples_per_epoch:', self.samples_per_epoch)
        print('Sampler: cycle_length:', self.cycle_length)
        print('Sampler: rank:', self.rank)
        print('Sampler: world_size:', self.world_size)

    def set_epoch(self, epoch: int):
        """Called by the training loop at the start of each epoch."""
        self.epoch = epoch

    def __iter__(self):
        """
        Generate the sample indices for this epoch on this rank only.
        """
        # 1) Figure out which cycle we're in
        new_cycle = self.epoch // self.cycle_length

        # 2) If we've moved to a new cycle, re-shuffle (once per cycle).
        if new_cycle != self.cycle:
            self.cycle = new_cycle
            if self.shuffle:
                g = torch.Generator()
                # combine base seed + self.cycle so each cycle is reproducibly unique
                g.manual_seed(self.seed + self.cycle)
                # shuffle in-place or create a shuffled list
                idx_tensor = torch.randperm(self.dataset_size, generator=g)
                self.shuffled_indices = idx_tensor.tolist()
            else:
                # if no shuffle, just keep in ascending order each cycle
                self.shuffled_indices = list(range(self.dataset_size))

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

        # 6) Finally, we subdivide for DDP rank => rank gets a round-robin subset
        rank_indices = chunk[self.rank :: self.world_size]

        return iter(rank_indices)

    def __len__(self):
        """
        Number of samples THIS RANK processes in one epoch.
        Typically ~ samples_per_epoch / world_size, 
        possibly 1 off if not evenly divisible.
        """
        return math.ceil(self.samples_per_epoch / self.world_size)