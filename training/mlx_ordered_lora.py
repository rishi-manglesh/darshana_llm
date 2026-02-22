#!/usr/bin/env python3
"""MLX LoRA training wrapper that preserves data ordering.

mlx_lm's iterate_batches() sorts examples by token length and then shuffles
batches randomly — destroying any intentional corpus ordering (Samkhya, Bloom).

This wrapper monkey-patches iterate_batches() to iterate sequentially,
preserving the order from train.jsonl.

Usage (drop-in replacement for `python -m mlx_lm lora`):
  python training/mlx_ordered_lora.py --model ... --data ... --train ...
"""

import numpy as np
import mlx.core as mx
from mlx_lm.tuner.trainer import CacheDataset
import mlx_lm.tuner.trainer as trainer_module


def iterate_batches_ordered(
    dataset,
    batch_size,
    max_seq_length,
    loop=False,
    seed=None,
    comm_group=None,
):
    """Sequential iterate_batches — preserves dataset ordering.

    Key differences from original:
    1. No sorting by token length (preserves corpus order)
    2. No random permutation of batches (sequential iteration)
    """
    n = len(dataset)
    if n < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size}"
            f" examples but only has {n}."
        )

    if comm_group is not None:
        offset = comm_group.rank()
        step = comm_group.size()
    else:
        offset = 0
        step = 1
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")

    # Sequential indices — NO sorting by length
    idx = list(range(n))

    # Make batches in sequential order
    batch_idx = [
        idx[i + offset : i + offset + batch_size : step]
        for i in range(0, n - batch_size + 1, batch_size)
    ]

    while True:
        # Sequential iteration — NO random permutation
        for i in range(len(batch_idx)):
            batch = [dataset[j] for j in batch_idx[i]]
            if len(batch[0]) == 2:
                batch, offsets = zip(*batch)
            else:
                offsets = [0] * len(batch)
            lengths = [len(x) for x in batch]
            if max(lengths) > max_seq_length:
                print(
                    f"[WARNING] Some sequences are longer than {max_seq_length} tokens. "
                    f"The longest sentence {max(lengths)} will be truncated to {max_seq_length}. "
                    "Consider pre-splitting your data to save memory."
                )

            pad_to = 32
            max_length_in_batch = 1 + pad_to * ((max(lengths) + pad_to - 1) // pad_to)
            max_length_in_batch = min(max_length_in_batch, max_seq_length)

            batch_arr = np.zeros((batch_size // step, max_length_in_batch), np.int32)

            for j in range(batch_size // step):
                truncated_length = min(lengths[j], max_seq_length)
                batch_arr[j, :truncated_length] = batch[j][:truncated_length]
                lengths[j] = truncated_length
            batch = mx.array(batch_arr)
            yield batch, mx.array(list(zip(offsets, lengths)))

        if not loop:
            break


if __name__ == "__main__":
    # Monkey-patch before importing main
    trainer_module.iterate_batches = iterate_batches_ordered
    print("[PATCH] iterate_batches replaced with ordered (sequential) version")

    from mlx_lm.lora import main
    main()
