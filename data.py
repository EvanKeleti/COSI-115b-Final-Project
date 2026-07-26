import itertools
import os
import re
from pathlib import Path
from typing import Generator, Iterable

import datasets
import torch
from datasets import Dataset
from dotenv import load_dotenv
from torch import Tensor

from preprocess import Preprocessor


CACHE_DIR = Path("cache/")


def load_wmt_data(split: str, indices: Iterable[int] = None):
    datasets.enable_progress_bars()
    dataset = datasets.load_dataset(
        "wmt/wmt19",
        "zh-en",
        split=split,
        streaming=False
    )
    if indices:
        dataset = dataset.select(indices)
    return dataset


def dataset_chunk_generator(dataset: Dataset, chunk_size: int) -> Generator[Dataset, None, None]:
    start_idx = 0
    while start_idx < len(dataset):
        if start_idx + chunk_size > len(dataset):
            chunk_size = len(dataset) - start_idx
        yield dataset.select(range(start_idx, start_idx + chunk_size))
        start_idx += chunk_size


def get_preprocessed_data(split: str, start_batch: int) -> list[Tensor]:
    chunk_dir = CACHE_DIR / split
    data = []
    prev_end = 0
    for p in sorted(chunk_dir.glob('*.pt')):
        m = re.search('_(.*)-(.*).pt', str(p))
        start, end = int(m.group(1)), int(m.group(2))
        # print(start, end)
        if start != prev_end:
            break
        prev_end = end
        if end < start_batch:
            continue
        chunk = torch.load(p, weights_only=False)
        if start_batch > start:
            data.extend(chunk[start_batch:])
        else:
            data.extend(chunk)
    return data


def preprocess_and_save_chunks(split: str, start_batch: int, num_chunks: int, pbar_pos: int = 0):
    batches_per_chunk = 100
    batch_size = 8
    chunk_dir = CACHE_DIR / split
    os.makedirs(chunk_dir, exist_ok=True)

    data_per_chunk = batches_per_chunk * batch_size
    start_datapoint = start_batch * batch_size

    ranges = set()
    for i in range(num_chunks):
        start = start_datapoint + i * data_per_chunk
        ranges.add((start, start + data_per_chunk))

    for p in sorted(chunk_dir.glob('*.pt')):
        m = re.search('_(.*)-(.*).pt', str(p))
        start, end = int(m.group(1)), int(m.group(2))
        ranges.discard((start, end))

    ranges = sorted(ranges)
    data = load_wmt_data(split).select(itertools.chain(*(range(start, end) for start, end in ranges)))

    langs = ['en', 'zh']
    preprocessor = Preprocessor(
        data,
        "xlm-roberta-base",
        langs,
        batch_size=8,
        output_batch_size=8,
        show_pbar=True,
        separate_thread=False,
        multithread=True,
        use_gpu=True,
        in_stages=False,
        verbose=False,
        pbar_pos=pbar_pos
    )

    try:
        range_idx = 0
        chunk = []
        for batch in preprocessor:
            chunk.append(batch)
            if len(chunk) == batches_per_chunk:
                start, end = ranges[range_idx][0] // batch_size, ranges[range_idx][1] // batch_size
                range_idx += 1
                torch.save(chunk, f"{chunk_dir}/chunk_{start:05d}-{end:05d}.pt")
                chunk = []
    finally:
        preprocessor.terminate_processors()


if __name__ == '__main__':
    load_dotenv()
    preprocess_and_save_chunks('train', 1200, 18)