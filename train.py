import gc
import os
from collections import deque
# Must be set before importing datasets
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # doesn't work

from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
from typing import Optional

import datasets
import torch
from accelerate import Accelerator
from dotenv import load_dotenv
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tornado.gen import multi
from tqdm import tqdm

from data import dataset_chunk_generator
from graph_builder import UPOS_MAP, REL_MAP, NER_MAP
from graph_encoder import RGATConfig
from preprocess import Preprocessor
from utils import plot_results

# os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/compiled_cache"

CHECKPOINT_BASE_DIR = Path("model/")

if torch.cuda.is_available():
    # print("Using GPU")
    device = torch.device("cuda")
else:
    # print("Using CPU")
    device = torch.device("cpu")



def train_on_batches(config: RGATConfig, batches, num_batches, checkpoint: dict, pbar, train_args) -> None:



    #### Variables used in loop
     = Path(checkpoint["dir"]) / "checkpoint.pt"
    batches_for_checkpoint = train_args.get('batches_for_checkpoint', 10)
    ####

    pbar.set_postfix_str("Training on chunk")
    for b, batch in tqdm(enumerate(batches, 1), desc=f"Training on {num_batches} batches", position=1, leave=False,
                         total=num_batches):
        # Context manager handles gradient accumulation math automatically
        with accelerator.accumulate(model):

            d.append((batch, outputs))

            accelerator.backward(loss)

            accumulated_loss_sum += accelerator.reduce(loss, reduction="mean").item()
            checkpoint["next_datapoint"] += scores.size(0)

            # If accumulated batch is finished
            if accelerator.sync_gradients:
                # Clip gradients
                # TODO - figure out cause of nonfinite gradients and see if can fix or if should just accept they will happen
                grad_norm: float = accelerator.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping).item()
                checkpoint["grad_norms"].append(grad_norm)
                # Get accumulated loss on whole batch for logging
                true_total_accumulated_loss = accumulated_loss_sum * accelerator.gradient_accumulation_steps
                checkpoint["losses"].append(true_total_accumulated_loss)
                accumulated_loss_sum = 0.0
                # NaN/Inf gradient detection is performed automatically through PyTorch's GradScaler
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # TODO if fast enough, only checkpoint after every chunk
                # Save progress every `batches_for_checkpoint` batches or at last batch
                if b % batches_for_checkpoint == 0 or b == num_batches:
                    # Plot losses and grad norms
                    plot_results(checkpoint["losses"], checkpoint["grad_norms"],
                                 save_path=str(Path(checkpoint["dir"]) / Path("losses_and_grad_norms.png")))
                    # Save checkpoint


                pbar.update(1)

    del model, optimizer, scheduler, accelerator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()



def train(
        model_config: RGATConfig,
        batch_size: int,
        checkpoint_dir: str | Path,
        peak_lr: float,
        preprocess_batch_size: int = 32,
        batches: Optional[int] = None,
        add_batches: bool = False,
        batches_per_chunk: int = 100,
        preprocess_before_chunk = False,
        # todo train args dataclass
        eval_chunk_interval: int = 5,
        accumulation_steps: int = 1,
        gradient_clipping: float = float('inf'),  # This lets us just get the grad norm but we don't clip
        batches_for_checkpoint: int = 10,
        # validation_batches=50,
) -> None:
    # Load saved progress
    tqdm.write("Attempting to load checkpoint file... ", end="")
    checkpoint_dir = Path(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    eval_checkpoint_dir = Path(checkpoint_dir) / Path("eval_checkpoints")
    os.makedirs(eval_checkpoint_dir, exist_ok=True)
    checkpoint_file = Path(checkpoint_dir) / "checkpoint.pt"


    # Get training data and select remaining datapoints

    datasets.enable_progress_bars()
    dataset = datasets.load_dataset("wmt/wmt19", "zh-en", streaming=False)
    train_data = dataset['train'].select(range(checkpoint["next_datapoint"], checkpoint["num_train_batches"] * batch_size))



    if not preprocess_before_chunk:
        batches_per_chunk = len(train_data) // batch_size - finished_batches

    with tqdm(initial=finished_batches, total=checkpoint["num_train_batches"], position=0, desc="Training",
              smoothing=0, miniters=1) as pbar:

        for c, chunk in enumerate(chunks, 1):
            # todo chunk pbar?
            # First preprocess entire chunk of dataset for max throughput
            # TODO - see if sending to cpu doesn't actual save enough VRAM to make faster
            batches = []
            # todo set_desc to Loading processors
            # todo join on processor instead of making list

            if preprocess_before_chunk:
                for batch in preprocessor:
                    # Send to cpu to not take up VRAM
                    batches.append(
                        {lang: {k: tensor.cpu() for k, tensor in tensors.items()}
                         for lang, tensors in batch.items()}
                    )
                assert len(batches) == batches_per_chunk, f"Expected {batches_per_chunk} batches, got {len(batches)}"
            else:
                batches = preprocessor
            # Train on data in this chunk
            train_on_batches(model_config, batches, batches_per_chunk, checkpoint, pbar, train_args)
            # finished_batches += len(batches)

            # TODO evaluate code
            # if c % eval_chunk_interval == 0:
            #     val_loss, val_acc = evaluate_model(model, val_loader, validation_batches, tqdm_pos=2)
            #     last_loss, last_acc = checkpoint.get("last_val_loss", 0), checkpoint.get("last_val_acc", 0)
            #     if no_eval_yet and last_loss:
            #         tqdm.write(f"Previous Validation Results: avg loss = {last_loss:.3f}, acc = {last_acc * 100:.2f}")
            #         no_eval_yet = False
            #     tqdm.write(f"Validation Results: avg loss = {val_loss:.2f}, acc = {val_acc * 100:.1f}")
            #
            #     checkpoint["last_val_loss"] = val_loss
            #     checkpoint["last_val_acc"] = val_acc
            #     # Save non-overwritten checkpoint after every eval
            #     torch.save(checkpoint, eval_checkpoint_dir / Path(f"batch_{finished_batches}.pt"))


# todo - use 'steps' and 'batches' to distinguish accumulation
# todo what if don't need to del processor - can it automatically get of gpu to make room for model?
# todo - looks like sinkhorn might actually be the bottleneck

def main():
    load_dotenv()

    config = RGATConfig(
        n_layers=2,
        n_heads=16,
        d_upos=16,
        num_upos=len(UPOS_MAP),
        d_ner=16,
        d_lang=16,
        num_ner=len(NER_MAP),
        d_relations=32,
        num_relations=len(REL_MAP) * 2,  # times two since reverse edges are included
        langs=["zh", "en"],
    )

    # TODO figure out max batches gpu can actually handle (12 > 8 even if not 16)
    train(
        model_config=config,
        batch_size=16,
        checkpoint_dir=CHECKPOINT_BASE_DIR / Path(str(config)),
        peak_lr=1e-4,
        preprocess_batch_size=48,
        preprocess_before_chunk=True,
        batches=10000,
        batches_per_chunk=100,
        add_batches=False,
        accumulation_steps=1,  # with InfoNCE, not true accumulation - TODO alternative methods (e.g. MoCo style queue)
        gradient_clipping=1,
        batches_for_checkpoint=50,
        eval_chunk_interval=5,
        # validation_batches=50,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
