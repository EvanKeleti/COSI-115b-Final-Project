import os
from pathlib import Path
from typing import Optional

import datasets
import torch
from accelerate import Accelerator
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import evaluate_model
from graph_builder import UPOS_MAP, REL_MAP, NER_MAP
from model import GraphMatcher, ModelConfig
from utils import plot_results

# os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/compiled_cache"
CHECKPOINT_DIR = Path("model/3_layers")
EVAL_CHECKPOINT_DIR = CHECKPOINT_DIR / Path("eval_checkpoints")

if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")


def train(
        start_lr: float,
        model_config: ModelConfig,
        batch_size: int,
        max_batches: Optional[int] = None,
        gradient_clipping: float = float('inf'),  # This lets us just get the grad norm but we don't clip
        accumulation_steps: int = 1,
        batches_for_checkpoint: int = 10,
        eval_interval_batches: int = 2000,
        validation_batches=50,
) -> None:
    assert batch_size % accumulation_steps == 0, "Batch size must be divisible by accumulation steps"
    from preprocess import graph_collate_fn
    # To track losses and grad norms for plotting
    accumulated_loss_sum = 0.0
    # Initialize model, optimizer, and scheduler
    model = GraphMatcher(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr)
    # Cosine Annealing Schedule with Linear Warmup is considered the industry gold standard for InfoNCE loss
    # I trained for a while at lr=1e-4 without linear warmup, so I switched directly to cosine annealing (at batch 4180)
    #  - Cosine Annealing made loss higher? And model less accurate? Why?
    #  - maybe I did it too soon, and rate of learning did not match rate of new data
    #  - but that doesn't explain why accuracy is so much worse on validation set (maybe just unlucky batches?)
    # At batch 5660 - reset lr to 1e-4, cosine annealing with 10k T_max, set InfoNCE temp to learnable parameter
    # Generally not good to restart cosine curve from scratch (I did it anyways) (also reset optimizer cause new param)
    # Now restarting training with 3 layers of RGAT
    # scheduler = CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-6)
    total_steps = max_batches
    warmup_steps = total_steps // 10  # 10% warmup
    decay_steps = total_steps - warmup_steps
    # Warmup Scheduler: Linearly increase LR from 10% of max to max
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    # Okay learning rate is too slow with annealing. I'm just going to do flat learning rate after warmup
    # # Decay Scheduler: Cosine decay over remaining steps down to a minimum fraction
    # decay_scheduler = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=1e-6)
    # # Combine them sequentially
    # scheduler = SequentialLR(
    #     optimizer,
    #     schedulers=[warmup_scheduler, decay_scheduler],
    #     milestones=[warmup_steps]
    # )
    scheduler = warmup_scheduler
    # Load saved progress
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(EVAL_CHECKPOINT_DIR, exist_ok=True)
    checkpoint_file = CHECKPOINT_DIR / "checkpoint.pt"
    if checkpoint_file.exists():
        # checkpoint = torch.load("model/eval_checkpoints/batch_4000.pt", weights_only=True)
        checkpoint = torch.load(checkpoint_file, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    else:
        checkpoint = {"next_datapoint": 0, "losses": [], "grad_norms": [], "model_config": model_config}
    # # Manually alter the learning rate to 1e-4 mid-training
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = 1e-4
    print('=' * 100)
    if checkpoint.get("losses", False):
        print(f"Checkpoint found. Resuming training at datapoint {checkpoint['next_datapoint']}.")
    else:
        print(f"No checkpoint found. Starting training at datapoint 0.")
    print('=' * 100)
    # Load dataset
    datasets.enable_progress_bars()
    dataset = datasets.load_dataset("wmt/wmt19", "zh-en", streaming=False)
    # Get training data and select remaining datapoints
    if checkpoint["next_datapoint"] == 0:
        remaining_train_data = dataset['train']
    else:
        remaining_train_data = dataset['train'].select(range(checkpoint["next_datapoint"], len(dataset['train'])))
    # TODO - figure out way for preprocessing to run in parallel to model (if it isn't already)
    # Training and validation dataloaders
    sub_batch = batch_size // accumulation_steps
    train_loader = DataLoader(remaining_train_data, batch_size=sub_batch, collate_fn=graph_collate_fn, shuffle=False)
    val_loader = DataLoader(dataset['validation'], batch_size=sub_batch, collate_fn=graph_collate_fn, shuffle=True)

    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=accumulation_steps,  # Simulates a large batch size using small steps
    )
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    train_iter = iter(train_loader)
    # val_iter = iter(val_loader)
    model.train()

    total_batches = len(dataset['train']) // batch_size
    finished_batches = checkpoint['next_datapoint'] // batch_size
    remaining_batches = total_batches - finished_batches
    max_batches = min(max_batches, remaining_batches) if max_batches is not None else remaining_batches
    current_batch = finished_batches + 1

    # TODO combining graph encodings from both sub-batches means that when scoring, each positive pair will be constrasted
    #  with more negative examples -> stronger learning
    validate = False
    no_eval_yet = True
    with tqdm(initial=finished_batches, total=total_batches, position=0,
              desc="Batches trained on out of total, at latest checkpoint") as epoch_bar:
        for b in tqdm(range(current_batch, current_batch + max_batches), desc="Training", total=max_batches,
                      position=1):

            for i in range(accumulation_steps):
                batch = next(train_iter)
                # Context manager handles gradient accumulation math automatically
                with accelerator.accumulate(model):
                    scores = model(batch)
                    loss = model.get_loss_on_batch(scores)

                    accelerator.backward(loss)

                    accumulated_loss_sum += accelerator.reduce(loss, reduction="mean").item()

                    # If accumulated batch is finished
                    if accelerator.sync_gradients:
                        # Clip gradients
                        # TODO - figure out cause of nonfinite gradients and see if can fix or if should just accept they will happen
                        grad_norm: float = accelerator.clip_grad_norm_(model.parameters(),
                                                                       max_norm=gradient_clipping).item()
                        checkpoint["grad_norms"].append(grad_norm)
                        # Get accumulated loss on whole batch for logging
                        true_total_accumulated_loss = accumulated_loss_sum * accelerator.gradient_accumulation_steps
                        checkpoint["losses"].append(true_total_accumulated_loss)
                        accumulated_loss_sum = 0.0
                        # NaN/Inf gradient detection is performed automatically through PyTorch's GradScaler
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

            if b % eval_interval_batches == 0:
                validate = True
                val_loss, val_acc = evaluate_model(model, val_loader, validation_batches, tqdm_pos=2)

                last_loss, last_acc = checkpoint.get("last_val_loss", 0), checkpoint.get("last_val_acc", 0)
                if no_eval_yet and last_loss:
                    tqdm.write(f"Previous Validation Results: avg loss = {last_loss:.3f}, acc = {last_acc * 100:.2f}")
                    no_eval_yet = False
                tqdm.write(f"Validation Results: avg loss = {val_loss:.2f}, acc = {val_acc * 100:.1f}")

                checkpoint["last_val_loss"] = val_loss
                checkpoint["last_val_acc"] = val_acc
                model.train()

            # Save progress every `batches_for_checkpoint` batches
            if b % batches_for_checkpoint == 0:
                # Plot losses and grad norms
                plot_results(checkpoint["losses"], checkpoint["grad_norms"],
                             save_path=str(CHECKPOINT_DIR / Path("losses_and_grad_norms.png")))
                # Save checkpoint
                checkpoint["next_datapoint"] += batch_size * batches_for_checkpoint
                checkpoint["model_state_dict"] = model.state_dict()
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()
                torch.save(checkpoint, checkpoint_file)
                # Save non-overwritten checkpoint after every eval
                if validate:
                    torch.save(checkpoint, EVAL_CHECKPOINT_DIR / Path(f"batch_{b}.pt"))
                    validate = False
                epoch_bar.update(batches_for_checkpoint)


def add_dependencies(config: ModelConfig):
    checkpoint = torch.load('model/3_layers/checkpoint.pt', weights_only=True)
    model = GraphMatcher(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.add_dependency_embeddings()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, 'model/3_layers/checkpoint.pt')


if __name__ == "__main__":
    # add_dependencies(config)
    CONFIG = ModelConfig(
        n_layers=3,
        n_heads=2,
        d_upos=16,
        num_upos=len(UPOS_MAP),
        d_ner=16,
        num_ner=len(NER_MAP),
        d_relations=32,
        num_relations=len(REL_MAP) * 2,  # times two since reverse edges are included
        langs=["zh", "en"],
    )

    train(
        start_lr=1e-4,
        gradient_clipping=1,
        model_config=CONFIG,
        batch_size=16,
        max_batches=2000,
        # TODO - figure out where bottleneck is - might actually be in sinkhorn, which is why increasing batch size
        #  increases time per iter so much - num times sinkhorn algorithm is run is batch^2,
        #  so increasing batch size from b to 2b increases sinkhorn from b^2 to 4b^2
        accumulation_steps=2,  # with InfoNCE, not true accumulation - TODO alternative methods (e.g. MoCo style queue)
        batches_for_checkpoint=10,
        eval_interval_batches=500,
        validation_batches=50,
    )
