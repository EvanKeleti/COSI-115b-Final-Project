import itertools
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import torch
from accelerate import Accelerator
from dacite import from_dict
from dotenv import load_dotenv
from jaxtyping import Float
from tensordict import TensorDict
from torch import Tensor
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

from data_processing.data import load_wmt_data
from data_processing.graph_builder import UPOS_MAP, NER_MAP, REL_MAP
from data_processing.preprocess import GraphCollator, Preprocessor, SKIP
from models.graph_encoder import RGATConfig, RGAT
from models.graph_scorer import GraphScorer, ScorerConfig
from models.loss import MoCoConfig, MoCoInfoNCELoss
from utils import plot_results


@dataclass
class TrainArgs:
    peak_lr: float
    # final_lr: float
    total_steps: int
    batch_size: int
    rgat_config: RGATConfig
    scorer_config: ScorerConfig
    moco_config: MoCoConfig
    restart_moco: bool = False
    gradient_accumulation_steps: int = 1
    gradient_clipping: float = 1
    batches_for_checkpoint: int = 10  # todo or steps?

    def __str__(self):
        return (
            f"{str(self.rgat_config)}-{str(self.moco_config)}-{self.total_steps}_steps-{self.batch_size}_batch"
            + (f"-acc_{self.gradient_accumulation_steps}" if self.gradient_accumulation_steps > 1 else "")
        )
    # todo treat train_args like dictionary key to find right checkpoint, instead of a ridiculously long file name


class Pipeline:

    def __init__(
            self,
            checkpoint_dir: str | Path,
            train_args: TrainArgs,
            verbose: bool = True,
    ):
        self.train_data = None
        self.preprocessor = None

        self.checkpoint_file = Path(checkpoint_dir) / "checkpoint.pt"
        print(f"Looking for checkpoint at {self.checkpoint_file}.")

        if Path(self.checkpoint_file).exists():
            self.state = torch.load(self.checkpoint_file, weights_only=True)
            self.train_args = from_dict(data_class=TrainArgs, data=self.state['train_args'])
            self.train_args.restart_moco = train_args.restart_moco
            if self.train_args.restart_moco:
                self.train_args.moco_config = train_args.moco_config
            if verbose:
                tqdm.write(f"Checkpoint found at batch {self.state['next_datapoint'] // self.train_args.batch_size}.")
            first = False
        else:
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.state = {
                "next_datapoint": 0,
                "losses": [],
                "grad_norms": [],
                "accuracies": [],
                "train_args": asdict(train_args),
            }
            first = True
            self.train_args = train_args
            if verbose:
                tqdm.write("Checkpoint not found.")

        self.accelerator = Accelerator(
            mixed_precision="fp16",
            gradient_accumulation_steps=self.train_args.gradient_accumulation_steps,
        )
        # todo how to use it?
        # self.accelerator.register_for_checkpointing(self.state)

        self._init_rgat()
        self._init_scorer()
        self._init_moco()

        self._init_optimizer()
        self._init_scheduler()

        (
            self.rgat,
            self.scorer,
            self.moco_loss,
            self.optimizer,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.rgat, self.scorer,
            self.moco_loss,
            self.optimizer, self.scheduler
        )

        if not first:
            self.accelerator.load_state(str(self.checkpoint_file.parent / "accelerate"))

        self.loss = 0

    def _init_rgat(self):
        self.rgat = RGAT(self.train_args.rgat_config)
        if sd := self.state.get("rgat_state_dict", False):
            self.rgat.load_state_dict(sd)

    def _init_scorer(self):
        self.scorer = GraphScorer(self.train_args.scorer_config)
        if sd := self.state.get("scorer_state_dict", False):
            self.scorer.load_state_dict(sd, strict=False)

    def _init_moco(self):
        if self.train_args.moco_config.use_momentum_encoder:
            if self.rgat is None:
                raise ValueError("RGAT is not initialized.")
            self.train_args.moco_config.online_encoder = self.rgat
        self.moco_loss = MoCoInfoNCELoss(self.train_args.moco_config)
        if (sd := self.state.get("moco_state_dict", False)) and not self.train_args.restart_moco:
            self.moco_loss.load_state_dict(sd)

    def _init_optimizer(self):
        if self.rgat is None:
            raise ValueError("RGAT is not initialized.")
        self.optimizer = torch.optim.AdamW([
            {"params": self.rgat.parameters(), "lr": self.train_args.peak_lr},
            {"params": [p for name, p in self.scorer.named_parameters() if name != "sinkhorn.log_reg"],
             "lr": self.train_args.peak_lr},
            {"params": [self.scorer.sinkhorn.log_reg], "lr": 1e-5}
        ])
        if sd := self.state.get("optimizer_state_dict", False):
            self.optimizer.load_state_dict(sd)

    def _init_scheduler(self):
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized.")
        warmup_steps = self.train_args.total_steps // 10  # 10% warmup
        decay_steps = self.train_args.total_steps - warmup_steps
        # Warmup Scheduler: Linearly increase LR from 10% of max to max
        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        # Decay Scheduler: Cosine decay over remaining steps down to a minimum fraction
        decay_scheduler = CosineAnnealingLR(self.optimizer, T_max=decay_steps, eta_min=self.train_args.peak_lr / 100)
        # Combine them sequentially
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps]
        )
        if sd := self.state.get("scheduler_state_dict", False):
            self.scheduler.load_state_dict(sd)

    def save_checkpoint(self):
        # self.state["rgat_state_dict"] = self.rgat.state_dict()
        # self.state["scorer_state_dict"] = self.scorer.state_dict()
        # self.state["moco_state_dict"] = self.moco_loss.state_dict()
        # self.state["optimizer_state_dict"] = self.optimizer.state_dict()
        # self.state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(self.state, self.checkpoint_file)  # todo don't double save with torch and accelerate
        self.accelerator.save_state(output_dir=str(self.checkpoint_file.parent / "accelerate"))

    # todo do in moco - accuracy against many negatives
    def get_accuracy_on_batch(self, scores: Float[Tensor, "batch batch"]) -> float:
        # Correct predictions are on diagonal of score matrix
        correct = torch.arange(scores.size(0), device=scores.device)
        # Get row accuracy
        predictions_row = torch.argmax(scores, dim=1)
        correct_row = (predictions_row == correct).sum().item()
        acc_row = correct_row / scores.size(0)
        # Get column accuracy
        predictions_col = torch.argmax(scores.T, dim=1)
        correct_col = (predictions_col == correct).sum().item()
        acc_col = correct_col / scores.size(0)
        # Average
        symmetric_accuracy = (acc_row + acc_col) / 2
        return symmetric_accuracy

    @torch.no_grad()
    def evaluate(self, batch_size: int): # todo use queue when evaluating?
        self.rgat.eval()
        self.scorer.eval()
        validation_data = load_wmt_data('validation')
        num_val = len(validation_data)
        del validation_data
        graph_collator = GraphCollator(
            langs=['zh', 'en'],
            batch_size=self.train_args.batch_size,
            split='validation',
            selection_range=(0, num_val),
        )
        preprocessor = Preprocessor(
            graph_collator,
            preprocess_batch_size=self.train_args.batch_size,
            out_batch_size=batch_size
        )
        try:
            accuracy_sum = 0.0

            pbar = tqdm(
                enumerate(preprocessor, 1),
                total=num_val,
                smoothing=0.1,
                desc=f"Evaluating on {num_val} batches",
                leave=True,
            )

            for b, batch in pbar:
                inputs = {lang: graphs.to(self.accelerator.device) for lang, graphs in batch.items()}

                outputs = {lang: self.rgat(graphs) for lang, graphs in inputs.items()}

                scores = self.scorer(outputs)
                accuracy_sum += self.get_accuracy_on_batch(scores)

                diag = scores.diag()

                mask = ~torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)
                off = scores[mask]

                tqdm.write(
                    f"""
                    ==== Batch {b} ====
                    diag mean: {diag.mean().item():.4f}, diag min: {diag.min().item():.4f}
                    off mean : {off.mean().item():.4f}, off max : {off.max().item():.4f}
                    """
                )

                accuracy = accuracy_sum / b
                pbar.set_postfix_str(f"Accuracy: {accuracy * 100:.2f}")

        finally:
            preprocessor.stop_processing()

    def train_step(self, batch: dict[str, TensorDict], verbose: bool = False):
        with self.accelerator.accumulate(self.rgat, self.scorer):
            inputs = {lang: graphs.to(self.accelerator.device) for lang, graphs in batch.items()}

            outputs = {lang: self.rgat(graphs) for lang, graphs in inputs.items()}
            for lang, output in outputs.items():
                if torch.isnan(output["nodes"]).any().item():
                    tqdm.write(f"{lang} encoder output contains NaN.")

            loss, accuracy = self.moco_loss(inputs, outputs, self.scorer, self.rgat)

            self.accelerator.backward(loss)

            self.loss += loss.detach().item()

            if self.accelerator.sync_gradients:
                combined_parameters = itertools.chain(self.rgat.parameters(), self.scorer.parameters())
                grad_norm = self.accelerator.clip_grad_norm_(
                    combined_parameters,
                    max_norm=self.train_args.gradient_clipping
                ).item()
                if verbose:
                    tqdm.write(f"Loss: {self.loss:.3f}, Accuracy: {accuracy:.2f}, Grad Norm: {grad_norm:.1f}")
                self.state['grad_norms'].append(grad_norm)
                if not self.state.get('accuracies', False): # todo remove later
                    self.state['accuracies'] = []
                self.state['accuracies'].append(accuracy)
                self.state['losses'].append(self.loss)
                self.loss = 0

            # p = next(self.rgat.parameters())
            # before = p.detach().clone()
            # p2 = next(self.scorer.parameters())
            # before2 = p2.detach().clone()

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # print("rgat: ", (before - p).abs().max(), "scorer", (before2 - p2).abs().max())


    def _training_loop(
            self,
            train_data: Iterable[dict[str, TensorDict]],
            batches_for_checkpoint: int,
            verbose: bool,
    ):  # todo num_steps arg?
        self.rgat.train()
        self.scorer.train()
        # self.scorer.sinkhorn.log_reg.requires_grad = False

        finished_batches = self.state['next_datapoint'] // self.train_args.batch_size - self.state.get('skipped_batches', 0)

        pbar = tqdm(  # todo
            enumerate(train_data, finished_batches + 1),
            total=self.train_args.total_steps,
            initial=finished_batches,
            desc=f"Training on {self.train_args.total_steps} batches",
            smoothing=0.01,
            position=0,
        )

        for b, batch in pbar:

            if b == self.train_args.total_steps:
                break

            self.state['next_datapoint'] += next(iter(batch.values())).size(0)
            if batch == SKIP:
                tqdm.write(f"Batch {b} skipped")
                continue

            try:
                self.train_step(batch, verbose)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    lens = {lang: graphs["node_mask"].size(1) for lang, graphs in batch.items()}
                    error_str = (
                        f"[OOM Warning] Skipped batch {b}. Max nodes: "
                        ','.join([f"{lang}: {num}" for lang, num in lens.items()])
                    )
                    tqdm.write(error_str)
                    if not self.state.get('oom_log', False):
                        self.state["oom_log"] = []
                    self.state["oom_log"].append((b, lens))
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()
                else:
                    raise e
                # for now don't keep training, since don't know when / how often will oom
                raise e

            if b % batches_for_checkpoint == 0 or b == self.train_args.total_steps:
                # Save checkpoint
                self.save_checkpoint()
                # tqdm.write(
                #     f"Batch {b}: "
                #     f"Allocated: {torch.cuda.memory_allocated() // 1024**2}, "
                #     f"Reserved: {torch.cuda.memory_reserved() // 1024**2}"
                # )
                # Plot losses and grad norms
                plot_results(self.state, save_path=str(self.checkpoint_file.parent / "training_progress.png"))

    def train(
            self,
            batches_for_checkpoint: int,
            preprocess_batch_size: int = None,
            verbose: bool = False,
            debug_nan: bool = False,
    ):
        if preprocess_batch_size is None:
            preprocess_batch_size = self.train_args.batch_size
        elif preprocess_batch_size % self.train_args.batch_size != 0:
            raise ValueError("Preprocess batch size must be divisible by training batch size")
        finished_batches = self.state['next_datapoint'] // self.train_args.batch_size
        if finished_batches == self.train_args.total_steps:
            print(f"Training already finished at step {finished_batches}")
            # todo - way to add another stage to training (keep checkpoints at end of each stage)
            return
        graph_collator = GraphCollator(
            langs=['zh', 'en'],
            batch_size=preprocess_batch_size, # todo `preprocess_batch_size`
            split='train',
            selection_range=(self.state['next_datapoint'], self.train_args.total_steps * self.train_args.batch_size),
        )
        preprocessor = Preprocessor(
            graph_collator,
            preprocess_batch_size=preprocess_batch_size,
            out_batch_size=self.train_args.batch_size
        )
        try:
            if debug_nan:
                torch.autograd.set_detect_anomaly(True)
            self._training_loop(preprocessor, batches_for_checkpoint, verbose)
        finally:
            preprocessor.stop_processing()


def main():
    load_dotenv()
    CHECKPOINT_BASE_DIR = Path("checkpoints/")

    rgat_config = RGATConfig(
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
    scorer_config = ScorerConfig(
        init_sinkhorn_reg=0.2
    )
    moco_config = MoCoConfig(
        init_tau=0.07,
        use_momentum_encoder=False,
        langs=["zh", "en"],
        K=0,
        momentum=0.98,
    )
    # train_data = load_wmt_data('train')
    train_args = TrainArgs(
        peak_lr=1e-4,
        # final_lr=1e-6, # todo
        # total_steps=len(train_data) // 8,
        total_steps=50000,
        batch_size=16,
        rgat_config=rgat_config,
        scorer_config=scorer_config,
        moco_config=moco_config,
        gradient_accumulation_steps=1,
        gradient_clipping=1,
    )
    # del train_data
    pipeline = Pipeline(CHECKPOINT_BASE_DIR / str(train_args), train_args)

    pipeline.train(batches_for_checkpoint=50, preprocess_batch_size=train_args.batch_size, verbose=False)
    # pipeline.evaluate(8)


if __name__ == '__main__':
    main()
