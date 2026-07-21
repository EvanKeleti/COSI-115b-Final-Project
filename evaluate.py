from pathlib import Path

import datasets
import torch
from accelerate import Accelerator
from datasets import Dataset
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

from generate_data import generate_hard_negatives
from graph_builder import UPOS_MAP, NER_MAP, REL_MAP
from model import GraphMatcher, ModelConfig


@torch.no_grad()
def evaluate_model(model, val_loader, num_batches, tqdm_pos: int = 0):
    model.eval()

    total_loss = 0.0
    accuracy_sum = 0.0

    avg_val_loss = accuracy = 0.0
    for b, batch in tqdm(enumerate(val_loader, 1), total=num_batches, position=tqdm_pos, leave=False,
                         desc=f"Evaluating model on {num_batches} batches"):
        if b == num_batches + 1:
            break

        scores = model(batch)
        total_loss += model.get_loss_on_batch(scores).item()
        accuracy_sum += model.get_accuracy_on_batch(scores)

        avg_val_loss = total_loss / b
        accuracy = accuracy_sum / b

        tqdm.write(f"\r{f'Average loss: {avg_val_loss:.4f}':<20} {f'Accuracy: {accuracy:.4f}':<20}")

    return avg_val_loss, accuracy


def eval_on_hard_negatives(config: ModelConfig, checkpoint: str | Path, num_batches: int = None):
    translation_variations: list[tuple[int, list[str]]] = generate_hard_negatives(1000, split='validation')
    if num_batches is None:
        num_batches = len(translation_variations)

    dataset = datasets.load_dataset("wmt/wmt19", "zh-en", streaming=False)['validation']
    data_dict = {'translation': []}
    for i, vars in translation_variations:
        pair = dataset[i]['translation']
        # Fourth pair in each batch of four is the correct one
        vars.append(pair['en'])
        data_dict['translation'].extend([{'zh': pair['zh'], 'en': var} for var in vars])

    hard_negs = Dataset.from_dict(data_dict)
    from preprocess import graph_collate_fn
    val_loader = DataLoader(hard_negs, batch_size=4, collate_fn=graph_collate_fn, shuffle=False)

    checkpoint = torch.load(checkpoint, weights_only=True)
    graph_matcher = GraphMatcher(config)
    graph_matcher.load_state_dict(checkpoint['model_state_dict'])
    graph_matcher.cuda()
    graph_matcher.eval()

    total_correct = 0
    # TODO make code more efficient - since chinese sentence is currently repeated
    # todo - log stats after each batch
    for b, batch in tqdm(enumerate(val_loader, 1), total=num_batches, position=0,
                         desc=f"Evaluating model on {num_batches} batches of hard negatives"):
        if b == num_batches + 1:
            break

        scores = graph_matcher(batch)
        best = torch.argmax(torch.diagonal(scores)).item()
        if best == 3:
            total_correct += 1

    accuracy = total_correct / num_batches
    print(f"Accuracy: {accuracy * 100:.3f}")


def eval_on_validation_set(config: ModelConfig, checkpoint: str | Path, num_batches: int = None):
    dataset = datasets.load_dataset("wmt/wmt19", "zh-en", split='validation')
    from preprocess import graph_collate_fn
    val_loader = DataLoader(dataset, batch_size=16, collate_fn=graph_collate_fn, shuffle=True)
    checkpoint = torch.load(checkpoint, weights_only=True)


    graph_matcher = GraphMatcher(config)
    graph_matcher.load_state_dict(checkpoint['model_state_dict'])
    graph_matcher.cuda()
    graph_matcher.eval()

    accelerator = Accelerator()
    graph_matcher, val_loader = accelerator.prepare(graph_matcher, val_loader)

    print(evaluate_model(graph_matcher, val_loader, num_batches))


if __name__ == '__main__':
    load_dotenv()
    config = ModelConfig(
        n_layers=2,
        n_heads=2,
        d_upos=16,
        num_upos=len(UPOS_MAP),
        d_ner=16,
        num_ner=len(NER_MAP),
        d_relations=32,
        num_relations=len(REL_MAP) * 2,  # times two since reverse edges are included
        langs=["zh", "en"],
    )

    # Accuracy: 29.630 - not much better than random chance
    # eval_on_hard_negatives(config, 'model/2_layers/eval_checkpoints/batch_4000.pt', num_batches=None)
    eval_on_validation_set(config, 'model/2_layers/eval_checkpoints/batch_4000.pt', num_batches=200)
