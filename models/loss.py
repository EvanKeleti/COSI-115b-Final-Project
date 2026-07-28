import itertools
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn, Tensor
from tqdm import tqdm

from .graph_encoder import RGAT
from .graph_scorer import GraphScorer


# gpt function
def pad_nodes(x, target_nodes, value=0.0):
    """
    Pad tensor along node dimension.

    x: (..., N, ...)
       Usually (B, N, d) or (B, N)

    target_nodes: desired node dimension size
    """

    n = x.size(1)

    if n == target_nodes:
        return x

    pad = target_nodes - n

    if x.ndim == 3:
        # (B, N, d)
        return F.pad(x, (0, 0, 0, pad), value=value)

    elif x.ndim == 2:
        # (B, N)
        return F.pad(x, (0, pad), value=value)

    else:
        raise ValueError("Unsupported tensor shape")


# class InfoNCELoss(nn.Module):
#
#     def __init__(self, init_tau: float):
#         super().__init__()
#
#         # Smaller temperature -> Stronger discrimination between positive and negatives
#         self.temperature: float = init_tau
#         # self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / config.init_tau)))
#         self.register_buffer("temperature", torch.tensor(init_tau))
#
# def forward(
#         self,
#         scores: Float[Tensor, "batch1 batch2"],
#         labels: Float[Tensor, "batch1"],
# ) -> Float[Tensor, ""]:
#     # InfoNCE loss
#     # with torch.no_grad():
#     #     self.logit_scale.clamp_(0, torch.log(torch.tensor(100.0)).item())
#     # scores = scores * self.logit_scale.exp()
#     scores = scores / self.temperature
#
#     labels = torch.arange(scores.size(0), device=scores.device)
#     loss_1 = F.cross_entropy(scores, labels)
#     loss_2 = F.cross_entropy(scores.T, labels)
#     loss = (loss_1 + loss_2) / 2
#     return loss


@dataclass
class MoCoConfig:
    use_momentum_encoder: bool
    langs: list[str]
    K: int
    momentum: float
    init_tau: float
    max_nodes: int = 64
    d_node: int = 768
    online_encoder: Optional[RGAT] = None

    def __str__(self):
        return f"{self.K}_queue"


class MoCoQueue(nn.Module):

    def __init__(self, config: MoCoConfig):
        super().__init__()

        self.max_nodes = config.max_nodes

        random_queue = torch.randn(config.K, config.max_nodes, config.d_node)
        normalized_queue = F.normalize(random_queue, p=2, dim=2)
        self.register_buffer(
            "queue_nodes",
            normalized_queue,
        )

        self.register_buffer(
            "queue_mask",
            torch.ones(config.K, config.max_nodes)
        )

        self.register_buffer(
            "queue_ptr",
            torch.zeros(1, dtype=torch.long)
        )

    @torch.no_grad()
    def forward(self, nodes: Tensor, node_mask: Tensor):
        """
        Enqueue and dequeue if max_nodes of nodes will fit in queue, otherwise do nothing
        """
        if self.queue_nodes.size(0) == 0:
            return

        B = nodes.size(0)
        ptr = int(self.queue_ptr)
        end = ptr + B

        nodes, node_mask = nodes.detach(), node_mask.detach()
        nodes, node_mask = pad_nodes(nodes, self.max_nodes), pad_nodes(node_mask, self.max_nodes)

        # Only add to queue if max_nodes of batch is less than or equal to that of the queue
        if nodes.size(1) <= self.max_nodes:
            if end <= self.queue_nodes.size(0):
                self.queue_nodes[ptr:end] = nodes
                self.queue_mask[ptr:end] = node_mask
            else:
                first = self.queue_nodes.size(0) - ptr
                self.queue_nodes[ptr:] = nodes[:first]
                self.queue_nodes[:B - first] = nodes[first:]
                self.queue_mask[ptr:] = node_mask[:first]
                self.queue_mask[:B - first] = node_mask[first:]

            self.queue_ptr[0] = end % self.queue_nodes.size(0)


class MomentumEncoder(nn.Module):

    def __init__(self, config: MoCoConfig):
        super().__init__()

        self.momentum_encoder = RGAT(config.online_encoder.config)
        self.momentum_encoder.load_state_dict(config.online_encoder.state_dict())  # todo way to save momentum encoder
        for param_q, param_k in zip(self.momentum_encoder.parameters(), config.online_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # Initialize k with q
            param_k.requires_grad = False

        self.register_buffer("momentum", torch.tensor(config.momentum))

    @torch.no_grad()
    def forward(self, batch: TensorDict) -> TensorDict:
        return self.momentum_encoder(batch)

    @torch.no_grad()
    def momentum_update(self, online_encoder: RGAT):
        for p_q, p_k in zip(online_encoder.parameters(), self.momentum_encoder.parameters()):
            p_k.data.mul_(self.momentum).add_(p_q.data, alpha=1 - self.momentum)


class MoCoInfoNCELoss(nn.Module):

    def __init__(self, config: MoCoConfig):
        super().__init__()
        assert len(config.langs) == 2

        self.K = config.K
        self.max_nodes = config.max_nodes

        self.register_buffer("temperature", torch.tensor(config.init_tau))
        if config.K != 0:
            self.queues = nn.ModuleDict({lang: MoCoQueue(config) for lang in config.langs})
            self.momentum_encoder = MomentumEncoder(config) if config.use_momentum_encoder else None

    def forward(
            self,
            inputs: dict[str, TensorDict],
            outputs: dict[str, TensorDict],
            scorer: GraphScorer,
            online_encoder: RGAT = None
    ):
        max_nodes = max(output["nodes"].size(1) for output in outputs.values())
        if self.K == 0 or max_nodes > 2 * self.max_nodes: # todo
            if self.K != 0:
                tqdm.write(f"max_nodes is {max_nodes}, not using queue")
            scores = scorer(outputs)
            scores = scores / self.temperature

            labels = torch.arange(scores.size(0), device=scores.device)
            loss_1 = F.cross_entropy(scores, labels)
            loss_2 = F.cross_entropy(scores.T, labels)
            loss = (loss_1 + loss_2) / 2
            with torch.no_grad():
                accuracy_1 = (torch.argmax(scores, dim=-1) == labels).sum() / scores.size(0)
                accuracy_2 = (torch.argmax(scores.T, dim=-1) == labels).sum() / scores.size(0)
                accuracy = (accuracy_1 + accuracy_2) / 2
            return loss, accuracy.item() * 100 # todo this loss and accuracy are misleading, since the queue isn't being used

        losses = []
        accuracies = []
        for l_query, l_key in itertools.permutations(outputs.keys()):
            # Concatenate negative language samples from current batch and from queue
            queue = self.queues[l_key]
            max_nodes = max(outputs[l_key]["nodes"].size(1), queue.queue_nodes.size(1))
            keys = pad_nodes(outputs[l_key]["nodes"], max_nodes)
            key_mask = pad_nodes(outputs[l_key]["node_mask"], max_nodes)
            queue_keys, queue_key_mask = pad_nodes(queue.queue_nodes, max_nodes), pad_nodes(queue.queue_mask, max_nodes)
            keys = torch.cat([keys, queue_keys], dim=0)
            key_mask = torch.cat([key_mask, queue_key_mask], dim=0)
            # Score query language against other lang of same batch and queue
            scores = scorer({
                l_query: outputs[l_query],
                l_key: TensorDict({"nodes": keys, "node_mask": key_mask}, batch_size=keys.size(0)),
            })
            # print("SCORES: ", scores)
            # if torch.isnan(outputs[l_query]["nodes"]).any().item():
            #     tqdm.write(f"Q contains NaN.")
            # if torch.isnan(keys).any().item():
            #     tqdm.write(f"K contains NaN.")
            # if torch.isnan(queue.queue_nodes).any().item():
            #     tqdm.write("Queue contains NaN.")

            labels = torch.arange(scores.size(0), device=scores.device)
            accuracy = (torch.argmax(scores, dim=-1) == labels).sum() / scores.size(0)
            accuracies.append(accuracy.detach().item())
            scores = scores / self.temperature
            # tqdm.write("CE: " + str(F.cross_entropy(scores, labels).item()))
            # tqdm.write("Mean positive probability: " + str(probs[torch.arange(scores.size(0)), labels].mean().item()))
            losses.append(F.cross_entropy(scores, labels))

        # Add encodings to queue, using momentum encoder if one was given
        for lang in inputs.keys():
            if self.momentum_encoder is not None:
                graphs = self.momentum_encoder(inputs[lang])
            else:
                graphs = outputs[lang]
            self.queues[lang](graphs["nodes"], graphs["node_mask"])
        if self.momentum_encoder is not None:
            if not online_encoder:
                raise ValueError("If momentum encoder is being used, online_encoder must not be None")
            self.momentum_encoder.momentum_update(online_encoder)

        return torch.stack(losses).mean(), torch.tensor(accuracies).mean().item() * 100
