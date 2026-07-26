import math
from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float, Int
from tensordict import TensorDict
from torch import Tensor, nn, softmax


@dataclass
class RGATConfig:
    n_layers: int
    n_heads: int
    d_upos: int
    num_upos: int
    d_ner: int
    num_ner: int
    d_relations: int
    num_relations: int
    d_lang: int
    langs: list[str]  # TODO change to num langs
    # d_srl: int # TODO - next step is to add SRL
    # num_srl: int
    embedding_model: str = "xlm-roberta-base" # todo this shouldn't be here
    d_model: int = 768
    init_tau: float = 0.07

    def __str__(self):
        return f"{self.n_layers}_layer-{self.n_heads}_head"


class RelationalAttention(nn.Module):
    def __init__(self, config: RGATConfig):
        super().__init__()

        assert config.d_model % config.n_heads == 0
        self.d_attention = int(config.d_model / config.n_heads)
        self.num_heads = config.n_heads

        self.W_k = nn.Linear(config.d_model, config.d_model)
        self.W_q = nn.Linear(config.d_model, config.d_model)
        self.W_v = nn.Linear(config.d_model, config.d_model)

        self.W_o = nn.Linear(config.d_model, config.d_model)

        # Embeddings for relations between nodes. idx 0 = padding; idx 1 = self relation
        self.rel_key = nn.Embedding(config.num_relations + 2, config.d_model, padding_idx=0)  # for attention keys
        self.rel_value = nn.Embedding(config.num_relations + 2, config.d_model, padding_idx=0)  # for attention values

    def forward(
            self,
            nodes: Float[Tensor, "batch max_nodes d_model"],
            relations: Int[Tensor, "batch max_nodes max_nodes"],
            node_mask: Int[Tensor, "batch max_nodes"],
    ) -> Float[Tensor, "batch max_nodes d_model"]:
        # Get Q, K, V projections
        proj = [self.W_q(nodes), self.W_k(nodes), self.W_v(nodes)]
        # Reshape for multi-head attention (B, H, N, N, D)
        Q, K, V = map(lambda A: rearrange(A, "B n (h d_a) -> B h n d_a", d_a=self.d_attention), proj)
        # Compute attention scores
        scores = Q.matmul(rearrange(K, "B h n d_a -> B h d_a n")) / math.sqrt(self.d_attention)
        # Get relational embeddings and reshape for multi-head (B, H, N, N, D)
        R = self.rel_key(relations)
        R = rearrange(R, "B i j (h d_a) -> B h i j d_a", d_a=self.d_attention)
        # Calculate contribution from relation vectors
        rel_scores = torch.einsum("bhid,bhijd->bhij", Q, R) / math.sqrt(self.d_attention)
        scores += rel_scores
        # Mask scores by attending only to actual relations between nodes - no relation is represented by 0
        valid_attention = (relations != 0)
        idx = torch.arange(relations.size(1), device=relations.device)
        valid_attention[:, idx, idx] = True  # Have padding self-attend so softmax is well-defined
        masked_scores = scores.masked_fill(~valid_attention.unsqueeze(1), float('-inf'))
        # Get attention per head
        per_head_attn = softmax(masked_scores, dim=-1)
        # Zero out attention for padding nodes
        per_head_attn = per_head_attn * node_mask[:, None, :, None]
        # Get relation-aware values
        R_v = self.rel_value(relations)
        R_v = rearrange(R_v, "B i j (h d_a) -> B h i j d_a", d_a=self.d_attention)
        V_rel = V.unsqueeze(2) + R_v  # (B, H, N, N, D)
        # Get values weighted by attention
        out = torch.einsum("bhij,bhijd->bhid", per_head_attn, V_rel)
        # Merge heads
        out = rearrange(out, "B h T d_a -> B T (h d_a)")
        out = self.W_o(out)
        out = out * node_mask[:, :, None]
        return out


class MLP(nn.Module):

    def __init__(self, d: int):
        super().__init__()

        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.gelu = nn.GELU()

    def forward(
            self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:
        return self.fc2(self.gelu(self.fc1(x)))


class GraphEncoderBlock(nn.Module):

    def __init__(self, config: RGATConfig):
        super().__init__()

        self.mlp = MLP(config.d_model)
        self.attention = RelationalAttention(config)
        self.pre_layer_norm = nn.LayerNorm(config.d_model)
        self.post_layer_norm = nn.LayerNorm(config.d_model)

    def forward(
            self,
            nodes: Float[Tensor, "batch max_nodes d_node"],
            relations: Float[Tensor, "batch max_nodes max_nodes"],
            node_mask: Int[Tensor, "batch max_nodes"],
    ) -> Float[Tensor, "batch max_nodes d_model"]:
        h_norm = self.pre_layer_norm(nodes)
        h_mid = nodes + self.attention(h_norm, relations, node_mask)
        h_mid_norm = self.post_layer_norm(h_mid)
        h_out = h_mid + self.mlp(h_mid_norm)
        h_out = h_out * node_mask[:, :, None]
        return h_out


class RGAT(nn.Module):

    def __init__(self, config: RGATConfig):
        super().__init__()

        self.config = config

        self.root_node = nn.Parameter(torch.zeros(config.d_model))
        self.upos_embeddings = nn.Embedding(config.num_upos + 1, config.d_upos, padding_idx=0)
        self.ner_embeddings = nn.Embedding(config.num_ner + 1, config.d_ner, padding_idx=0)
        self.language_embeddings = nn.Embedding(len(config.langs), config.d_lang)

        self.d_features = config.d_upos + config.d_ner + config.d_lang
        self.struct_proj = nn.Linear(self.d_features, config.d_model)
        self.struct_ln = nn.LayerNorm(config.d_model)

        self.backbone = nn.ModuleList([GraphEncoderBlock(config) for _ in range(config.n_layers)])
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.device = None

        self._init_weights()

    # TODO verify want to init this way
    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)

    def forward(self, batch: TensorDict) -> TensorDict | dict: # todo
        # TODO move send to device out of model code
        # Send all input tensors to device
        # self.device = next(self.parameters()).device
        # for key, tensor in batch.items():
        #     if self.device.type != tensor.device.type:
        #         batch[key] = tensor.to(self.device)

        # Get embeddings for features and concatenate them
        struct = torch.cat([
            self.upos_embeddings(batch['upos']),
            self.ner_embeddings(batch['ner']),
            self.language_embeddings(batch['lang'].unsqueeze(-1).expand(-1, batch['upos'].size(1)))
        ], dim=-1)
        # Project into d_model
        struct = self.struct_ln(self.struct_proj(struct))
        x = batch['xlmr'] + struct
        # todo - norm?
        # Concatenate root node to each row of node embeddings
        expanded_root = self.root_node[None, None, :].expand(x.size(0), -1, -1)
        x = torch.cat([expanded_root, x], dim=1)
        # Encode
        for encoder in self.backbone:
            x = encoder(x, relations=batch['relations'], node_mask=batch['node_mask'])
        x = self.final_layer_norm(x) * batch['node_mask'][:, :, None]
        # return TensorDict({
        #     "nodes": x,
        #     "node_mask": batch['node_mask'],
        # }, batch_size=x.size(0))
        return {
            "nodes": x,
            "node_mask": batch['node_mask'],
        }

    def freeze_layers(self, layer_idx: list[int]) -> None:
        layers = list(self.backbone)
        for i in layer_idx:
            for param in layers[i].parameters():
                param.requires_grad = False

    def freeze_embeddings(self) -> None:
        for param in self.parameters(recurse=False):
            param.requires_grad = False
