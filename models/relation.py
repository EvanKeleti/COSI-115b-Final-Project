import math

import torch
from torch import nn, Tensor, softmax
from jaxtyping import Float, Int
from einops import rearrange

from models.graph_encoder import RGATConfig


class WeightedRelationalAttention(nn.Module):
    def __init__(self, config: RGATConfig):
        super().__init__()

        assert config.d_model % config.n_heads == 0
        self.d_attention = int(config.d_model / config.n_heads)
        self.num_heads = config.n_heads

        self.W_k = nn.Linear(config.d_model, config.d_model)
        self.W_q = nn.Linear(config.d_model, config.d_model)
        self.W_v = nn.Linear(config.d_model, config.d_model)

        self.W_o = nn.Linear(config.d_model, config.d_model)

        # Embeddings for relations between nodes.
        self.rel_key = nn.Parameter(torch.zeros(config.num_relations, config.d_model))  # for attention keys
        self.rel_value = nn.Parameter(torch.zeros(config.num_relations, config.d_model))  # for attention values

    def forward(
            self,
            nodes: Float[Tensor, "batch max_nodes d_model"],
            relation_weights: Float[Tensor, "batch max_nodes max_nodes num_rel"],
            node_mask: Int[Tensor, "batch max_nodes"],
    ) -> Float[Tensor, "batch max_nodes d_model"]:
        # Get Q, K, V projections
        proj = [self.W_q(nodes), self.W_k(nodes), self.W_v(nodes)]
        # Reshape for multi-head attention (B, H, N, N, D)
        Q, K, V = map(lambda A: rearrange(A, "B n (h d_a) -> B h n d_a", d_a=self.d_attention), proj)
        # Compute attention scores
        scores = Q.matmul(rearrange(K, "B h n d_a -> B h d_a n")) / math.sqrt(self.d_attention)
        # Get relational embeddings and reshape for multi-head (B, H, N, N, D)
        # todo - only look at triangular matrix? since otherwise would learn more than one relation per pair?
        # Get weighted average of relation vectors for each node pair
        R = torch.einsum("bijr,rd->bijd", relation_weights, self.rel_key)
        R = rearrange(R, "B i j (h d_a) -> B h i j d_a", d_a=self.d_attention)
        # Calculate contribution from relation vectors
        rel_scores = torch.einsum("bhid,bhijd->bhij", Q, R) / math.sqrt(self.d_attention)
        scores += rel_scores
        # Mask scores by attending only to actual relations between nodes - no relation is represented by 0
        valid_attention = torch.einsum("bi,bj->bij", node_mask, node_mask)
        idx = torch.arange(node_mask.size(1), device=node_mask.device)
        valid_attention[:, idx, idx] = True  # Have padding self-attend so softmax is well-defined
        masked_scores = scores.masked_fill(~valid_attention.unsqueeze(1), float('-inf'))
        # Get attention per head
        per_head_attn = torch.softmax(scores, dim=-1)
        # Zero out attention for padding nodes
        per_head_attn = per_head_attn * node_mask[:, None, :, None]
        # Get relation-aware values
        R_v = torch.einsum("bijr,rd->bijd", relation_weights, self.rel_value)
        R_v = rearrange(R_v, "B i j (h d_a) -> B h i j d_a", d_a=self.d_attention)
        V_rel = V.unsqueeze(2) + R_v  # (B, H, N, N, D)
        # Get values weighted by attention
        out = torch.einsum("bhij,bhijd->bhid", per_head_attn, V_rel)
        # Merge heads
        out = rearrange(out, "B h T d_a -> B T (h d_a)")
        out = self.W_o(out)
        out = out * node_mask[:, :, None]
        return out


class RelationPredictor(nn.Module):

    def __init__(self, config: RGATConfig):
        super().__init__()

        self.mid_dim = 128

        self.R = nn.Parameter(torch.zeros(self.mid_dim, self.mid_dim, config.num_relations)) # (M, M, R)
        self.R_q = nn.Linear(config.d_model, self.mid_dim) # (D, M)
        self.R_k = nn.Linear(config.d_model, self.mid_dim) # (D, M)


    def forward(
            self,
            nodes: Float[Tensor, "batch max_nodes d_model"],
    ) -> Float[Tensor, "batch max_nodes max_nodes num_rel"]:
        Q = self.R_q(nodes)  # (B, N, M)
        K = self.R_k(nodes)  # (B, N, M)

        # b = batch, i = query node (row), j = key node (col)
        # x = Q mid_dim axis, y = K mid_dim axis, r = relation axis
        R = torch.einsum('bix,bjy,xyr->bijr', Q, K, self.R)

        return R  # (B, N, N, R)
