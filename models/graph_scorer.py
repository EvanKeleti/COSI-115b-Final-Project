from dataclasses import dataclass

import torch
from jaxtyping import Float
from networkx.classes import nodes
from tensordict import TensorDict
from torch import Tensor, nn

import torch.nn.functional as F


@dataclass
class ScorerConfig:
    init_sinkhorn_reg: float
    sinkhorn_reg_m: float = 1
    sinkhorn_iters: int = 20
    num_features = 7
    mlp_d1: int = 64
    mlp_d2: int = 32


class UnbalancedSinkhorn(nn.Module):

    def __init__(self, config: ScorerConfig):
        super().__init__()

        self.log_reg = nn.Parameter(torch.log(torch.tensor(config.init_sinkhorn_reg)))

        self.reg_m = config.sinkhorn_reg_m
        self.max_iter = config.sinkhorn_iters


    def forward(
            self,
            C: Float[Tensor, "batch max_n max_m"],
            a: Float[Tensor, "batch max_n"],
            b: Float[Tensor, "batch max_m"],
    ):
        """
        Parallel batched unbalanced Sinkhorn supporting 3D tensors.
        Inputs:
          C: (Batch, Max_N, Max_M) Zero-padded Cost Matrix
          a: (Batch, Max_N) Marginal source weights (padded areas set to 0)
          b: (Batch, Max_M) Marginal target weights (padded areas set to 0)
        """
        with torch.amp.autocast(device_type="cuda", enabled=False):
            C = C.float()
            a = a.float()
            b = b.float()

            batch_size, max_n, max_m = C.shape
            # print(
            #     "COST: ",
            #     C.min().item(),
            #     C.max().item(),
            #     C.mean().item(),
            # )
            reg = self.log_reg.exp().clamp(1e-3, 1.0)

            # 1. Compute the Gibbs Kernel
            K = torch.exp(-C / reg)  # Shape: (Batch, Max_N, Max_M)

            # 2. Initialize dual scaling vectors
            u = torch.ones(batch_size, max_n, device=C.device, dtype=torch.float32)
            v = torch.ones(batch_size, max_m, device=C.device, dtype=torch.float32)

            # Exponent modifier for unbalanced KL-divergence penalty
            fi = self.reg_m / (self.reg_m + reg) if self.reg_m != float("inf") else 1

            # 3. Fixed-iteration loop (ensures uniform parallel GPU execution)
            for i in range(self.max_iter):
                # Update u: a / (K @ v)
                kv = torch.bmm(K, v.unsqueeze(-1)).squeeze(-1)
                # print(f"kv: min: {kv.min().item()}, max: {kv.max().item()}")
                # Avoid division by zero in padded zones using torch.clamp
                # torch.exp(fi * torch.log(torch.clamp(a / torch.clamp(kv, min=1e-12), min=1e-20)))
                u = torch.pow((a + 1e-20) / torch.clamp(kv, min=1e-12), fi)
                # print(f"u: min: {u.min().item()}, max: {u.max().item()}")

                # Update v: b / (K.T @ u)
                ktu = torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)
                # print(f"ktu: min: {ktu.min().item()}, max: {ktu.max().item()}")
                v = torch.pow((b + 1e-20) / torch.clamp(ktu, min=1e-12), fi)
                # print(f"v: min: {v.min().item()}, max: {v.max().item()}")

            # 4. Reconstruct the full transport plan matrices natively
            # P = diag(u) @ K @ diag(v) -> u[:, :, None] * K * v[:, None, :]
            plans = u.unsqueeze(-1) * K * v.unsqueeze(1)

        return plans  # Shape: (Batch, Max_N, Max_M)


class MLPScorer(nn.Module):

    def __init__(self, config: ScorerConfig):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.num_features, config.mlp_d1),
            nn.GELU(),
            nn.LayerNorm(config.mlp_d1),

            nn.Linear(config.mlp_d1, config.mlp_d2),
            nn.GELU(),
            nn.LayerNorm(config.mlp_d2),

            nn.Linear(config.mlp_d2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class GraphScorer(nn.Module):
    def __init__(self, config: ScorerConfig):
        super().__init__()

        self.num_features = config.num_features
        self.mlp_scorer = MLPScorer(config)
        self.sinkhorn = UnbalancedSinkhorn(config)

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

    def forward(self, graphs: dict[str, TensorDict]) -> Float[Tensor, "batch batch"]:
        # Get pairwise similarity scores
        scores = self.compute_similarity_scores(graphs)
        return scores

    def compute_similarity_scores(self, graphs: dict[str, TensorDict]) -> Float[Tensor, "batch batch"]:
        l1, l2 = graphs.keys()
        # Normalize so matmul gives cosine similarity
        nodes_1 = F.normalize(graphs[l1]['nodes'], dim=-1) # (B1, N1, H)
        nodes_2 = F.normalize(graphs[l2]['nodes'], dim=-1) # (B2, N2, H)
        # Get similarity matrices for each pair across batches
        # Flatten into B^2 batches of similarity matrices for each pair in batches
        S = torch.einsum('bnh,cmh->bcnm', nodes_1, nodes_2).flatten(0, 1) # (B1 * B2, N1, N2)
        # Get mask of invalid pairs due to padding, and repeat them so they match the flattened similarity matrix
        mask_1 = graphs[l1]['node_mask'].repeat_interleave(nodes_2.size(0), dim=0) # (B1 * B2, N1)
        mask_2 = graphs[l2]['node_mask'].repeat(nodes_1.size(0), 1) # (B1 * B2, N2)
        # Divide by lengths to get uniform weights for non-padded entries
        weights_1 = mask_1 / mask_1.sum(dim=-1)[:, None] # (B1 * B2, N1)
        weights_2 = mask_2 / mask_2.sum(dim=-1)[:, None]  # (B1 * B2, N2)
        # Transform to non-negative cost domain; [-1, 1] -> [0, 2] and scale to [0, 1] for optimal stability
        cost = (1.0 - S) / 2.0
        # Get TODO - I should understand more of the math behind this
        P = self.sinkhorn(cost, weights_1, weights_2)
        # Use similarity and transport plan matrices to get features
        weighted_sim = P * S
        matched_similarity = weighted_sim.sum(dim=(1, 2))
        matched_mass = P.sum(dim=(1, 2))
        average_similarity = matched_similarity / (matched_mass + 1e-8)
        # best_similarity_a = weighted_sim.max(dim=2)
        # best_similarity_a = (best_similarity_a * mask_a).sum(dim=1) / mask_a.sum(dim=1)
        # best_similarity_b = weighted_sim.max(dim=1)
        # best_similarity_b = (best_similarity_b * mask_b).sum(dim=1) / mask_b.sum(dim=1)
        P_flat = P.flatten(1)
        entropy = -(P_flat * torch.log(P_flat + 1e-8)).sum(dim=1)
        mass_1 = P.sum(dim=2)
        mass_2 = P.sum(dim=1)
        # a_mass_error = (mass_1 - a_padded).abs().sum(dim=1)
        # b_mass_error = (mass_2 - b_padded).abs().sum(dim=1)
        unmatched_a = ((weights_1 - mass_1).clamp(min=0)).sum(dim=1)
        unmatched_b = ((weights_2 - mass_2).clamp(min=0)).sum(dim=1)
        mean_transport = matched_mass / (mask_1.sum(dim=1) * mask_2.sum(dim=1))
        max_transport = P.amax(dim=(1, 2))
        mean_sim = S.sum(dim=(1, 2)) / (mask_1.sum(dim=1) * mask_2.sum(dim=1))

        features = torch.stack([
            matched_similarity,
            average_similarity,
            # best_similarity_a,
            # best_similarity_b,
            entropy,
            # a_mass_error,
            # b_mass_error,
            unmatched_a,
            unmatched_b,
            mean_transport,
            max_transport,
            # mean_sim,
        ], dim=1)
        assert features.size(1) == self.num_features
        if torch.isnan(features).any().item():
            print("FEATURES: ", features)

        scores = self.mlp_scorer(features)
        # CHANGE to just trying this
        # scores = matched_similarity
        scores = scores.unflatten(0, (nodes_1.size(0), nodes_2.size(0)))
        return scores

    def get_alignments(self, M, relations_row, relations_col): # todo temp
        row_best = torch.argmax(M, dim=-1)
        col_best = torch.argmax(M, dim=-2)
        row_choices = torch.zeros(M.shape)
        col_choices = torch.zeros(M.shape)
        row_idx = torch.arange(M.size(-1), device=M.device)
        row_choices[:, row_idx, row_best] = 1
        col_idx = torch.arange(M.size(-2), device=M.device)
        col_choices[:, col_idx, col_best] = 1
        row_col_agree = row_idx == col_idx.T
        merged_nodes = row_idx + col_idx.T
        # todo verify this math
        num_merged = (merged_nodes.sum(dim=-1) > 1).sum(dim=-1).item()
        row_to_merged_map = row_best
        col_to_merged_map = row_best[col_best]
        # Eliminate reverse edges and self loops that were just used for attention todo
        relations_row *= relations_row % 2
        relations_col *= relations_col % 2
        row_rel_idx = torch.argmax(relations_row, dim=-1)
        col_rel_idx = torch.argmax(relations_col, dim=-1)
        row_map = row_to_merged_map[row_rel_idx]
        col_map = col_to_merged_map[col_rel_idx]
        merged_idx = torch.arange(num_merged, device=M.device)
        mapped_row_relations = torch.zeroes(M.size(0), relations_row.size(1), num_merged, device=M.device)
        mapped_col_relations = torch.zeroes(M.size(0), relations_col.size(1), num_merged, device=M.device)
        mapped_row_relations[:, torch.arange(relations_row.size(1), device=M.device), row_map] = 1 # todo not 1?
        mapped_col_relations[:, torch.arange(relations_col.size(1), device=M.device), col_map] = 1
        merged = torch.zeros(M.size(0), num_merged, num_merged, device=M.device)
        row_idx_3d = row_map.unsqueeze(-1).expand(-1, -1, mapped_row_relations.size(-1))
        col_idx_3d = col_map.unsqueeze(-1).expand(-1, -1, mapped_row_relations.size(-1))
        merged_relations_row = merged.scatter_reduce(dim=1, index=row_idx_3d, src=mapped_row_relations, reduce='sum')
        merged_relations_col = merged.scatter_reduce(dim=1, index=col_idx_3d, src= mapped_col_relations, reduce='sum')
        merged_cols = merged.scatter_reduce(dim=1, index=col_)
        # row_relations_merged[:, merged_idx, row_labels] = 1 # todo fix this
        # col_relations_merged[:, merged_idx, col_labels] = 1
        # todo check that there are relations between nodes that are merged? - aka should have a self loop with 1 for each merged node
        # Check that merged nodes share relations?


        row_agree_idx = row_col_agree.sum(dim=-1)
        # Now we want to check that tokens with 1 to 1 alignments share relations? maybe
        # distinguish between direction of relation?
        row_rels = relations_row[:, row_agree_idx]
        col_agree_idx = row_col_agree.sum(dim=-2)
        col_rels = relations_col[:, col_agree_idx]



