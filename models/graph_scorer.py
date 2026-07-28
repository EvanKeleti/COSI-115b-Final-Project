from dataclasses import dataclass

import torch
from jaxtyping import Float
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
        nodes_1 = graphs[l1]['nodes'] # (B1, N1)
        nodes_2 = graphs[l2]['nodes'] # (B2, N2)
        batch_size_1 = nodes_1.size(0) # B1 # TODO remove these unnecessary variables that clutter function
        batch_size_2 = nodes_2.size(0) # B2
        # Normalize so matmul gives cosine similarity
        H_1 = F.normalize(nodes_1, dim=-1) # (B1, N1, H)
        H_2 = F.normalize(nodes_2, dim=-1) # (B2, N2, H)
        # Get similarity matrices for each pair across batches
        S = torch.einsum('bnh,cmh->bcnm', H_1, H_2) # (B1, B2, N1, N2)
        # Flatten into B^2 batches of similarity matrices for each pair in batches
        S = S.flatten(0, 1) # (B1 * B2, N1, N2)
        # assert sim.shape == (batch_size_1 * batch_size_2, H_1.size(1), H_2.size(1))
        # Get mask of invalid pairs due to padding
        mask_a = graphs[l1]['node_mask']  # (B1, N1)
        mask_b = graphs[l2]['node_mask']  # (B2, N2)
        # Repeat masks so they match the flattened similarity matrix
        mask_a = mask_a.repeat_interleave(batch_size_2, dim=0) # (B1 * B2, N1)
        mask_b = mask_b.repeat(batch_size_1, 1) # (B1 * B2, N2)
        # Divide by lengths to get uniform weights for non-padded entries
        a_padded = mask_a / mask_a.sum(dim=-1)[:, None] # (B1 * B2, N1)
        b_padded = mask_b / mask_b.sum(dim=-1)[:, None]  # (B1 * B2, N2)
        assert a_padded.shape == (batch_size_1 * batch_size_2, mask_a.size(1)), f"{a_padded.shape}"
        assert b_padded.shape == (batch_size_1 * batch_size_2, mask_b.size(1)), f"{b_padded.shape}"

        # Transform to non-negative cost domain; [-1, 1] -> [0, 2]
        cost = 1.0 - S
        # Scale cost matrix to [0, 1] for optimal stability
        cost = cost / 2.0
        # Get TODO - I should understand more of the math behind this
        P = self.sinkhorn(cost, a_padded, b_padded)
        # Use similarity and transport plan matrices to get features
        # Note: 'a' is rows and 'b' is columns
        weighted_sim = P * S
        # tqdm.write(str(torch.isfinite(weighted_sim).sum()))
        matched_similarity = weighted_sim.sum(dim=(1, 2))
        matched_mass = P.sum(dim=(1, 2))
        average_similarity = matched_similarity / (matched_mass + 1e-8)
        # best_similarity_a = weighted_sim.max(dim=2)
        # best_similarity_a = (best_similarity_a * mask_a).sum(dim=1) / mask_a.sum(dim=1)
        # best_similarity_b = weighted_sim.max(dim=1)
        # best_similarity_b = (best_similarity_b * mask_b).sum(dim=1) / mask_b.sum(dim=1)
        P_flat = P.flatten(1)
        entropy = -(P_flat * torch.log(P_flat + 1e-8)).sum(dim=1)
        a_mass = P.sum(dim=2)
        b_mass = P.sum(dim=1)
        # a_mass_error = (a_mass - a_padded).abs().sum(dim=1)
        # b_mass_error = (b_mass - b_padded).abs().sum(dim=1)
        unmatched_a = ((a_padded - a_mass).clamp(min=0)).sum(dim=1)
        unmatched_b = ((b_padded - b_mass).clamp(min=0)).sum(dim=1)
        mean_transport = matched_mass / (mask_a.sum(dim=1) * mask_b.sum(dim=1))
        max_transport = P.amax(dim=(1, 2))
        mean_sim = S.sum(dim=(1, 2)) / (mask_a.sum(dim=1) * mask_b.sum(dim=1))

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
        scores = scores.unflatten(0, (batch_size_1, batch_size_2))
        return scores
