import math
import ot

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor, nn, softmax


@dataclass
class ModelConfig:
    n_heads: int
    n_layers: int
    d_upos: int
    num_upos: int
    d_ner: int
    num_ner: int
    d_relations: int
    num_relations: int
    # d_srl: int # TODO - next step is to add SRL
    # num_srl: int
    langs: list[str]
    d_model: int = None  # Actual value will be calculated in RGAT.__init__()
    d_word: int = 768
    init_tau: float = 0.07


class RelationalAttention(nn.Module):
    def __init__(self, config: ModelConfig):
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
        node_mask = torch.diagonal(relations, dim1=-2, dim2=-1) != 0
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
        return out


class GELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))  # fmt: skip


class MLP(nn.Module):

    def __init__(self, d: int):
        super().__init__()

        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.gelu = GELU()

    def forward(
            self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:
        return self.fc2(self.gelu(self.fc1(x)))


class GraphEncoderBlock(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.mlp = MLP(config.d_model)
        self.attention = RelationalAttention(config)
        self.pre_layer_norm = nn.LayerNorm(config.d_model)
        self.post_layer_norm = nn.LayerNorm(config.d_model)

    def forward(
            self,
            nodes: Float[Tensor, "batch max_nodes d_node"],
            relations: Float[Tensor, "batch max_nodes max_nodes"],
    ) -> Float[Tensor, "batch max_nodes d_model"]:
        h_norm = self.pre_layer_norm(nodes)
        h_mid = nodes + self.attention(h_norm, relations)
        h_mid_norm = self.post_layer_norm(h_mid)
        h_out = h_mid + self.mlp(h_mid_norm)
        return h_out


class RGAT(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        config.d_model = config.d_word + config.d_upos + config.d_ner

        self.root_node = nn.Parameter(torch.zeros(config.d_model))
        self.upos_embeddings = nn.Embedding(config.num_upos + 1, config.d_upos, padding_idx=0)
        self.ner_embeddings = nn.Embedding(config.num_ner + 1, config.d_ner, padding_idx=0)

        self.backbone = nn.ModuleList([GraphEncoderBlock(config) for _ in range(config.n_layers)])
        self.final_layer_norm = nn.LayerNorm(config.d_model)  # todo should i have this?
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

    def forward(self, batch: dict[str, Tensor]) -> Float[Tensor, "batch max_nodes d_model"]:
        # Send all input tensors to device
        self.device = next(self.parameters()).device
        for key, tensor in batch.items():
            if self.device.type != tensor.device.type:
                batch[key] = tensor.to(self.device)

        # Get embeddings for features and concatenate them to xlmr embeddings
        upos_emb = self.upos_embeddings(batch['upos'])
        ner_emb = self.ner_embeddings(batch['ner'])
        node_embs: Float[Tensor, "batch max_nodes d_node"] = torch.cat([batch['xlmr'], upos_emb, ner_emb], dim=-1)
        # Add root node to each row of node embeddings
        expanded_root = self.root_node[None, None, :].expand(node_embs.size(0), -1, -1)
        node_embs = torch.cat([expanded_root, node_embs], dim=1)
        for encoder in self.backbone:
            node_embs = encoder(node_embs, relations=batch['relations'])
        node_embs = self.final_layer_norm(node_embs)
        return node_embs

    def freeze_layers(self, layer_idx: list[int]) -> None:
        layers = list(self.backbone)
        for i in layer_idx:
            for param in layers[i].parameters():
                param.requires_grad = False

    def freeze_embeddings(self) -> None:
        for param in self.parameters(recurse=False):
            param.requires_grad = False

class GraphMatcher(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.graph_encoder = RGAT(config)
        # self.mlp_scorer = MLP(10)
        # Smaller temperature -> Stronger discrimination between positive and negatives
        self.temperature: float = config.init_tau
        # Started at batch
        # self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / config.init_tau)))

        self.device = None

    def compute_similarity(
            self,
            nodes_1: Float[Tensor, "batch nodes1_max d"],
            relations_1: Float[Tensor, "batch nodes1_max nodes1_max"],
            nodes_2: Float[Tensor, "batch nodes2_max d"],
            relations_2: Float[Tensor, "batch nodes2_max nodes2_max"],
    ) -> Float[Tensor, "batch batch"]:
        # Normalize so matmul gives cosine similarity
        H_1 = F.normalize(nodes_1, dim=-1)
        H_2 = F.normalize(nodes_2, dim=-1)

        # Get mask of invalid pairs due to padding
        mask_1 = torch.diagonal(relations_1, dim1=-2, dim2=-1)
        mask_2 = torch.diagonal(relations_2, dim1=-2, dim2=-1)

        batch_size = nodes_1.size(0)
        scores = torch.empty(batch_size, batch_size, device=self.device)

        # TODO - optimize this
        #  - currently done this way since similarity matrices are different sizes for each item in batch
        #  - but might be a more efficient way to batch it (e.g. with GeomLoss)
        for i in range(batch_size):
            for j in range(batch_size):
                # Get similarity matrix and slice away padding
                n1 = mask_1[i].sum(dim=0, keepdim=True).item()
                n2 = mask_2[j].sum(dim=0, keepdim=True).item()
                sim = torch.matmul(H_1[i, :n1, :], H_2[j, :n2, :].transpose(-1, -2))
                # Transform to non-negative cost domain; [-1, 1] -> [0, 2]
                cost = 1.0 - sim
                # Scale cost matrix to [0, 1] for optimal stability
                cost = cost / 2.0

                # TODO - verify these parameters don't cause problems with training
                P = ot.unbalanced.sinkhorn_unbalanced(
                    # Uniform weights - most common for graph matching
                    a=torch.full((n1,), 1.0 / n1, device=self.device),
                    b=torch.full((n2,), 1.0 / n2, device=self.device),
                    M=cost,
                    # changed to 0.1 to try to prevent numerical errors
                    reg=0.1,  # 0.05 - Allows for fairly sharp but not 1-to-1 matching, common value for graph matching
                    reg_m=1.0,  # Penalty for unmatched mass (nodes) - not too small or large
                    method='sinkhorn',
                    numIterMax=5000,  # Increased from default (1000) to allow more time to converge
                    stopThr=2e-6,  # Sightly relaxed threshold to avoid choking on final minor iterations
                )
                graph_score = (P * sim).sum()  # TODO can extract features from P and use MLP to predict score
                scores[i, j] = graph_score
        return scores

    def forward(self, input: dict[str, dict[str, Tensor]]) -> Float[Tensor, "batch batch"]:
        self.device = next(self.parameters()).device
        outputs = []
        relations = []
        for lang_input in input.values():
            relations.append(lang_input['relations'])
            outputs.append(self.graph_encoder(lang_input))
        # Get pairwise similarity scores
        scores = self.compute_similarity(outputs[0], relations[0], outputs[1], relations[1])
        return scores

    def get_loss_on_batch(self, scores: Float[Tensor, "batch batch"]) -> Float[Tensor, ""]:
        # InfoNCE loss
        # with torch.no_grad():
        #     self.logit_scale.clamp_(0, torch.log(torch.tensor(100.0)).item())
        # scores = scores * self.logit_scale.exp()
        scores = scores / self.temperature

        labels = torch.arange(scores.size(0), device=self.device)
        loss_1 = F.cross_entropy(scores, labels)
        loss_2 = F.cross_entropy(scores.T, labels)
        loss = (loss_1 + loss_2) / 2
        return loss

    def get_accuracy_on_batch(self, scores: Float[Tensor, "batch batch"]) -> float:
        # Correct predictions are on diagonal of score matrix
        correct = torch.arange(scores.size(0), device=self.device)
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
    def add_dependency_embeddings(self) -> dict:
        """
        Needed because an unexpected dependency appeared (at batch 540)
        """
        for graph_encoder_block in self.graph_encoder.backbone:
            attn = graph_encoder_block.attention

            old_rel_key = attn.rel_key
            shape = old_rel_key.weight.shape
            attn.rel_key = nn.Embedding(shape[0] + 2, shape[1])
            attn.rel_key.weight[:shape[0]] = old_rel_key.weight

            old_rel_value = attn.rel_value
            attn.rel_value = nn.Embedding(shape[0] + 2, shape[1])
            attn.rel_value.weight[:shape[0]] = old_rel_value.weight

        return self.state_dict()

    @torch.no_grad()
    def add_encoder_layer(self) -> dict:
        new_backbone = [block for block in self.graph_encoder.backbone]
        new_backbone.append(GraphEncoderBlock(self.config))
        self.graph_encoder.backbone = nn.ModuleList(new_backbone)
        return self.state_dict()