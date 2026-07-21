import os
from dataclasses import fields

import stanza
import torch
from dotenv import load_dotenv
from jaxtyping import Float, Int
from stanza import DownloadMethod, Document
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from graph_builder import build_graph, Node, Edge, SELF_REL

load_dotenv()  # Load hugging face token (HF_TOKEN="")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, device_map="auto")

langs = ['zh', 'en']
# todo - wrap in an init function so can import this file without always loading nlps
# TODO check if more efficient overall to run stanza on cpu cores, or if there is a way to use with accelerate
NLPS = {
    lang: stanza.Pipeline(
        lang,
        processors='tokenize,pos,lemma,depparse,ner',
        tokenize_no_ssplit=True,
        use_gpu=True,
        download_method=DownloadMethod.REUSE_RESOURCES,
    ) for lang in langs
}


# TODO keep start/end tokens? - maybe for when expand to not be perfect sentence alignment
@torch.no_grad()
def get_features(
        sentences: list[str],
        lang: str,
) -> tuple[list[Document], Float[Tensor, "batch max_words hidden"]]:
    # Get stanza annotations
    # Since tokenize_no_split=True, there should only be one sentence in each doc
    docs: list[Document] = NLPS[lang].bulk_process(sentences)

    # Tokenize for XLM-R
    xlmr_inputs = tokenizer(
        sentences,
        padding=True,
        return_tensors="pt",
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    xlmr_offsets = xlmr_inputs.pop("offset_mapping")
    xlmr_special_tokens_mask = xlmr_inputs.pop("special_tokens_mask")
    actual_token_mask = (xlmr_special_tokens_mask == 0) & (xlmr_inputs["attention_mask"] == 1)

    # Get XLM-R embeddings
    xlmr_inputs = {k: v.to(next(model.parameters()).device) for k, v in xlmr_inputs.items()}
    xlmr_outputs = model(**xlmr_inputs)
    xlmr_hidden: Float[Tensor, "batch max_xlmr_len hidden"] = xlmr_outputs.last_hidden_state

    # Create alignment matrix for mean pooling of embeddings of each word's sub-words
    max_words = max(doc.num_words for doc in docs)
    max_xlmr_len = xlmr_hidden.size(1)
    alignment_matrix: Float[Tensor, "batch max_words max_xlmr_len"] = torch.zeros(
        (len(sentences), max_words, max_xlmr_len), dtype=torch.int)

    for b_idx, doc in enumerate(docs):
        word_idx = 0
        for tok in doc.iter_tokens():
            # NOTE: This logic only works for languages where MWTs map directly back to source text
            word_start = tok.start_char
            for word in tok.words:
                for xlmr_idx, (xlmr_start, xlmr_end) in enumerate(xlmr_offsets[b_idx].tolist()):
                    overlap_start = max(xlmr_start, word_start)
                    overlap_end = min(xlmr_end, word_start + len(word.text))
                    # Skip XLM-R special tokens/padding tokens
                    if not actual_token_mask[b_idx, xlmr_idx]:
                        continue
                    if overlap_start < overlap_end:
                        alignment_matrix[b_idx, word_idx, xlmr_idx] = 1
                word_start += len(word.text)
                word_idx += 1

    # Normalize rows for Mean Pooling (prevent division by zero for padded spots)
    row_sums = alignment_matrix.sum(dim=-1, keepdim=True)
    row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
    normalized_alignment = alignment_matrix / row_sums
    # Batch Matrix Multiplication to project XLM-R embeddings to Token space
    aligned_word_embeddings: Float[Tensor, "batch max_words hidden"] = torch.bmm(
        normalized_alignment.to(xlmr_hidden.device),
        xlmr_hidden
    )

    return docs, aligned_word_embeddings


def make_graphs(
        docs: list[Document],
        embeddings: Float[Tensor, "batch max_words hidden"]
) -> tuple[list[list[Node]], list[list[Edge]]]:
    nodes_lists, edges_lists = [], []
    for i, doc in enumerate(docs):
        graph = build_graph(doc, embeddings[i])
        nodes_lists.append(graph["nodes"])
        edges_lists.append(graph["edges"])
    return nodes_lists, edges_lists


def sentences_to_graph_tensors(
        sentences: list[str],
        lang: str,
) -> dict[str, Float[Tensor, "batch max_nodes d_xlmr"] | Int[Tensor, "batch max_nodes max_nodes"]]:
    docs, embeddings = get_features(sentences, lang)
    # Send to cpu for creation of graph tensors
    nodes_lists, edges_lists = make_graphs(docs, embeddings.to(torch.device("cpu")))

    graph_tensors = {}

    max_nodes = max(len(nodes) for nodes in nodes_lists)
    # Tensor that represent relations between nodes and self relations for each node
    batch_size = len(sentences)
    # + 1 since root node will be added at position 0 in the RGAT
    graph_tensors["relations"] = adjacency_tensor = torch.zeros((batch_size, max_nodes + 1, max_nodes + 1), dtype=torch.int)
    # Add self loops
    self_idx = torch.arange(max_nodes + 1)
    adjacency_tensor[:, self_idx, self_idx] = SELF_REL  # =1

    first_node = nodes_lists[0][0]
    for field in fields(Node):
        name = field.name
        val = getattr(first_node, name)
        if isinstance(val, Tensor):
            graph_tensors[name] = torch.zeros(batch_size, max_nodes, val.size(0), dtype=val.dtype)
        else:
            graph_tensors[name] = torch.zeros(batch_size, max_nodes, dtype=torch.int)

    for b_idx, (nodes, edges) in enumerate(zip(nodes_lists, edges_lists)):
        # Add edges between nodes
        for edge in edges:
            # Relations between nodes start at 2 and alternate with reverse edges
            # forward edge: 2 -> 2, 3 -> 4, 4 -> 6
            adjacency_tensor[b_idx][edge.head, edge.target] = (edge.relation * 2) - 2
            # reverse edge: 2 -> 3, 3 -> 5, 4 -> 7
            adjacency_tensor[b_idx][edge.target, edge.head] = (edge.relation * 2) - 1

        for field in fields(Node):
            name = field.name
            graph_tensors[name][b_idx, :len(nodes)] = torch.stack([torch.as_tensor(getattr(node, name)) for node in nodes])

    return graph_tensors


def graph_collate_fn(batch: list[dict[str, dict[str, str]]]) -> dict[str, dict[str, Tensor]]:
    langs = list(batch[0]['translation'].keys())
    assert len(langs) == 2
    sentences = {lang: [pair['translation'][lang] for pair in batch] for lang in langs}

    tensor_dict = {lang: sentences_to_graph_tensors(sentences[lang], lang) for lang in langs}

    return tensor_dict
