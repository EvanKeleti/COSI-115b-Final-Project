from __future__ import annotations

import itertools
import multiprocessing
import os
import queue
import shutil
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Optional

import datasets
import stanza
from dacite import from_dict, Config
from datasets import Dataset, concatenate_datasets
from dotenv import load_dotenv
from jaxtyping import Float
from ot.backend import torch
from stanza import Document, DownloadMethod
from tensordict import TensorDict, pad_sequence
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, TokenizersBackend
from typing_extensions import override

from data import load_wmt_data, CACHE_DIR
from graph_builder import Graph, build_graph, SELF_REL, Node


# todo decrease script startup time


class StopExecution(Exception):
    pass


class Processor(ABC):

    def __init__(
            self,
            in_queues: dict[str, queue.Queue | multiprocessing.Queue],
            out_queues: dict[str, queue.Queue | multiprocessing.Queue],
            stop_event: multiprocessing.Event,
    ):
        """

        :param in_queues:
        :param out_queues:
        :param stop_event:
        """
        self.in_queues = in_queues
        self.out_queues = out_queues
        self.stop_event = stop_event

    def _can_continue(self) -> bool:
        if self.stop_event.is_set():
            raise StopExecution
        return True

    def _get_input(self) -> dict[str, Any]:
        inputs = {}
        for key in self.in_queues.keys():
            while self._can_continue():
                try:
                    inputs[key] = self.in_queues[key].get(timeout=0.1)
                    break
                except queue.Empty:
                    time.sleep(0.01)  # <-- Prevents worker threads from burning CPU
        return inputs

    def _put_output(self, outputs: dict[str, Any]) -> None:
        for key, value in outputs.items():
            while self._can_continue():
                try:
                    self.out_queues[key].put(value, timeout=0.1)
                    break
                except queue.Full:
                    time.sleep(0.01)

    def get_output(self) -> Any:
        output = {}
        for key in self.in_queues.keys():
            while self._can_continue():
                try:
                    output[key] = self.out_queues[key].get(timeout=0.1)
                    break
                except queue.Empty:
                    time.sleep(0.01)  # <-- Prevents worker threads from burning CPU
        return output

    def _worker_loop(self):
        try:
            while self._can_continue():
                inputs = self._get_input()
                outputs = self.process(inputs) if inputs is not None else None
                self._put_output(outputs)

                # Yield GIL right after putting data
                time.sleep(0.001)
        except StopExecution:
            pass  # Exit gracefully

    def _init_processor(self):
        """
        Called by `start_processor`, should handle all necessary setup. Override if needed.
        """
        pass

    def start_processor(self, cuda_stream: bool = False):
        """
        Calls `_init_processor` before starting work in a separate thread.
        :param cuda_stream: Whether to start the processor with a separate CUDA stream.
        """

        def work_fn():
            self._init_processor()
            if cuda_stream:
                import torch
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    self._worker_loop()
            else:
                self._worker_loop()

        t = threading.Thread(target=work_fn)
        t.start()

    @abstractmethod
    def process(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Processes inputs and returns outputs. Must pass on `None` poison pill as return dictionary value.
        """
        pass


class StanzaPipeline(Processor):

    def __init__(
            self,
            in_queues: dict[str, queue.Queue | multiprocessing.Queue],
            out_queues: dict[str, queue.Queue | multiprocessing.Queue],
            stop_event: multiprocessing.Event,
            processors: list[str],
            use_gpu: bool = True,
    ):
        super().__init__(in_queues, out_queues, stop_event)

        # todo move this logic to parent
        if len(in_queues) != 1:
            raise ValueError(f"StanzaPipeline can only take a single input queue. Received {len(in_queues)} instead.")
        if len(out_queues) != 1:
            raise ValueError(f"StanzaPipeline can only take a single output queue. Received {len(out_queues)} instead.")

        self.lang = next(iter(in_queues.keys()))
        self.processors = processors
        self.use_gpu = use_gpu
        self.nlp = None

    @override
    def _init_processor(self) -> None:
        self.nlp = stanza.Pipeline(
            self.lang,
            processors=','.join(self.processors),
            tokenize_no_ssplit=True,
            use_gpu=self.use_gpu,
            download_method=DownloadMethod.REUSE_RESOURCES,
            verbose=False,
        )

    def process(self, inputs: dict[str, list[str]]) -> dict[str, list[Document]]:
        return {self.lang: self.nlp.bulk_process(inputs[self.lang]) if inputs[self.lang] is not None else None}


class GraphBuilder(Processor):
    """
    Takes docs from 1 or more StanzaPipelines, collates and converts them into Graphs, and outputs the graphs.
    """

    def __init__(  # todo remove if unnecessary
            self,
            in_queues: dict[str, queue.Queue],
            out_queues: dict[str, multiprocessing.Queue],
            stop_event: multiprocessing.Event,
    ):
        super().__init__(in_queues, out_queues, stop_event)

    def process(self, inputs: dict[str, list[Document]]) -> dict[str, list[Graph]]:
        return {
            lang: [build_graph(doc, lang) for doc in docs] if docs is not None else None
            for lang, docs in inputs.items()
        }


class DataSplitter(Processor):
    """
    Reads wmt data and outputs it to a separate queue for each language.
    """

    def __init__(
            self,
            dataset: Dataset,
            out_queues: dict[str, queue.Queue],
            stop_event: multiprocessing.Event,
            langs: list[str],
            batch_size: int,
    ):
        super().__init__({}, out_queues, stop_event)

        self.langs = langs
        self.dataloader = iter(DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda batch: {lang: [pair['translation'][lang] for pair in batch] for lang in self.langs}
        ))

    def process(self, inputs: dict) -> dict[str, list[str] | None]:
        """
        Should always receive an empty dict as input since there are no input queues.
        Outputs poison pill (`None`) when dataset has been fully read.
        """
        try:
            return next(self.dataloader)
        except StopIteration:
            return {lang: None for lang in self.langs}


class SentencesToGraphs:
    """
    Coordinates a StanzaPipeline and a GraphBuilder.
    """

    def __init__(
            self,
            lang: str,
            in_queue: multiprocessing.Queue,
            out_queue: multiprocessing.Queue,
            stop_event: multiprocessing.Event,
    ):
        self.lang = lang
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.stop_event = stop_event

    def __call__(self):
        sp_to_gb = {self.lang: queue.Queue()}
        self.nlp = StanzaPipeline(
            in_queues={self.lang: self.in_queue},
            out_queues=sp_to_gb,
            processors=['tokenize', 'pos', 'lemma', 'depparse', 'ner'],
            stop_event=self.stop_event,
        )
        self.gb = GraphBuilder(
            in_queues=sp_to_gb,
            out_queues={self.lang: self.out_queue},
            stop_event=self.stop_event,
        )

        self.nlp.start_processor()
        self.gb.start_processor()


class GraphCollator:
    """
    Coordinates two SentencesToGraphs corresponding to the given dataset and langs.
    """

    def __init__(
            self,
            langs: list[str],
            batch_size: int,
            split: Optional[str] = None,
            selection_range: Optional[tuple[int, int]] = None,
            dataset: Optional[Dataset] = None,
            save_graphs: bool = True,
            save_interval: int = 100,
    ):
        if dataset is None and (split is None or selection_range is None):
            raise ValueError("Must give either a dataset or a split and index range.")
        if dataset is not None and (split is not None or selection_range is not None):
            raise ValueError("Must give either a dataset or a split and index range.")
        if split is not None and split != 'train' and split != 'validation':
            raise ValueError("Split must be either 'train' or 'validation'.")

        self.save = False if dataset else save_graphs
        self.save_interval = save_interval

        self.finished_data = None
        self.selected_finished = None
        self.data_dir = CACHE_DIR / split
        if not dataset:  # todo make not need to save from beginning
            if self.data_dir.exists():
                self.finished_data = datasets.load_from_disk(self.data_dir)
                tqdm.write(f"Found {len(self.finished_data)} preprocessed graph pairs.")
                if selection_range[0] < len(self.finished_data):
                    end = min(len(self.finished_data), selection_range[1])
                    self.selected_finished = self.finished_data.select(range(selection_range[0], end))
                    selection_range = (end, selection_range[1])
                    self.save = True
                else:
                    self.save = True if selection_range[0] == len(self.finished_data) else save_graphs  # can't save without breaking order

            if selection_range[0] < selection_range[1]:
                dataset = load_wmt_data(split, range(selection_range[0], selection_range[1]))

        self.langs = langs
        self.batch_size = batch_size
        self.stop_event = multiprocessing.Event()

        self.data_queues = {lang: multiprocessing.Queue(maxsize=20) for lang in langs}
        self.graph_out_queues = {lang: multiprocessing.Queue() for lang in langs}

        if dataset is not None:
            self.data_splitter = DataSplitter(
                dataset=dataset,
                out_queues=self.data_queues,
                stop_event=self.stop_event,
                langs=langs,
                batch_size=batch_size,
            )
            self.sents_to_graphs = {
                lang: SentencesToGraphs(
                    lang=lang,
                    in_queue=self.data_queues[lang],
                    out_queue=self.graph_out_queues[lang],
                    stop_event=self.stop_event,
                )
                for lang in langs
            }
        else:
            self.data_splitter = self.sents_to_graphs = None

        self.active = False

    def start_processing(self) -> None:
        if self.data_splitter is None or self.active:
            return

        self.data_splitter.start_processor()
        tqdm.write("Spawning stanza pipelines...", end="")
        ctx = multiprocessing.get_context(method='spawn')
        for std in self.sents_to_graphs.values():
            p = ctx.Process(target=std)
            p.start()
        self.active = True

    def save_finished(self, reload: bool = True):
        if self.save and self.active:
            temp_path = Path(str(self.data_dir) + "_temp")
            self.finished_data.save_to_disk(str(temp_path))  # todo, don't keep in memory till the end?
            try:
                del self.finished_data
                del self.selected_finished
            except AttributeError:
                pass
            if self.data_dir.exists():
                shutil.rmtree(self.data_dir)
            temp_path.rename(self.data_dir)
            if reload:
                self.finished_data = datasets.load_from_disk(self.data_dir)

    def stop_processing(self) -> None:
        self.stop_event.set()
        self.save_finished(reload=False)
        for q in self.data_queues.values():
            q.cancel_join_thread()
        for q in self.graph_out_queues.values():
            q.cancel_join_thread()
        self.active = False

    def __iter__(self):
        try:
            if self.selected_finished:
                config = Config(cast=[tuple])
                idx = 0
                while idx < len(self.selected_finished) and not self.stop_event.is_set():
                    graph_dict = self.selected_finished[idx:idx + self.batch_size]
                    graph_dict = {
                        lang: [from_dict(data_class=Graph, data=g, config=config) for g in graph_dict[lang]]
                        for lang in graph_dict.keys()
                    }
                    yield graph_dict
                    idx += self.batch_size

            if not self.active:
                self.start_processing()
            batch = 1
            while not self.stop_event.is_set():
                graphs = {}
                for lang, q in self.graph_out_queues.items():
                    while not self.stop_event.is_set():
                        try:
                            out = q.get(timeout=0.1)
                            if batch == 0:
                                tqdm.write("Done")
                            if out is None:
                                raise StopIteration
                            graphs[lang] = out
                            break
                        except queue.Empty:
                            time.sleep(0.01)
                if self.save:
                    new_ds = Dataset.from_dict({key: [asdict(val) for val in vals] for key, vals in graphs.items()})
                    if self.finished_data:
                        self.finished_data = concatenate_datasets([self.finished_data, new_ds])
                    else:
                        self.finished_data = new_ds
                    if batch % self.save_interval == 0:
                        self.save_finished()
                yield graphs
                batch += 1
        except Exception as e:
            if not isinstance(e, StopIteration):
                raise e


class Featurizer:

    def __init__(
            self,
            # embedding_model: str, # todo
            graph_collator: GraphCollator,
            out_batch_size: int,
    ):
        self.graphs_iter = iter(graph_collator)
        self.out_batch_size = out_batch_size  # todo

        self.model_name = "xlm-roberta-base"
        self.emb_tokenizer: TokenizersBackend = AutoTokenizer.from_pretrained(self.model_name)
        self.emb_model: PreTrainedModel = AutoModel.from_pretrained(self.model_name, device_map="auto")
        self.device = next(self.emb_model.parameters()).device

    def __iter__(self):
        try:
            while True:  # todo
                graphs_dict = next(self.graphs_iter)
                langs = []
                splits = []
                for lang, graphs in graphs_dict.items():
                    langs.append(lang)
                    splits.append(len(graphs))
                langs = graphs_dict.keys()
                graphs = list(itertools.chain(*graphs_dict.values()))
                td = self.graphs_to_tensordict(graphs)
                tds = td.split(split_size=list(splits), dim=0)
                td_dict = {lang: td for lang, td in zip(langs, tds)}
                yield td_dict
        except StopIteration:
            pass

    @torch.no_grad()
    def get_aligned_embeddings(self, graphs: list[Graph]) -> Float[Tensor, "batch max_nodes hidden"]:
        sentences = [g.text for g in graphs]

        # Tokenize for XLM-R
        xlmr_inputs = self.emb_tokenizer(
            sentences,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        ).to(device=self.device, non_blocking=True)
        xlmr_offsets = xlmr_inputs.pop("offset_mapping")
        xlmr_special_tokens_mask = xlmr_inputs.pop("special_tokens_mask")

        # Get XLM-R embeddings
        xlmr_outputs = self.emb_model(**xlmr_inputs)
        xlmr_hidden: Float[Tensor, "batch max_xlmr_len hidden"] = xlmr_outputs.last_hidden_state

        batch_size = len(graphs)

        # 1. Extract XLM-R offsets (already on GPU)
        # offsets shape: [B, MaxXlmrTokens, 2]
        xlmr_starts = xlmr_offsets[..., 0].unsqueeze(1)  # Shape: [B, 1, MaxXlmrTokens]
        xlmr_ends = xlmr_offsets[..., 1].unsqueeze(1)  # Shape: [B, 1, MaxXlmrTokens]

        # 2. Gather all word spans on the CPU first (very fast)
        max_nodes = max(len(g.nodes) for g in graphs)

        target_dtype = xlmr_offsets.dtype

        span_starts = torch.zeros((batch_size, max_nodes), dtype=target_dtype, pin_memory=True)
        span_ends = torch.zeros((batch_size, max_nodes), dtype=target_dtype, pin_memory=True)
        span_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool, pin_memory=True)

        for b_idx, graph in enumerate(graphs):
            for node_idx, node in enumerate(graph.nodes):
                span_starts[b_idx, node_idx] = node.span[0]
                span_ends[b_idx, node_idx] = node.span[1]
                span_mask[b_idx, node_idx] = True

        # 3. Push word metadata to the GPU in a single quick transfer
        span_starts = span_starts.to(self.device, non_blocking=True).unsqueeze(-1)  # Shape: [B, MaxNodes, 1]
        span_ends = span_ends.to(self.device, non_blocking=True).unsqueeze(-1)  # Shape: [B, MaxNodes, 1]
        span_mask = span_mask.to(self.device, non_blocking=True).unsqueeze(-1)  # Shape: [B, MaxNodes, 1]

        # 4. Compute vectorized intersections using tensor broadcasting
        overlap_starts = torch.maximum(xlmr_starts, span_starts)
        overlap_ends = torch.minimum(xlmr_ends, span_ends)
        has_overlap = overlap_starts < overlap_ends  # Shape: [B, MaxNodes, MaxXlmrTokens]

        # 5. Apply the special token / valid word masks
        token_mask = (xlmr_special_tokens_mask == 0) & (xlmr_inputs["attention_mask"] == 1)
        token_mask = token_mask.unsqueeze(1)  # Shape: [B, 1, MaxXlmrTokens]

        # Create alignment matrix for mean pooling of embeddings of each word's sub-words
        # Matrix is 1 where there is an overlap, the token is valid, and the word exists
        alignment_matrix = (has_overlap & token_mask & span_mask).to(xlmr_hidden.dtype)

        # 6. Normalize rows for Mean Pooling (prevent division by zero)
        row_sums = alignment_matrix.sum(dim=-1, keepdim=True)
        row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
        normalized_alignment = alignment_matrix / row_sums

        # 7. Batch Matrix Multiplication to project to Token space
        aligned_word_embeddings = torch.bmm(normalized_alignment, xlmr_hidden)

        return aligned_word_embeddings

    @torch.no_grad()
    def graphs_to_tensordict(self, graphs: list[Graph]) -> TensorDict:
        max_nodes = max(len(graph.nodes) for graph in graphs)
        batch_size = len(graphs)

        keys = [field.name for field in fields(Node)]

        # Get batched tensors for node features
        node_tds = []
        for g in graphs:
            collated_nodes = {}
            for key in keys:
                first_item = getattr(g.nodes[0], key)

                if isinstance(first_item, int):
                    raw_features = [getattr(node, key) for node in g.nodes]
                    collated_nodes[key] = torch.tensor(raw_features, dtype=torch.int, device=self.device)
                # else:
                #     collated_nodes[key] = torch.stack(raw_features).to(self.device, non_blocking=True)

            node_tds.append(TensorDict(collated_nodes, batch_size=len(g.nodes)))

        # todo - pad in batches of size out_batch?
        # Pad sequences across batch
        graphs_td = pad_sequence(node_tds, padding_value=0, return_mask=False)
        graphs_td.batch_size = torch.Size([batch_size])

        graphs_td["lang"] = torch.tensor([graph.lang for graph in graphs], device=self.device)
        # Add + 1 since root node will be added at position 0 in the RGAT
        graphs_td["node_mask"] = torch.zeros((batch_size, max_nodes + 1), dtype=torch.bool, device=self.device)
        graphs_td["relations"] = torch.zeros((batch_size, max_nodes + 1, max_nodes + 1), dtype=torch.int,
                                             device=self.device)

        # Add self loops
        self_idx = torch.arange(max_nodes + 1, device=self.device)
        graphs_td["relations"][:, self_idx, self_idx] = SELF_REL  # =1

        for b_idx, g in enumerate(graphs):
            # Root node + valid graph nodes are set to 1
            graphs_td["node_mask"][b_idx, :len(g.nodes) + 1] = True
            # Add edges between nodes
            heads = torch.tensor([edge.head for edge in g.edges], dtype=torch.int, device=self.device)
            targets = torch.tensor([edge.target for edge in g.edges], dtype=torch.int, device=self.device)
            relations = torch.tensor([edge.relation for edge in g.edges], dtype=torch.int, device=self.device)
            # Relations between nodes start at 2 and alternate with reverse edges
            # forward edge: 2 -> 2, 3 -> 4, 4 -> 6
            graphs_td["relations"][b_idx, heads, targets] = (relations * 2) - 2
            # reverse edge: 2 -> 3, 3 -> 5, 4 -> 7
            graphs_td["relations"][b_idx, targets, heads] = (relations * 2) - 1

        graphs_td['xlmr'] = self.get_aligned_embeddings(graphs)

        return graphs_td


if __name__ == '__main__':
    load_dotenv()
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    num_sentences = 10000 * 8
    batch_size = 32
    num_batches = num_sentences / batch_size

    finished_data = datasets.load_from_disk(CACHE_DIR / 'train')
    start = len(finished_data)
    del finished_data

    # dataset = load_wmt_data('train', range(num_sentences))

    graph_collator = GraphCollator(
        langs=['zh', 'en'],
        batch_size=batch_size,
        split='train',
        selection_range=(start, num_sentences),
        save_graphs=True,
    )
    # featurizer = Featurizer(graph_collator, 8)
    try:
        pbar = tqdm(initial=start, total=num_sentences, desc=f"Building {num_sentences} graph_pairs")
        with pbar:
            for batch in graph_collator:
                pbar.update(len(batch['en']))
    finally:
        graph_collator.stop_processing()
