import gc
import math
import os
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, fields
from typing import Any

import stanza
import torch
from datasets import Dataset
from dotenv import load_dotenv
from jaxtyping import Float
from stanza import DownloadMethod, Document
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, TokenizersBackend, PreTrainedModel
from transformers.utils import logging as hf_logging

from graph_builder import build_graph, SELF_REL, Graph, Node

from tensordict import TensorDict, pad_sequence

import multiprocessing as mp

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

class Processor(ABC):

    def __init__(self, use_gpu: bool, stop_event: mp.Event):
        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                print("Cuda not available, using cpu instead")
                use_gpu = False
        if not use_gpu:
            self.device = torch.device("cpu")
        self.use_gpu = use_gpu
        self.input_q = queue.Queue()
        self.output_q = queue.Queue(4)  # todo - max size if necessary
        self.stop_event = stop_event
        self.is_main_thread = True
        self.is_active = mp.Event()

    def __iter__(self):
        while not self.is_active.is_set() and not self.stop_event.is_set():
            time.sleep(0.01)
        while not self.stop_event.is_set():
            try:
                out = self.output_q.get(timeout=0.1)
                if out is None:
                    break
                yield out
            except queue.Empty:
                try:
                    if self.is_main_thread:
                        item = self.input_q.get(timeout=0.1)
                        self._out_put(self.process(item) if item is not None else None)
                except queue.Empty:
                    time.sleep(0.01)  # <-- Yield CPU to the collator thread

    def _out_put(self, item):
        while not self.stop_event.is_set():
            try:
                self.output_q.put(item, timeout=0.1)
                break
            except queue.Full:
                time.sleep(0.1)

    def _worker_loop(self):
        while not self.stop_event.is_set():
            try:
                input = self.input_q.get(timeout=0.1)
                if input is None:
                    break

                output = self.process(input)
                self._out_put(output)
                # Yield GIL right after putting data so collator can read it
                time.sleep(0.001)
            except queue.Empty:
                time.sleep(0.01)  # <-- Prevents worker threads from burning CPU
        self._out_put(None)

    def _run_in_cuda_stream(self) -> None:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self._worker_loop()


    def run_in_thread(self) -> None:
        self._init_processor()
        self.is_active.set()
        self.is_main_thread = False
        work_fn = self._run_in_cuda_stream if self.use_gpu else self._worker_loop
        t = threading.Thread(target=work_fn)
        t.start()


    def run_in_process(self) -> None:
        # Swap standard queues to MP-compatible queues
        self.input_q = mp.Queue()
        self.output_q = mp.Queue()

        if self.use_gpu:
            worker_fn = self._run_in_cuda_stream
        else:
            worker_fn = self._worker_loop

        p = mp.Process(target=worker_fn)
        p.start()

    def signal_no_more_input(self) -> None:
        self.add_input_to_pipeline(None)

    def add_input_to_pipeline(self, input) -> None:
        while not self.stop_event.is_set():
            try:
                self.input_q.put(input, timeout=0.1)
                break
            except queue.Full:
                time.sleep(0.01)

    @abstractmethod
    def _init_processor(self) -> None:
        pass

    @abstractmethod
    def process(self, input: list[str]) -> Any:
        pass

    @abstractmethod
    def dismantle(self) -> None:
        pass


class StanzaProcessor(Processor):

    def __init__(self, lang: str, processors: list[str], use_gpu: bool = True, stop_event: mp.Event = mp.Event()):
        super().__init__(use_gpu, stop_event)

        self.lang = lang
        self.processors = processors
        self.nlp = None

    def _init_processor(self) -> None:
        self.nlp = stanza.Pipeline(
            self.lang,
            processors=','.join(self.processors),
            tokenize_no_ssplit=True,
            use_gpu=self.use_gpu,
            download_method=DownloadMethod.REUSE_RESOURCES,
            verbose=False,
        )
        self.is_active.set()

    @torch.no_grad()  # todo get rid of nograd if don't need
    def process(self, sentences: list[str]) -> list[Document]:
        # Since tokenize_no_split=True, there should only be one sentence in each doc
        docs = self.nlp.bulk_process(sentences)
        return docs

    def dismantle(self):
        for processor_name in list(self.nlp.processors.keys()):
            del self.nlp.processors[processor_name]
        self.nlp.processors.clear()
        del self.nlp
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class XLMRProcessor(Processor):

    def __init__(self, model_name: str, use_gpu: bool = True, stop_event: mp.Event = mp.Event()):
        super().__init__(use_gpu, stop_event)  # todo use use_gpu

        self.model_name = model_name
        self.emb_tokenizer = None
        self.emb_model = None

    def _init_processor(self) -> None:
        self.emb_tokenizer: TokenizersBackend = AutoTokenizer.from_pretrained(self.model_name)
        self.emb_model: PreTrainedModel = AutoModel.from_pretrained(self.model_name, device_map="auto")

    @torch.no_grad()
    def process(self, sentences: list[str]) -> TensorDict:
        # Tokenize for XLM-R
        xlmr_inputs = self.emb_tokenizer(
            sentences,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        xlmr_offsets = xlmr_inputs.pop("offset_mapping")
        xlmr_special_tokens_mask = xlmr_inputs.pop("special_tokens_mask")
        actual_token_mask = (xlmr_special_tokens_mask == 0) & (xlmr_inputs["attention_mask"] == 1)

        target_device = next(self.emb_model.parameters()).device

        # Pin memory for non-blocking GPU transfer # todo why?
        xlmr_inputs = {
            k: v.pin_memory().to(target_device, non_blocking=True)
            for k, v in xlmr_inputs.items()
        }

        # Get XLM-R embeddings
        xlmr_inputs = {k: v.to(next(self.emb_model.parameters()).device) for k, v in xlmr_inputs.items()}
        xlmr_outputs = self.emb_model(**xlmr_inputs)
        xlmr_hidden: Float[Tensor, "batch max_xlmr_len hidden"] = xlmr_outputs.last_hidden_state

        return TensorDict({
            "offsets": xlmr_offsets.pin_memory().to(target_device, non_blocking=True),
            "token_mask": actual_token_mask.pin_memory().to(target_device, non_blocking=True),
            "hidden": xlmr_hidden,
        }, batch_size=len(sentences))

    def dismantle(self):
        del self.emb_tokenizer, self.emb_model
        self.emb_tokenizer = self.emb_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class Preprocessor:

    def __init__(
            self,
            dataset: Dataset,
            embedding_model: str,
            langs: list[str],
            batch_size: int,
            output_batch_size: int,
            show_pbar=True,  # TODO difference between loading processor pbar and processing pbar
            pbar_pos: int = 0,
            multithread: bool = True,
            separate_thread: bool = False,
            use_gpu: bool = True,
            in_stages: bool = False,
            verbose: bool = False,
    ):
        assert len(langs) == 2, "Preprocessor should only be initialized with two languages"
        assert batch_size % output_batch_size == 0, "Batch size must be divisible by the output batch size"
        self.langs = langs
        self.in_size = batch_size
        self.out_size = output_batch_size
        self.num_batches = math.ceil(len(dataset) / output_batch_size)
        self.dataloader = iter(DataLoader(dataset, batch_size=self.in_size, collate_fn=self._sentence_collate_fn))
        self.pbar_pos = pbar_pos
        self.show_pbar = show_pbar
        self.multithread = multithread
        self.use_gpu = use_gpu
        self.embedding_model = embedding_model
        self.in_stages = in_stages
        self.verbose = verbose
        # todo move below out
        load_dotenv()  # Load hugging face token (HF_TOKEN="")
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        self.stop_event = mp.Event()

        self.nlps: dict[str, StanzaProcessor] = {}
        self.nlp_iters = {}
        if not in_stages:
            self._init_nlps(use_gpu)

        self.xlmr = None
        self.xlmr_iter = None
        if not in_stages:
            self._init_xlmr(use_gpu)

        self.q = deque()
        self.q_lock = threading.Lock()
        self.active = True

        self.thread = separate_thread
        if separate_thread:
            worker = threading.Thread(target=self._fill_queue)
            worker.start()

    def _init_nlps(self, use_gpu: bool) -> None:
        if self.verbose:
            pos = self.pbar_pos if not self.in_stages else self.pbar_pos + 1
            pbar = tqdm(desc=f"Loading stanza pipelines", position=pos, leave=False, total=2)
        else:
            pbar = None
        for lang in self.langs:
            nlp = StanzaProcessor(
                lang,
                processors=['tokenize', 'pos', 'lemma', 'depparse', 'ner'],
                use_gpu=use_gpu,
                stop_event=self.stop_event,
            )
            if self.multithread:
                nlp.run_in_thread()
            self.nlps[lang] = nlp
            self.nlp_iters[lang] = iter(nlp)
            if pbar:
                pbar.update()
        if pbar:
            pbar.close()

    # TODO - clean up multi proc/thread arguments and what is passed to Preprocessor()
    def _init_xlmr(self, use_gpu: bool) -> None:
        if self.verbose:
            pos = self.pbar_pos if not self.in_stages else self.pbar_pos + 1
            pbar = tqdm(desc=f"Loading xlmr tokenizer and model", position=pos, leave=False, total=1)
        else:
            pbar = None
        self.xlmr = XLMRProcessor(self.embedding_model, use_gpu, self.stop_event)
        if self.multithread:
            self.xlmr.run_in_thread()
        self.xlmr_iter = iter(self.xlmr)
        if pbar:
            pbar.update()
            pbar.close()

    def start_processing(self) -> None:
        if not self.in_stages:
            for batch in self.dataloader:
                if self.stop_event.is_set(): # todo don't just eat entire dataset first?
                    break
                sentences = []
                for lang in self.langs:
                    sentences.extend(batch[lang])
                    self.nlps[lang].add_input_to_pipeline(batch[lang])
                self.xlmr.add_input_to_pipeline(sentences)
                time.sleep(0.01)
            for nlp in self.nlps.values():
                nlp.signal_no_more_input()
            self.xlmr.signal_no_more_input()
        else:
            # Stanza first, then xlmr
            self._init_nlps(self.use_gpu)
            if self.show_pbar:
                pbar = tqdm(desc="Processing with stanza", position=self.pbar_pos + 1, leave=False,
                            total=self.num_batches)
            else:
                pbar = None
            batches = [batch for batch in self.dataloader]
            for batch in batches:
                for lang in self.langs:
                    self.nlps[lang].add_input_to_pipeline(batch[lang])
            for nlp in self.nlps.values():
                nlp.signal_no_more_input()
            # Collect docs
            doc_lists: dict[str, list[list[Document]]] = {lang: [] for lang in self.langs}
            for docs in zip(*self.nlp_iters.values()):
                for i, lang in enumerate(self.langs):
                    doc_lists[lang].append(docs[i])
                if pbar:
                    pbar.update(len(docs[0]))
            self.nlp_iters = {lang: iter(doc_lists[lang]) for lang in self.langs}
            for nlp in self.nlps.values():
                nlp.dismantle()
            self._init_xlmr(self.use_gpu)
            for batch in batches:
                sentences = []
                for lang in self.langs:
                    sentences.extend(batch[lang])
                self.xlmr.add_input_to_pipeline(sentences)
            self.xlmr.signal_no_more_input()

    def __iter__(self):
        if self.show_pbar:
            self.pbar = tqdm(position=self.pbar_pos, leave=True, total=self.num_batches,  # smoothing=0,
                             desc=f"Preprocessing {self.num_batches} batches")
        t = threading.Thread(target=self.start_processing) # todo make this more robust
        t.start()
        if not self.active:
            raise Exception("Preprocessor is not active - data already processed once")
        try:
            while not self.stop_event.is_set():
                try:
                    with self.q_lock:
                        item = self.q.popleft()
                    if item is None:
                        raise StopIteration
                    if self.show_pbar:
                        self.pbar.update(1)
                    yield item
                except IndexError:
                    if not self.thread:
                        self._add_to_queue()  # Will raise StopIteration if dataloader is empty
                    time.sleep(0.1)
        except StopIteration:
            if not self.in_stages:
                self.dismantle_processors()
            else:
                self.xlmr.dismantle()

    def terminate_processors(self, dismantle: bool = False):
        self.stop_event.set()
        if dismantle:
            self.dismantle_processors()

    def _add_to_queue(self):
        graphs = {}
        # if self.verbose:
        #     tqdm.write("Main thread: waiting for xlmr...")
        td_merged = next(self.xlmr_iter)
        embs = {lang: td for lang, td in zip(self.langs, td_merged.chunk(chunks=2, dim=0))}
        # if self.verbose:
        #     tqdm.write("Main thread: waiting for stanza...")
        for lang in self.langs:
            docs = next(self.nlp_iters[lang])
            aligned_emb = self.get_aligned_embeddings(docs, embs[lang])
            graphs[lang] = [build_graph(doc, aligned_emb[i], lang) for i, doc in enumerate(docs)]
        # if self.verbose:
        #     tqdm.write("Main thread: got batch, converting to tds...")
        # todo merge en and zh first?
        tds = {
            lang: self.graphs_to_tensordict(graphs[lang]).split(split_size=self.out_size, dim=0)
            for lang in self.langs
        }
        num_out = len(next(iter(tds.values())))
        tds_list = [
            {lang: val[i] for lang, val in tds.items()} for i in range(num_out)
        ]
        # if self.verbose:
        #     tqdm.write("Main thread: Conversion done, putting in queue.")
        with self.q_lock:
            for td in tds_list:
                self.q.append(td)
        # if self.verbose:
        #     tqdm.write("Main thread: Added to queue!")

    def _fill_queue(self):
        try:
            while not self.stop_event.is_set():
                self._add_to_queue()
        except StopIteration:
            pass

    def dismantle_processors(self):
        if not self.active:
            return
        for nlp in self.nlps.values():
            nlp.dismantle()
        self.xlmr.dismantle()
        self.active = False

    # TODO keep start/end tokens? - maybe for when expand to not be perfect sentence alignment
    # todo move out of Preprocessor?
    @torch.no_grad()
    def get_aligned_embeddings(
            self,
            docs: list[Document], # todo - take the graphs instead - stanza processor constructs and returns graphs
            embeddings: TensorDict,
    ) -> Float[Tensor, "batch max_words hidden"]:
        """
        Note: GPT helped me rewrite this function to parallelize - loops were causing bottleneck
        NOTE: This logic only works for languages where MWTs map directly back to source text
        """
        hidden = embeddings["hidden"]
        device = hidden.device
        batch_size = len(docs)

        # 1. Extract XLM-R offsets (already on GPU)
        # offsets shape: [B, MaxXlmrTokens, 2]
        xlmr_offsets = embeddings["offsets"]
        xlmr_starts = xlmr_offsets[..., 0].unsqueeze(1)  # Shape: [B, 1, MaxXlmrTokens]
        xlmr_ends = xlmr_offsets[..., 1].unsqueeze(1)  # Shape: [B, 1, MaxXlmrTokens]

        # 2. Gather all word spans on the CPU first (very fast)
        max_words = max(doc.num_words for doc in docs)

        target_dtype = xlmr_offsets.dtype

        word_starts = torch.zeros((batch_size, max_words), dtype=target_dtype)
        word_ends = torch.zeros((batch_size, max_words), dtype=target_dtype)
        word_mask = torch.zeros((batch_size, max_words), dtype=torch.bool)

        for b_idx, doc in enumerate(docs):
            word_idx = 0
            for tok in doc.iter_tokens():
                word_start = tok.start_char
                for word in tok.words:
                    word_starts[b_idx, word_idx] = word_start
                    word_ends[b_idx, word_idx] = word_start + len(word.text)
                    word_mask[b_idx, word_idx] = True

                    word_start += len(word.text)
                    word_idx += 1

        # 2. Push word metadata to the GPU in a single quick transfer
        word_starts = word_starts.to(device, non_blocking=True).unsqueeze(-1)  # Shape: [B, MaxWords, 1]
        word_ends = word_ends.to(device, non_blocking=True).unsqueeze(-1)  # Shape: [B, MaxWords, 1]
        word_mask = word_mask.to(device, non_blocking=True).unsqueeze(-1)  # Shape: [B, MaxWords, 1]

        # 4. Compute vectorized intersections using tensor broadcasting
        overlap_starts = torch.maximum(xlmr_starts, word_starts)
        overlap_ends = torch.minimum(xlmr_ends, word_ends)
        has_overlap = overlap_starts < overlap_ends  # Shape: [B, MaxWords, MaxXlmrTokens]
        # todo - for all entity tokens, overlap combination of cmoposite entity t

        # 5. Apply the special token / valid word masks
        token_mask = embeddings["token_mask"].unsqueeze(1)  # Shape: [B, 1, MaxXlmrTokens]

        # Create alignment matrix for mean pooling of embeddings of each word's sub-words
        # Matrix is 1 where there is an overlap, the token is valid, and the word exists
        alignment_matrix = (has_overlap & token_mask & word_mask).to(hidden.dtype)

        # 6. Normalize rows for Mean Pooling (prevent division by zero)
        row_sums = alignment_matrix.sum(dim=-1, keepdim=True)
        row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
        normalized_alignment = alignment_matrix / row_sums

        # 7. Batch Matrix Multiplication to project to Token space
        aligned_word_embeddings = torch.bmm(normalized_alignment, hidden)

        return aligned_word_embeddings

    # todo make this function called by worker thread
    @torch.no_grad()
    def graphs_to_tensordict(self, graphs: list[Graph]) -> TensorDict:
        max_nodes = max(len(graph.nodes) for graph in graphs)
        batch_size = len(graphs)

        device = graphs[0].nodes[0].xlmr.device

        keys = [field.name for field in fields(Node)]

        # Get batched tensors for node features
        node_tds = []
        for g in graphs:
            collated_nodes = {}
            for key in keys:
                raw_features = [getattr(node, key) for node in g.nodes]

                first_item = raw_features[0]

                if isinstance(first_item, torch.Tensor):
                    collated_nodes[key] = torch.stack(raw_features).to(device, non_blocking=True)
                else:
                    collated_nodes[key] = torch.tensor(raw_features, dtype=torch.int, device=device)

            node_tds.append(TensorDict(collated_nodes, batch_size=len(g.nodes)))

        # todo - pad in batches of size out_batch?
        # Pad sequences across batch
        graphs_td = pad_sequence(node_tds, padding_value=0, return_mask=False)
        graphs_td.batch_size = torch.Size([batch_size])

        graphs_td["lang"] = torch.tensor([graph.lang for graph in graphs], device=device)
        # Add + 1 since root node will be added at position 0 in the RGAT
        graphs_td["node_mask"] = torch.zeros((batch_size, max_nodes + 1), dtype=torch.bool, device=device)
        graphs_td["relations"] = torch.zeros((batch_size, max_nodes + 1, max_nodes + 1), dtype=torch.int, device=device)

        # Add self loops
        self_idx = torch.arange(max_nodes + 1, device=device)
        graphs_td["relations"][:, self_idx, self_idx] = SELF_REL  # =1

        for b_idx, g in enumerate(graphs):
            # Root node + valid graph nodes are set to 1
            graphs_td["node_mask"][b_idx, :len(g.nodes) + 1] = True
            # Add edges between nodes
            heads = torch.tensor([edge.head for edge in g.edges], dtype=torch.int, device=device)
            targets = torch.tensor([edge.target for edge in g.edges], dtype=torch.int, device=device)
            relations = torch.tensor([edge.relation for edge in g.edges], dtype=torch.int, device=device)
            # Relations between nodes start at 2 and alternate with reverse edges
            # forward edge: 2 -> 2, 3 -> 4, 4 -> 6
            graphs_td["relations"][b_idx, heads, targets] = (relations * 2) - 2
            # reverse edge: 2 -> 3, 3 -> 5, 4 -> 7
            graphs_td["relations"][b_idx, targets, heads] = (relations * 2) - 1

        return graphs_td

    def _sentence_collate_fn(self, batch: list[dict[str, dict[str, str]]]) -> dict[str, list[str]]:
        sentences = {lang: [pair['translation'][lang] for pair in batch] for lang in self.langs}
        return sentences
