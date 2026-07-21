import asyncio
import json
import os
import pickle
import re
import textwrap

import datasets
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from client.providers import Query
from client.query import query_model_with_backoff
from graph_builder import NER_MAP, REL_MAP, UPOS_MAP
from model import GraphMatcher, RGAT, ModelConfig

os.makedirs("test", exist_ok=True)

# Light sanity-check tests to ensure code runs with no exceptions or errors thrown

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


def test_preprocess():
    from preprocess import get_features
    datasets.enable_progress_bars()
    dataset = datasets.load_dataset("wmt/wmt19", "zh-en", streaming=False)
    data = get_features(dataset['train'][:2])
    with open("test/preprocess_output.pkl", "wb") as f:
        pickle.dump(data, f)


# def test_build_graphs():
#     with open("test/preprocess_output.pkl", "rb") as f:
#         data = pickle.load(f)
#     graphs = build_graphs(data)
#     batch = []
#     for graph_en, graph_zh in zip(graphs['en'], graphs['zh']):
#         batch.append({'en': graph_en, 'zh': graph_zh})
#     with open("test/build_graphs_output.pkl", "wb") as f:
#         pickle.dump(data, f)
#     batch = graph_collate_fn(batch)
#     with open("test/model_input.pkl", "wb") as f:
#         pickle.dump(batch, f)


def test_data_load():
    from preprocess import graph_collate_fn
    datasets.enable_progress_bars()
    dataset = datasets.load_dataset("wmt/wmt19", "zh-en", streaming=False)
    train_loader = DataLoader(
        dataset["train"],
        batch_size=2,
        collate_fn=graph_collate_fn,
    )

    for batch in train_loader:
        print(textwrap.fill(str(batch), width=150))
        with open("test/test_data.pkl", "wb") as f:
            pickle.dump(batch, f)
        break


def test_graph_encoder():
    with open("test/model_input.pkl", "rb") as f:
        batch = pickle.load(f)
    model = RGAT(config)
    output = model(batch['en'])
    print(output)


def test_graph_matcher():
    with open("test/model_input.pkl", "rb") as f:
        batch = pickle.load(f)
    model = GraphMatcher(config).to(torch.device("cuda"))
    loss = model.get_loss_on_batch(batch)
    print(type(loss))
    print(loss)


def test_query_hard_negatives():
    dataset = datasets.load_dataset("wmt/wmt19", "zh-en", streaming=False)['train']
    for item in dataset:
        pair = item['translation']
        query = Query(turns=[{"user": str(pair)}])
        # print(query)
        response = asyncio.run(query_model_with_backoff("A", query))
        # print(textwrap.fill(str(response), width=150))
        match = re.search(r'(\[.*\])', response.text, flags=re.DOTALL)
        if match:
            match_str = match.group(0).strip()
            answers = json.loads(match_str)
            print(answers)
        else:
            print("Regex not matched")
            print(response.text)
        break

def test_json_extraction():
    text = """
            ```json
            [
              "1929 and 1989?",
              "Between 1929 and 1989?",
              "1929 to 1989?"
            ]
            ```
            """
    match = re.search(r'(\[.*\])', text, flags=re.DOTALL)
    if match:
        match_str = match.group(0).strip()
        answers = json.loads(match_str)
        print(answers)

if __name__ == '__main__':
    load_dotenv()
    # test_preprocess()
    # test_build_graphs()
    # test_graph_encoder()
    # test_data_load()
    # test_query_hard_negatives()
    test_json_extraction()