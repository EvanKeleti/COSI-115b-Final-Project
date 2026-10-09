"""
Various functions used to figure things out during development process.
"""

import numpy as np
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_processing.data import load_wmt_data
from data_processing.preprocess import StanzaProcessor, Preprocessor

import multiprocessing as mp

langs = ['en', 'zh']

def plot_graph_size_distribution(num_samples: int, batch_size: int) -> None:
    """
    Find and plot the distribution of graph sizes for num_samples sentences for both Chinese and English.
    """
    data = load_wmt_data('train').select(range(num_samples))
    dataloader = iter(DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: {lang: [pair['translation'][lang] for pair in batch] for lang in langs},
    ))

    stop_event = mp.Event()
    nlps = {
        lang: StanzaProcessor(
            lang,
            processors=['tokenize', 'pos', 'lemma', 'depparse', 'ner'],
            use_gpu=True,
            stop_event=stop_event,
        )
        for lang in langs
    }

    try:
        for lang in langs:
            nlps[lang].run_in_thread()
        for batch in dataloader:
            for lang in langs:
                nlps[lang].add_input_to_pipeline(batch[lang])
        node_counts = {lang: [] for lang in langs}
        for docs1, docs2 in tqdm(zip(nlps['en'], nlps['zh']), total=num_samples * batch_size):
            node_counts['en'].extend(doc.num_words + 1 for doc in docs1)
            node_counts['zh'].extend(doc.num_words + 1 for doc in docs2)
    except Exception as e:
        stop_event.set()
        raise e

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for lang, ax in zip(langs, axs):
        ax.hist(node_counts[lang])
        ax.set_title(f"{lang} Distribution")
        ax.set_xlabel('Node Count')
        ax.set_ylabel('Num Sentences')

    plt.tight_layout()
    plt.savefig("plots/node_count_distribution.png")
    plt.close()

    for lang in langs:
        print(f"{lang} Distribution")
        print("95th percentile = ", np.percentile(node_counts[lang], 95))
        print("99th percentile = ", np.percentile(node_counts[lang], 99))


def preprocess_throughput_experiment(batch_size: int, num_samples: int = 1000) -> None:
    """
    Function to experiment with batch size for preprocessing for max throughput.
    """
    langs = ['en', 'zh']
    data = load_wmt_data('train').select(range(num_samples))

    preprocessor = Preprocessor(
        data,
        "xlm-roberta-base",
        langs,
        batch_size=batch_size,
        output_batch_size=8,
        show_pbar=True,
        separate_thread=False,
        multithread=True,
        use_gpu=True,
        in_stages=False,
        verbose=True,
    )

    try:
        for batch in preprocessor:
            pass
    finally:
        preprocessor.terminate_processors()


def stanza_throughput_experiment(batch_size: int, num_samples = 1000) -> None:
    data = load_wmt_data('train').select(range(num_samples))
    dataloader = iter(DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: {lang: [pair['translation'][lang] for pair in batch] for lang in langs},
    ))

    stop_event = mp.Event()
    nlps = {
        lang: StanzaProcessor(
            lang,
            processors=['tokenize', 'pos', 'lemma', 'depparse', 'ner'],
            use_gpu=True,
            stop_event=stop_event,
        )
        for lang in langs
    }

    try:
        for batch in dataloader:
            for lang in langs:
                nlps[lang].add_input_to_pipeline(batch[lang])
        for lang in langs:
            nlps[lang].run_in_thread()
        with tqdm(position=0, leave=True, total=num_samples, desc=f"Processing {num_samples} sentences",
             smoothing=0) as pbar:
            for docs1, docs2 in zip(*nlps.values()):
                pbar.update(len(docs1))
    except Exception as e:
        stop_event.set()
        raise e

def main():
    # plot_graph_size_distribution(5000, 64)
    preprocess_throughput_experiment(8, 10000)
    # stanza_throughput_experiment(16, 10000)

if __name__ == '__main__':
    load_dotenv()
    main()
