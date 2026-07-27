from dataclasses import dataclass

import torch
from stanza import Document

langs = ['en', 'zh']
LANG_MAP = {lang: i for i, lang in enumerate(langs)}

upos_list = [
    "<PAD>", "<UNK>", "ADJ", "ADP", "ADV", "AUX",
    "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART",
    "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]
upos_list += ["ENTITY"]  # placeholder for entity nodes
UPOS_MAP = {upos: i for i, upos in enumerate(upos_list, 1)}  # 0 is reserved for padding

# morph_feats = [
#     # TODO - add morphological features
# ]

ontonotes_tags = [
    "O", "CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "MONEY",
    "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART",
]
NER_MAP = {tag: i for i, tag in enumerate(ontonotes_tags, 1)}  # 0 is reserved for padding
# dependencies for English and Chinese
deprels = [
    "acl",  # clausal modifier of noun (adnominal clause)
    "acl:relcl",  # relative clause modifier
    "advcl",  # adverbial clause modifier
    "advcl:relcl",  # adverbial relative clause modifier
    "advmod",  # adverbial modifier
    "advmod:df",
    "amod",  # adjectival modifier
    "appos",  # appositional modifier
    "aux",  # auxiliary
    "aux:pass",  # passive auxiliary
    "case",  # case marking
    "cc",  # coordinating conjunction
    "cc:preconj",  # preconjunct
    "ccomp",  # clausal complement
    "clf",  # classifier
    "compound",  # compound
    "compound:ext",
    "compound:prt",  # phrasal verb particle
    "conj",  # conjunct
    "cop",  # copula
    "csubj",  # clausal subject
    "csubj:outer",  # outer clause clausal subject
    "csubj:pass",  # clausal passive subject
    "dep",  # unspecified dependency
    "det",  # determiner
    "det:predet",
    "discourse",  # discourse element
    "discourse:sp",
    "dislocated",  # dislocated elements
    "expl",  # expletive
    "fixed",  # fixed multiword expression
    "flat",  # flat expression
    "flat:foreign",  # foreign words
    "flat:name",  # names
    "goeswith",  # goes with
    "iobj",  # indirect object
    "list",  # list
    "mark",  # marker
    "mark:adv",
    "mark:rel",
    "nmod",  # nominal modifier
    "nmod:desc",
    "nmod:poss",  # possessive nominal modifier
    "nmod:tmod",  # temporal modifier
    "nmod:unmarked",
    "nsubj",  # nominal subject
    "nsubj:outer",  # outer clause nominal subject
    "nsubj:pass",  # passive nominal subject
    "nummod",  # numeric modifier
    "obj",  # object
    "obl",  # oblique nominal
    "obl:agent",  # oblique agent in passive construction
    "obl:patient",
    "obl:unmarked",
    "orphan",  # orphan
    "parataxis",  # parataxis
    "punct",  # punctuation
    "reparandum",  # overridden disfluency
    "root",  # root
    "vocative",  # vocative
    "xcomp",  # open clausal complement
]
rel_list = deprels + ["PART_OF_ENTITY"]  # represents graph edge between entity and the words that are part of it
rel_list += ["obl:tmod"]  # Was encountered later in training, needed to add without messing up embeddings
REL_MAP = {rel: i for i, rel in enumerate(rel_list, 2)}  # 0 is reserved to indicate no relation
SELF_REL = 1  # 1 is reserved for self loop for attention masking


@dataclass
class Node:
    span: tuple[int, int]
    ner: int
    upos: int
    # srl: str # TODO - SRL is next thing to add to graph


@dataclass
class Edge:
    head: int
    target: int
    relation: int


@dataclass
class Graph:
    nodes: list[Node]
    edges: list[Edge]
    lang: int
    text: str


@torch.no_grad()
def build_graph(doc: Document, lang: str) -> Graph:
    assert len(doc.sentences) == 1
    sent = doc.sentences[0]
    nodes, edges = [], []
    # Add nodes and dependency edges for each word
    # NOTE: This logic only works for languages where MWTs map directly back to source text
    for token in sent.tokens:
        word_start = token.start_char
        for word in token.words:
            ner = word.parent.ner  # TODO - consider adding bios entity features
            base_tag = ner if ner == 'O' else ner[2:]
            nodes.append(Node(
                span=(word_start, word_start + len(word.text)),
                ner=NER_MAP[base_tag],
                upos=UPOS_MAP[word.upos],
            ))
            edges.append(Edge(
                head=word.head,  # 1 based indexing - 0 represents root, which has learned node embedding in model
                target=word.id,
                relation=REL_MAP[word.deprel],
            ))
            word_start += len(word.text)
    # Add entity node and span edges for each entity
    for ent in sent.ents:
        node_id = len(nodes) + 1  # Entity node will be added to end of list
        word_indices = []
        for token in ent.tokens:
            word_start = token.start_char
            for word in token.words:
                word_indices.append(word.id)
                edges.append(Edge(
                    head=node_id,
                    target=word.id,
                    relation=REL_MAP["PART_OF_ENTITY"],
                ))
                word_start += len(word.text)
        nodes.append(Node(
            span=(ent.start_char, ent.end_char),
            ner=NER_MAP[ent.type],
            upos=UPOS_MAP["ENTITY"],
        ))
    return Graph(nodes=nodes, edges=edges, lang=LANG_MAP[lang], text=sent.text)
