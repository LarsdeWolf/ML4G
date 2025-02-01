import itertools
import csv
import os
import pandas as pd
import spacy, coreferee
import numpy as np
import re
import tensorflow as tf
import tensorflow_hub as hub
from collections import defaultdict
from tqdm import tqdm
from spacy.tokenizer import Tokenizer
from spacy.matcher import PhraseMatcher
from rapidfuzz import fuzz


def entity_graph(point, pipe, embedder, ent_count, rel_count, extension=True):
    """
    Generates the entity graph for a given datapoint
    Parameters
    ----------
    point: datapoint (pd.DataFrame)
    pipe: spacy pipeline (spacy.lang.en.English)
    embedder: elmo embedder (tf.Module)
    ent_count: #entities in entity out file (int)
    rel_count: #relations in relation out file (int)
    extension: Whether to include fuzzy and cosine matching (bool)

    Returns
    -------
    MENTIONS: Dict of mentions (Dict[int, List[str, int, int, int]])
    NQUADS: Dict of nquads (Dict[int, List[int, str, int, str]])
    EMBEDDINGS: Dict of embeddings (Dict[int, np.Array[float]])
    q_embeddings: Query embeddings (np.Array[float])
    """
    query, subject, supports, criteria = preprocess_point(point, pipe)

    # Matcher for searching exact occurences of criteria
    matcher = PhraseMatcher(pipe.vocab)
    [[matcher.add(crit, [pipe.make_doc(crit + x)]) for x in ['', '.', ',']] for crit in criteria]

    # # EMBEDDINGS
    embeddings = get_embeddings(embedder, [doc.text for doc in supports])
    q_embeddings = get_embeddings(embedder, [query]).mean(axis=1)

    MENTIONS = defaultdict(list)
    NQUADS = defaultdict(list)
    EMBEDDINGS = defaultdict(list)
    with_edges = set()
    for i, doc in enumerate(supports):

        # Getting exact matches
        exact_matches = [((start, end), m) for m, start, end in matcher(doc)]
        doc_mention_i = len(MENTIONS)
        matches = set([])

        # Adding exact match nodes/embeddings
        for (mstart, mend), m in exact_matches:
            span = matcher.vocab.strings[m]
            match_idx = len(MENTIONS) + ent_count
            MENTIONS[match_idx] = [span, mstart, mend, i]
            EMBEDDINGS[match_idx] = embeddings[i, mstart:mend, :].mean(axis=0)
            matches.add((span, (mstart, mend), match_idx))

            # Adding coref nodes
            coref_matches = coref((mstart, mend), doc._.coref_chains)
            for costart, coend in coref_matches:
                comatch_idx = len(MENTIONS) + ent_count
                MENTIONS[comatch_idx] = [span, costart, coend, i]
                EMBEDDINGS[comatch_idx] = embeddings[i, costart:coend, :].mean(axis=0)
                matches.add((span, (costart, coend), comatch_idx))

                # Adding COREF Edges
                NQUADS[len(NQUADS) + rel_count] = [match_idx, 'COREF', comatch_idx, point.name]
                with_edges.add((match_idx, comatch_idx))

        # Extension including fuzzy matching
        if extension:
            # Only matching on non-found criteria candidates
            not_found = set(criteria) - set([x[0] for x in matches])
            fuzzy_matches = fuzzy_match(doc, not_found)
            for ent, (start, end), _ in fuzzy_matches:
                match_idx = len(MENTIONS) + ent_count
                MENTIONS[match_idx] = [ent, start, end, i]
                EMBEDDINGS[match_idx] = embeddings[i, start:end, :].mean(axis=0)
                matches.add((ent, (start, end), match_idx))

        # Adding DOC Edges
        doc_mention_ids = np.array(list(MENTIONS.keys())[doc_mention_i:len(MENTIONS)])
        for subject, object in itertools.combinations(doc_mention_ids, 2):
            NQUADS[len(NQUADS) + rel_count] = [subject, 'DOC', object, point.name]
            with_edges.add((subject, object))

    mentions_keys = np.array(list(MENTIONS.keys()))
    mentions_entities = np.array([a[0] for a in MENTIONS.values()])
    if extension:
        # Adding cosine similarity edges between nodes
        for subject, object in [x for x in itertools.combinations(mentions_keys, 2)]:
            A, B = EMBEDDINGS[subject], EMBEDDINGS[object]
            score = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
            if score > 0.8:
                NQUADS[len(NQUADS) + rel_count] = [subject, 'SEMANTIC', object, point.name]
                with_edges.add((subject, object))

    # Adding MATCH Edges
    for elem in set(mentions_entities):
        indexes = np.nonzero(mentions_entities == elem)[0]
        for subject, object in itertools.combinations(indexes, 2):
            NQUADS[len(NQUADS) + rel_count] = [mentions_keys[subject], 'MATCH', mentions_keys[object], point.name]
            with_edges.add((mentions_keys[subject], mentions_keys[object]))

    # Adding complement edges
    for subject, object in set([x for x in itertools.combinations(set(mentions_keys), 2)]) - with_edges:
        NQUADS[len(NQUADS) + rel_count] = [subject, 'COMPLEMENT', object, point.name]

    return MENTIONS, NQUADS, EMBEDDINGS, q_embeddings


def preprocess_point(point, pipe):
    """
    Preprocesses a datapoint by extracting the query, subject, supports, and criteria
    Parameters
    ----------
    point: datapoint (pd.DataFrame)
    pipe: spacy pipeline (spacy.lang.en.English)

    Returns
    -------
    query: Query of the datapoint (str)
    subject: Subject of the datapoint (str)
    supports: Annotated spacy support documents (List[spacy.tokens.doc.Doc])
    criteria: Criteria of the datapoint (List[str])
    """
    query = point['query'].strip().split(' ')
    subject = ' '.join(query[1:])
    query = ' '.join(query[0].split('_') + query[1:])
    supports = [pipe(' '.join(s.split()).lower()) for s in point['supports']]
    criteria = [subject] + point['candidates']  # criteria: Candidates U {subject}
    return query, subject, supports, criteria


def fuzzy_match(doc, criteria, N=2, treshold=0.75):
    """
    Fuzzy matching where we use a sliding window approach to match fuzzy candidates with windows of the doc
    Extracts N-best matches for each criteria
    Parameters
    ----------
    doc: spacy doc (spacy.tokens.doc.Doc)
    criteria: List of criteria to match (List[str])
    N: Number of best matches to extract for each criteria (int)
    treshold: Fuzzy matching treshold (float)

    Returns
    -------
    matches: List of N-best matches for each criteria (List[Tuple[str, Tuple[int, int], float]])
    """
    matches = []
    for crit in criteria:
        crit_matches = []
        for size in range(len(crit.split()) - 1, len(crit.split()) + 2):
            for i in range(len(doc) - size):
                span = doc[i:i + size]
                score = fuzz.ratio(span.text.lower(), crit.lower()) / 100
                if score > treshold and score != 1:
                    crit_matches.append((crit, (span.start, span.end), score))
        crit_matches.sort(key=lambda x: x[-1], reverse=True)
        matches.extend(crit_matches[:N])
    return matches


def coref(match, chains):
    """
    Extracts coref matches for a given match
    Parameters
    ----------
    match: Match to extract coref matches for (Tuple[int, int])
    chains: Coref chains (List[List[spacy.tokens.token.Token]])

    Returns
    -------
    corefmatches: List of coref matches (List[Tuple[int, int]])
    """
    corefmatches = []
    for chain in chains:
        for token in chain:
            if token.root_index in list(range(match[0], match[1])):
                refs = list(chain)
                refs.remove(token)
                corefmatches.append([(x.root_index, x.root_index + 1) for x in refs])
    if len(corefmatches) == 1:
        return corefmatches[0]
    return []


def get_embeddings(el, docs):
    """
    Get embeddings for a list of documents
    Parameters
    ----------
    el: elmo embedder (tf.Module)
    docs: List of documents (List[str])

    Returns
    -------
    embeddings: Embeddings for the documents (np.Array[float])
    """
    out = el(tf.constant(docs))
    elmo, lstm1, lstm2 = out['elmo'], out['lstm_outputs1'], out['lstm_outputs2']
    return tf.concat([elmo, lstm1, lstm2], axis=-1).numpy()


def process_row(row, pipe, embedder, ent_path, rel_path, emb_path, qemb_path):
    """
    Processes a row and writes the results to the respective files
    Parameters
    ----------
    row: Row to process (pd.DataFrame)
    pipe: spacy pipeline (spacy.lang.en.English)
    embedder: elmo embedder (tf.Module)
    ent_path: Path to entity file (str)
    rel_path: Path to relation file (str)
    emb_path: Path to embeddings file (str)
    qemb_path: Path to query embeddings file (str)

    Returns
    -------

    """
    global NUM_ENTS, NUM_NQUADS

    mentions, nquads, embeddings, q_emb = entity_graph(row, pipe, embedder, NUM_ENTS, NUM_NQUADS)
    with (open(ent_path, 'a', newline='') as file):
        writer = csv.writer(file)
        if NUM_ENTS == 0:
            writer.writerow(['id', 'entity', 'start_i', 'end_i', 'supports_i'])
        writer.writerows([[a] + b for a, b in mentions.items()])

    with (open(rel_path, 'a', newline='') as file):
        writer = csv.writer(file)
        if NUM_NQUADS == 0:
            writer.writerow(['id', 'object', 'relation', 'subject', 'datapoint_i'])
        writer.writerows([[a] + b for a, b in nquads.items()])

    with (open(emb_path, 'a', newline='') as file):
        writer = csv.writer(file)
        writer.writerows([b for _, b in embeddings.items()])

    with (open(qemb_path, 'a', newline='') as file):
        writer = csv.writer(file)
        writer.writerows(q_emb)

    NUM_ENTS += len(mentions)
    NUM_NQUADS += len(nquads)


if __name__ == '__main__':
    # Loading Spacy modules and ELMo Embedder
    PIPE = spacy.load('en_core_web_trf')
    PIPE.add_pipe('coreferee')
    PIPE.tokenizer = Tokenizer(PIPE.vocab, token_match=re.compile(r'\S+').match)
    ELMO = hub.load("https://tfhub.dev/google/elmo/3")
    ELMO = ELMO.signatures["default"]

    PATH = 'wikihop/'
    for SPLIT in ['train', 'dev']:
        DATA = pd.read_json(f'{PATH}{SPLIT}.json')
        for EXTENSION in ['normal', 'extended']:
            ENT_PATH = f'{PATH}{EXTENSION}/{SPLIT}_ents.csv'
            REL_PATH = f'{PATH}{EXTENSION}/{SPLIT}_nquads.csv'
            EMB_PATH = f'{PATH}{EXTENSION}/{SPLIT}_embeds.csv'
            QEMB_PATH = f'{PATH}{EXTENSION}/{SPLIT}_qembeds.csv'

            if SPLIT == 'train':
                N_POINTS = 500
            else:
                N_POINTS = 50

            tqdm.pandas(desc='Generating Graphs...')
            NUM_ENTS, NUM_NQUADS = 0, 0
            assert not any([os.path.isfile(path) for path in [ENT_PATH,
                                                              REL_PATH,
                                                              EMB_PATH,
                                                              QEMB_PATH]]), "CSV Files already exist"
            DATA.iloc[:N_POINTS].progress_apply(
                lambda row: process_row(row, PIPE, ELMO, ENT_PATH, REL_PATH, EMB_PATH, QEMB_PATH), axis=1)
