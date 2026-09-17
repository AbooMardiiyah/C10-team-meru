"""
Team Meru - AI6 | 1st Place Solution
Agricultural Extension RAG: Smart Retrieval for Farmers

Pipeline module with functions for:
  - Dense retrieval (fine-tuned e5-base-v2)
  - BM25 retrieval (basic + stemmed)
  - Reciprocal Rank Fusion (RRF)
  - Triple BM25 candidate generation
  - Cross-encoder ensemble training and scoring
  - nDCG@5 evaluation
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

# Token pattern used across retrievers
TOKEN_PAT = r"(?u)\b\w[\w\-]+\b"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load competition CSV files.

    Returns:
        (documents, train_queries, test_queries, qrels) DataFrames
    """
    docs = pd.read_csv(f"{data_dir}/documents.csv", dtype={"document_id": str})
    train_q = pd.read_csv(f"{data_dir}/train_queries.csv", dtype={"query_id": str})
    test_q = pd.read_csv(f"{data_dir}/test_queries.csv", dtype={"query_id": str})
    qrels = pd.read_csv(f"{data_dir}/qrels_train.csv", dtype={"query_id": str, "document_id": str})
    return docs, train_q, test_q, qrels


def build_corpus(docs: pd.DataFrame) -> list[str]:
    """Concatenate title + text for each document."""
    return (docs["title"].fillna("") + ". " + docs["text"].fillna("")).tolist()


# ---------------------------------------------------------------------------
# Topic discovery (for cross-topic fold splitting)
# ---------------------------------------------------------------------------

def discover_topics(qrels: pd.DataFrame) -> dict[int, list[str]]:
    """Group query IDs into topics based on shared relevant documents.

    Two queries belong to the same topic if they share any relevant document
    (rel >= 2). Uses union-find to merge transitively connected queries.

    Returns:
        dict mapping topic_id -> list of query_ids
    """
    qrels_grouped = {
        qid: dict(zip(g["document_id"], g["relevance"]))
        for qid, g in qrels.groupby("query_id")
    }

    # Find positive docs per query
    pos_docs = {}
    for qid, gains in qrels_grouped.items():
        pos_docs[qid] = frozenset(d for d, r in gains.items() if r >= 2)

    # Union-find via shared documents
    qid_to_topic: dict[str, int] = {}
    doc_to_topic: dict[str, int] = {}
    topic_id = 0

    for qid, pos in pos_docs.items():
        existing = set(doc_to_topic[d] for d in pos if d in doc_to_topic)
        if existing:
            merge_to = min(existing)
            qid_to_topic[qid] = merge_to
            for d in pos:
                doc_to_topic[d] = merge_to
            for tid in existing - {merge_to}:
                for q, t in list(qid_to_topic.items()):
                    if t == tid:
                        qid_to_topic[q] = merge_to
                for d, t in list(doc_to_topic.items()):
                    if t == tid:
                        doc_to_topic[d] = merge_to
        else:
            qid_to_topic[qid] = topic_id
            for d in pos:
                doc_to_topic[d] = topic_id
            topic_id += 1

    topics = defaultdict(list)
    for qid, tid in qid_to_topic.items():
        topics[tid].append(qid)
    return dict(topics)


def cross_topic_folds(topics: dict[int, list[str]], n_folds: int,
                      seed: int = 456) -> list[set[str]]:
    """Split topics into n_folds groups of query IDs.

    Each fold contains the query IDs of its assigned topics.
    Used as hold-out sets for cross-topic CE ensemble training.

    Returns:
        List of n_folds sets, where each set contains query IDs for that fold.
    """
    rng = np.random.RandomState(seed)
    tids = sorted(topics.keys())
    rng.shuffle(tids)

    folds = []
    fold_size = len(tids) // n_folds
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(tids)
        fold_qids = set()
        for tid in tids[start:end]:
            fold_qids.update(topics[tid])
        folds.append(fold_qids)
    return folds


# ---------------------------------------------------------------------------
# Dense retrieval (bi-encoder)
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Dense retrieval using a fine-tuned sentence-transformer bi-encoder.

    The bi-encoder (e5-base-v2 with HNM round 2) encodes all documents once,
    then scores queries by cosine similarity against the precomputed embeddings.
    """

    def __init__(self, corpus: list[str], model_path: str, device: str = "cuda"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_path, device=device)
        # e5-v2 uses "passage: " prefix for documents
        self.doc_emb = self.model.encode(
            ["passage: " + doc for doc in corpus],
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=True,
        )

    def retrieve(self, query: str, top_k: int = 100) -> list[int]:
        """Return top-k document indices for a query.

        Uses "query: " prefix as required by e5-v2.
        """
        q_emb = self.model.encode(
            ["query: " + query], normalize_embeddings=True
        )
        scores = (self.doc_emb @ q_emb.T).ravel()
        return np.argsort(-scores)[:top_k].tolist()


# ---------------------------------------------------------------------------
# BM25 retrieval (basic + stemmed)
# ---------------------------------------------------------------------------

class BM25BasicRetriever:
    """BM25 with simple whitespace tokenization (lowercased)."""

    def __init__(self, corpus: list[str]):
        tokenized = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 100) -> list[int]:
        """Return top-k document indices."""
        scores = self.bm25.get_scores(query.lower().split())
        return np.argsort(-scores)[:top_k].tolist()


class BM25StemmedRetriever:
    """BM25 with Porter stemming and stopword removal.

    Captures morphological variants (e.g., 'farming' -> 'farm') that
    basic tokenization misses.
    """

    def __init__(self, corpus: list[str]):
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer

        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))
        tokenized = [self._tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"\w+", text.lower())
        return [
            self.stemmer.stem(w)
            for w in tokens
            if w not in self.stop_words and len(w) > 1
        ]

    def retrieve(self, query: str, top_k: int = 100) -> list[int]:
        """Return top-k document indices."""
        scores = self.bm25.get_scores(self._tokenize(query))
        return np.argsort(-scores)[:top_k].tolist()


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def rrf_fuse(ranked_lists: list[list], k: int = 60) -> list:
    """Fuse multiple ranked lists via Reciprocal Rank Fusion.

    RRF score for doc d = sum over lists of 1 / (k + rank(d) + 1).
    Documents appearing in more lists and at higher ranks get higher scores.

    Args:
        ranked_lists: List of ranked lists (each is a list of doc identifiers).
        k: RRF constant (default 60, standard value from Cormack et al.).

    Returns:
        Fused ranked list of doc identifiers, sorted by descending RRF score.
    """
    scores: dict = {}
    for cands in ranked_lists:
        for pos, doc in enumerate(cands):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + pos + 1)
    return sorted(scores.keys(), key=lambda d: -scores[d])


# ---------------------------------------------------------------------------
# Triple BM25 candidate generation
# ---------------------------------------------------------------------------

def triple_bm25_candidates(
    query: str,
    doc_ids: list[str],
    dense_retriever: DenseRetriever,
    bm25_basic: BM25BasicRetriever,
    bm25_stemmed: BM25StemmedRetriever,
    top_k: int = 100,
    rrf_k: int = 60,
) -> list[str]:
    """Generate candidates from 3 retrieval signals fused via RRF.

    1. Dense retrieval (fine-tuned bi-encoder)
    2. BM25 with basic tokenization
    3. BM25 with stemmed tokenization

    Returns top_k document IDs after RRF fusion.
    """
    dense_idx = dense_retriever.retrieve(query, top_k)
    bm25b_idx = bm25_basic.retrieve(query, top_k)
    bm25s_idx = bm25_stemmed.retrieve(query, top_k)

    # Convert indices to doc IDs
    dense_ids = [doc_ids[i] for i in dense_idx]
    bm25b_ids = [doc_ids[i] for i in bm25b_idx]
    bm25s_ids = [doc_ids[i] for i in bm25s_idx]

    fused = rrf_fuse([dense_ids, bm25b_ids, bm25s_ids], k=rrf_k)
    return fused[:top_k]


# ---------------------------------------------------------------------------
# Cross-encoder training and scoring
# ---------------------------------------------------------------------------

def build_ce_training_examples(
    qrels: pd.DataFrame,
    train_qids: set[str],
    query_map: dict[str, str],
    doc_map: dict[str, str],
) -> list:
    """Build cross-encoder training examples from relevance judgments.

    Relevance grades (0-3) are normalized to [0, 1] as regression targets.

    Returns:
        List of InputExample objects.
    """
    from sentence_transformers import InputExample

    examples = []
    for _, row in qrels.iterrows():
        qid = row["query_id"]
        if qid not in train_qids:
            continue
        query = query_map.get(qid)
        doc = doc_map.get(row["document_id"])
        if query and doc:
            label = float(row["relevance"]) / 3.0
            examples.append(InputExample(texts=[query, doc], label=label))
    return examples


def enrich_with_hard_negatives(
    examples: list,
    qids: set[str],
    candidates: dict[str, list[str]],
    qrels_grouped: dict[str, dict[str, float]],
    query_map: dict[str, str],
    doc_map: dict[str, str],
    top_k: int = 30,
) -> list:
    """Add hard negative examples from retriever candidates.

    For each query, take the top-k retrieved docs that have no relevance
    judgment and add them as label=0.0 training pairs. These are hard
    negatives because the bi-encoder ranked them highly but they are
    not actually relevant.

    Returns:
        Extended list of InputExample objects.
    """
    from sentence_transformers import InputExample

    enriched = list(examples)
    for qid in qids:
        if qid not in query_map or qid not in candidates:
            continue
        judged = set(qrels_grouped.get(qid, {}).keys())
        for did in candidates[qid][:top_k]:
            if did not in judged and did in doc_map:
                enriched.append(
                    InputExample(texts=[query_map[qid], doc_map[did]], label=0.0)
                )
    return enriched


def train_cross_encoder(
    examples: list,
    model_name: str = "cross-encoder/ettin-reranker-68m-v1",
    epochs: int = 2,
    batch_size: int = 8,
    warmup_steps: int = 100,
    seed: int = 42,
    device: str = "cuda",
):
    """Train a cross-encoder on the given examples.

    Args:
        examples: List of InputExample with (query, doc) pairs and labels.
        model_name: HuggingFace model identifier.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        warmup_steps: Linear warmup steps for learning rate.
        seed: Random seed for reproducibility.
        device: Training device.

    Returns:
        Trained CrossEncoder model.
    """
    import torch
    from sentence_transformers import CrossEncoder
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)

    ce = CrossEncoder(model_name, num_labels=1, device=device)
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size, num_workers=0)
    ce.fit(
        train_dataloader=loader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )
    return ce


def train_ce_ensemble(
    qrels: pd.DataFrame,
    folds: list[set[str]],
    all_train_qids: set[str],
    candidates: dict[str, list[str]],
    query_map: dict[str, str],
    doc_map: dict[str, str],
    model_name: str = "cross-encoder/ettin-reranker-68m-v1",
    epochs: int = 2,
    batch_size: int = 8,
    warmup_steps: int = 100,
    hn_top_k: int = 30,
    device: str = "cuda",
) -> list:
    """Train a cross-topic CE ensemble.

    For each fold, hold out that fold's queries and train on the rest.
    Each CE sees a different subset of topics, providing diversity.

    Returns:
        List of trained CrossEncoder models.
    """
    qrels_grouped = {
        qid: dict(zip(g["document_id"], g["relevance"]))
        for qid, g in qrels.groupby("query_id")
    }

    ces = []
    for i, fold_holdout in enumerate(folds):
        fold_train_qids = all_train_qids - fold_holdout
        examples = build_ce_training_examples(qrels, fold_train_qids, query_map, doc_map)
        examples = enrich_with_hard_negatives(
            examples, fold_train_qids, candidates, qrels_grouped,
            query_map, doc_map, top_k=hn_top_k,
        )
        print(f"  Fold {i}: {len(fold_train_qids)} queries, {len(examples)} examples")
        ce = train_cross_encoder(
            examples,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            warmup_steps=warmup_steps,
            seed=42 + i,
            device=device,
        )
        ces.append(ce)
    return ces


def ce_ensemble_score(
    ces: list,
    query: str,
    candidate_ids: list[str],
    doc_map: dict[str, str],
    batch_size: int = 64,
) -> dict[str, float]:
    """Score candidates by averaging predictions from multiple cross-encoders.

    Returns:
        dict mapping document_id -> averaged CE score.
    """
    pairs = [[query, doc_map[did]] for did in candidate_ids]
    avg_scores = np.zeros(len(candidate_ids))
    for ce in ces:
        preds = np.asarray(ce.predict(pairs, batch_size=batch_size, show_progress_bar=False))
        avg_scores += preds
    avg_scores /= len(ces)
    return dict(zip(candidate_ids, avg_scores.tolist()))


def select_top_k(scores: dict[str, float], k: int = 5) -> list[str]:
    """Select top-k document IDs by descending score."""
    return sorted(scores.keys(), key=lambda d: -scores[d])[:k]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def ndcg_at_k(ranked_ids: list[str], gains: dict[str, float], k: int = 5) -> float:
    """Compute nDCG@k for a single query.

    Uses exponential gain: gain(rel) = 2^rel - 1
    Relevance grades: 3 (highly relevant), 2 (relevant), 1 (marginally), 0 (not).

    Args:
        ranked_ids: Ranked list of document IDs (best first).
        gains: dict mapping document_id -> relevance grade.
        k: Cutoff position.

    Returns:
        nDCG@k score in [0, 1].
    """
    dcg = 0.0
    for i, did in enumerate(ranked_ids[:k]):
        rel = gains.get(did, 0.0)
        dcg += (2.0 ** rel - 1.0) / np.log2(i + 2)

    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum((2.0 ** rel - 1.0) / np.log2(i + 2) for i, rel in enumerate(ideal))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate(
    rankings: dict[str, list[str]],
    qrels: pd.DataFrame,
    k: int = 5,
) -> float:
    """Compute mean nDCG@k over all queries in qrels.

    Args:
        rankings: dict mapping query_id -> ranked list of document_ids.
        qrels: DataFrame with columns (query_id, document_id, relevance).
        k: Cutoff position.

    Returns:
        Mean nDCG@k across all queries.
    """
    qrels_grouped = {
        qid: dict(zip(g["document_id"], g["relevance"]))
        for qid, g in qrels.groupby("query_id")
    }
    scores = []
    for qid, gains in qrels_grouped.items():
        ranked = rankings.get(qid, [])
        scores.append(ndcg_at_k(ranked, gains, k))
    return float(np.mean(scores))
