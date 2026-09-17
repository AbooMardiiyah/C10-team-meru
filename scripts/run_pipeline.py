#!/usr/bin/env python3
"""
Team Meru - AI6 | 1st Place Solution
Agricultural Extension RAG: Smart Retrieval for Farmers

End-to-end pipeline: data loading -> candidate retrieval -> CE reranking -> submission.csv

Usage:
    python scripts/run_pipeline.py \
        --data-dir data/ \
        --biencoder-path models/hnm_final_r2 \
        --output submission.csv \
        --n-folds 5

Requires:
    - Competition data CSVs in --data-dir
    - Fine-tuned bi-encoder (e5-base-v2 HNM R2) at --biencoder-path
    - GPU with ~4GB VRAM (ettin-68m is lightweight)
    - NLTK data: python -m nltk.downloader punkt_tab stopwords
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import (
    BM25BasicRetriever,
    BM25StemmedRetriever,
    DenseRetriever,
    build_corpus,
    ce_ensemble_score,
    cross_topic_folds,
    discover_topics,
    evaluate,
    load_data,
    rrf_fuse,
    select_top_k,
    train_ce_ensemble,
    triple_bm25_candidates,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Team Meru winning pipeline for Agricultural Extension RAG"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Directory containing competition CSV files",
    )
    parser.add_argument(
        "--biencoder-path", type=str, required=True,
        help="Path to fine-tuned bi-encoder model (hnm_final_r2)",
    )
    parser.add_argument(
        "--output", type=str, default="submission.csv",
        help="Output submission CSV path (default: submission.csv)",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of cross-topic CE folds (default: 5, competition winner)",
    )
    parser.add_argument(
        "--ce-model", type=str, default="cross-encoder/ettin-reranker-68m-v1",
        help="Cross-encoder model name (default: ettin-reranker-68m-v1)",
    )
    parser.add_argument(
        "--top-k-candidates", type=int, default=100,
        help="Number of candidates per retriever before fusion (default: 100)",
    )
    parser.add_argument(
        "--ce-epochs", type=int, default=2,
        help="Cross-encoder training epochs (default: 2)",
    )
    parser.add_argument(
        "--ce-batch-size", type=int, default=8,
        help="Cross-encoder training batch size (default: 8)",
    )
    parser.add_argument(
        "--hn-top-k", type=int, default=30,
        help="Number of hard negatives per query for CE training (default: 30)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for computation (default: auto-detect)",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Run on train set only and report validation nDCG@5",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"torch {torch.__version__}, CUDA: {torch.cuda.is_available()}")

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading data...")
    docs, train_q, test_q, qrels = load_data(args.data_dir)
    doc_ids = docs["document_id"].tolist()
    corpus = build_corpus(docs)
    doc_map = dict(zip(doc_ids, corpus))
    query_map = dict(zip(train_q["query_id"], train_q["query"]))
    test_query_map = dict(zip(test_q["query_id"], test_q["query"]))
    all_train_qids = set(train_q["query_id"])

    print(f"  Documents: {len(docs)}")
    print(f"  Train queries: {len(train_q)}")
    print(f"  Test queries: {len(test_q)}")
    print(f"  Relevance judgments: {len(qrels)}")

    # ------------------------------------------------------------------
    # Step 2: Build retrievers
    # ------------------------------------------------------------------
    print("\n[2/5] Building retrievers...")
    t0 = time.time()

    print("  Loading bi-encoder...")
    dense = DenseRetriever(corpus, args.biencoder_path, device=device)

    print("  Building BM25 (basic)...")
    bm25_basic = BM25BasicRetriever(corpus)

    print("  Building BM25 (stemmed)...")
    bm25_stemmed = BM25StemmedRetriever(corpus)

    print(f"  Retrievers built in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 3: Generate candidates
    # ------------------------------------------------------------------
    print("\n[3/5] Generating candidates...")
    top_k = args.top_k_candidates

    # Train candidates (dense only, used for CE hard negative mining)
    print("  Train candidates (dense)...")
    train_cands = {}
    for qid in train_q["query_id"]:
        idx = dense.retrieve(query_map[qid], top_k)
        train_cands[qid] = [doc_ids[i] for i in idx]

    # Test candidates (triple BM25 = dense + BM25-basic + BM25-stemmed + RRF)
    if not args.eval_only:
        print("  Test candidates (triple BM25)...")
        test_cands = {}
        for qid in test_q["query_id"]:
            test_cands[qid] = triple_bm25_candidates(
                test_query_map[qid], doc_ids, dense, bm25_basic, bm25_stemmed,
                top_k=top_k,
            )

    # Validation candidates (triple BM25)
    # Use 80/20 topic split for validation
    topics = discover_topics(qrels)
    rng = np.random.RandomState(42)
    tids = sorted(topics.keys())
    rng.shuffle(tids)
    n_val = max(1, len(tids) // 5)
    val_qids = set()
    for tid in tids[:n_val]:
        val_qids.update(topics[tid])

    print(f"  Validation queries: {len(val_qids)} (from {n_val} held-out topics)")
    print("  Val candidates (triple BM25)...")
    val_cands = {}
    for qid in val_qids:
        if qid in query_map:
            val_cands[qid] = triple_bm25_candidates(
                query_map[qid], doc_ids, dense, bm25_basic, bm25_stemmed,
                top_k=top_k,
            )

    # Free bi-encoder GPU memory before CE training
    del dense
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 4: Train cross-encoder ensemble
    # ------------------------------------------------------------------
    print(f"\n[4/5] Training {args.n_folds}-fold CE ensemble...")
    t0 = time.time()

    folds = cross_topic_folds(topics, args.n_folds)
    print(f"  Fold sizes: {[len(f) for f in folds]}")

    ces = train_ce_ensemble(
        qrels=qrels,
        folds=folds,
        all_train_qids=all_train_qids,
        candidates=train_cands,
        query_map=query_map,
        doc_map=doc_map,
        model_name=args.ce_model,
        epochs=args.ce_epochs,
        batch_size=args.ce_batch_size,
        hn_top_k=args.hn_top_k,
        device=device,
    )
    print(f"  {len(ces)} CEs trained in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 5: Score and rank
    # ------------------------------------------------------------------
    print("\n[5/5] Scoring and ranking...")

    # Validation evaluation
    print("  Scoring validation set...")
    val_rankings = {}
    for qid in sorted(val_qids):
        if qid not in val_cands:
            continue
        scores = ce_ensemble_score(ces, query_map[qid], val_cands[qid], doc_map)
        val_rankings[qid] = select_top_k(scores, k=5)

    val_ndcg = evaluate(val_rankings, qrels, k=5)
    print(f"  Validation nDCG@5: {val_ndcg:.4f} ({len(val_rankings)} queries)")

    if args.eval_only:
        print("\n--eval-only mode: skipping test submission.")
        return

    # Test scoring and submission
    print("  Scoring test set...")
    rows = []
    for qid in test_q["query_id"]:
        scores = ce_ensemble_score(ces, test_query_map[qid], test_cands[qid], doc_map)
        top5 = select_top_k(scores, k=5)
        for did in top5:
            rows.append({"QueryId": qid, "DocumentId": did})

    submission = pd.DataFrame(rows)
    submission.to_csv(args.output, index=False)
    print(f"\n  Submission saved: {args.output}")
    print(f"  Rows: {len(submission)}, Queries: {submission['QueryId'].nunique()}")
    print(f"  Validation nDCG@5: {val_ndcg:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
