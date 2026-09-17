# Agricultural Extension RAG: Smart Retrieval for Farmers

**1st Place Solution** by **Team Meru - AI6** (AI Saturdays Lagos, Cohort 10)

**Private LB: 0.95478** | Public LB: 0.96835 | Baseline TF-IDF: ~0.55

Competition: [Agricultural Extension RAG: Smart Retrieval for Farmers](https://www.kaggle.com/competitions/agricultural-extension-rag-smart-retrieval-for-farmers)

---

## Problem

Given a farmer's question, retrieve and rank the top-5 most relevant documents from a corpus of 695 agricultural extension factsheets. Evaluation metric is nDCG@5 with graded relevance (rel=3, 2, 1, 0). Train and test topics are **completely disjoint** -- the model must generalize to unseen agricultural topics.

## Dataset

| File | Description |
|---|---|
| `documents.csv` | 695 agricultural factsheets (title + text) |
| `train_queries.csv` | 308 training queries across ~122 topics |
| `test_queries.csv` | 200 test queries on disjoint topics |
| `qrels_train.csv` | Graded relevance judgments (0-3) |

Critical property: **train and test topics share zero overlap.** This means memorizing topic-specific patterns fails -- the system must learn genuine semantic matching.

## Approach

Our pipeline has three stages: dense+sparse retrieval, fusion, and cross-encoder reranking.

```
                    +-----------------+
   query ---------> | e5-base-v2      |----> dense top-100
                    | (HNM round 2)   |
                    +-----------------+
                                              |
   query ---------> BM25-basic  -----------> top-100  --+
                                              |          |
   query ---------> BM25-stemmed ----------> top-100  --+-- RRF (k=60) --> ~100 candidates
                                                         |
                                              +----------+
                                              |
                                    5x ettin-68m CE
                                    (cross-topic folds)
                                              |
                                       avg CE scores
                                              |
                                         top-5 output
```

### Stage 1: Bi-Encoder Retrieval

Fine-tuned `intfloat/e5-base-v2` with 2 rounds of hard negative mining (MNRL loss). The resulting `hnm_final_r2` model produces dense embeddings for all 695 documents.

### Stage 2: Triple BM25 Hybrid Candidates

Three independent candidate lists, each top-100:
- **Dense** retrieval from the fine-tuned bi-encoder
- **BM25-basic** with simple whitespace tokenization
- **BM25-stemmed** with Porter stemming + stopword removal

Fused via Reciprocal Rank Fusion (k=60) into ~100 candidates per query. This achieved **100% recall of all rel-3 documents** in the top-100.

### Stage 3: Cross-Encoder Ensemble Reranking

5 `cross-encoder/ettin-reranker-68m-v1` models, each trained on a different 80% of topics (cross-topic fold split). Each model is enriched with hard negatives: the top-30 unjudged documents from bi-encoder retrieval are added as label=0 training pairs.

At inference, all 5 CE scores are averaged per document, and the top-5 are selected.

## Results

| Version | Public LB | Description |
|---|---|---|
| v1 | 0.896 | Dense retrieval only |
| v2 | 0.916 | + MiniLM cross-encoder |
| v3 | 0.922 | + hard negative mining (round 2) |
| v5 | 0.940 | + ettin-68m CE (replaced MiniLM) |
| v8 | 0.951 | + BM25 hybrid retrieval (RRF) |
| v9 | 0.957 | + hard negative enriched CE training |
| v10 | 0.962 | + dense + BM25 combo stacking |
| v11 | 0.964 | + triple BM25 (basic + stemmed) |
| v13 | **0.968** | + 5-fold cross-topic CE ensemble |

See [`results/experiments.md`](results/experiments.md) for the full experiment log including all 15 versions, failed approaches, and error analysis.

## What Worked

- **BM25 hybrid recall** (+0.010): Adding BM25 candidates alongside dense retrieval captured lexical matches the bi-encoder missed.
- **Hard negative CE training** (+0.016): Training the cross-encoder on bi-encoder's top-ranked unjudged docs as negatives made it much better at distinguishing relevant from near-miss documents.
- **Better CE model** (+0.018): ettin-reranker-68m substantially outperformed MiniLM despite being smaller.
- **Cross-topic CE ensemble** (+0.004): Multiple fold-diverse CEs reduced variance on unseen topics.

## What Failed

- Bi-encoder ensembles and union-of-retrievers always hurt LB
- Larger CE models (bge-reranker-base) hurt validation
- LLM listwise reranking (Gemini, Llama-70B) hurt validation
- Zero-shot pretrained rerankers (bge-reranker-v2-m3) hurt LB by -0.015
- Cohere rerank API was unusable due to free-tier rate limits

## Repo Structure

```
C10-team-meru/
├── README.md                    this file
├── requirements.txt             Python dependencies
├── .gitignore
├── docs/                        cohort challenge documents
│   ├── problem_statement.pdf
│   ├── data_card.pdf
│   ├── impact_statement_card.pdf
│   └── stakeholder_engagement.pdf
├── notebooks/
│   └── final_submission.ipynb   winning Kaggle notebook (v13)
├── src/
│   └── pipeline.py              retrieval + reranking pipeline (importable module)
├── scripts/
│   └── run_pipeline.py          end-to-end: data → retrieval → CE rerank → submission.csv
├── results/
│   └── experiments.md           full experiment log (v1-v15) with scores and analysis
├── data/                        competition dataset (included for reproducibility)
│   ├── documents.csv
│   ├── train_queries.csv
│   ├── test_queries.csv
│   └── qrels_train.csv
```

## Reproduction

### 1. Setup

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt_tab stopwords
```

### 2. Data

Competition data is included in the `data/` directory.

### 3. Models

The fine-tuned bi-encoder (`hnm_final_r2`) must be available locally. Update the model path in `scripts/run_pipeline.py`. Cross-encoder models are trained from scratch during pipeline execution.

### 4. Run

```bash
python scripts/run_pipeline.py \
    --data-dir data/ \
    --biencoder-path models/hnm_final_r2 \
    --output submission.csv \
    --n-folds 5
```

First run takes approximately 15-20 minutes on a P100 GPU (candidate generation + CE training + scoring).

## Requirements

- Python 3.10+
- PyTorch with CUDA support
- ~4GB GPU memory (ettin-68m is lightweight)
- See `requirements.txt` for full dependency list

## Team

**Team Meru - AI6** | AI Saturdays Lagos, Cohort 10

## License

This solution is released for educational purposes as part of the AI Saturdays Lagos program.
