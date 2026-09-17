# Experiment Log

Full record of all submissions for the Agricultural Extension RAG competition by Team Meru - AI6.

## Leaderboard Submissions

| Version | Public LB | Private LB | Method | Date |
|---|---|---|---|---|
| v1 | 0.89648 | 0.89161 | Dense retrieval only (e5-base-v2, HNM R2) | 2026-08-05 |
| v2 | 0.91617 | 0.92732 | Dense + MiniLM cross-encoder reranking | 2026-08-05 |
| v3 | 0.92240 | 0.93407 | HNM round 2 bi-encoder + MiniLM CE | 2026-08-08 |
| v4 | 0.91848 | 0.93317 | Weighted bi-encoder ensemble + MiniLM CE (WORSE) | 2026-08-09 |
| v5 | 0.94047 | 0.92649 | HNM R2 + ettin-68m CE (replaced MiniLM) | 2026-08-09 |
| v6 | 0.94076 | 0.92513 | HNM R2 + ettin-68m CE + top-100 candidates | 2026-08-09 |
| v7 | 0.93855 | 0.93349 | Union of 3 bi-encoders + ettin-68m CE (WORSE) | 2026-08-10 |
| v8 | 0.95110 | 0.92888 | BM25 hybrid (RRF fusion) + baseline CE | 2026-08-10 |
| v9 | 0.95718 | 0.93969 | Hard negative enriched CE + dense candidates | 2026-08-10 |
| v10 | 0.96249 | 0.94435 | Hard neg CE + BM25 hybrid (dense + full combo) | 2026-08-10 |
| v11 | 0.96410 | 0.94632 | Hard neg CE + triple BM25 (basic + stemmed) | 2026-08-13 |
| v12 | 0.96083 | 0.94319 | Quad retrieval (ft + zs + bm25 + bm25stem) (WORSE) | 2026-08-13 |
| **v13** | **0.96835** | **0.95478** | **5-fold cross-topic CE ensemble + triple BM25 (WINNING)** | **2026-08-14** |
| v14 | 0.97048 | 0.95135 | 10-fold CE ensemble (CE-only, new public best) | 2026-09-06 |
| v15 | 0.95564 | 0.94288 | 10-fold CE + bge-reranker-v2-m3 blend (WORSE) | 2026-09-06 |

**Final standing:** 1st place, 0.95478 private LB (v13 selected)

---

## Phase 1: Baseline Dense Retrieval (v1-v3)

**v1 (0.896):** Fine-tuned e5-base-v2 with 2-round hard negative mining (MNRL loss). Dense-only retrieval, no reranking. Established the bi-encoder backbone used throughout.

**v2 (0.916):** Added MiniLM cross-encoder reranking on top-100 dense candidates. +0.020 over dense-only.

**v3 (0.922):** Confirmed that HNM round 2 improves the bi-encoder. Marginal gain with same CE.

## Phase 2: Cross-Encoder Improvements (v4-v7)

**v4 (0.918, WORSE):** Tried weighted ensemble of multiple bi-encoders before CE. Hurt performance -- ensembling bi-encoders adds noise when recall is already strong.

**v5 (0.940):** Switched from MiniLM to ettin-reranker-68m-v1. Major improvement (+0.018) despite the model being smaller. ettin was specifically designed for reranking.

**v6 (0.941):** Increased candidate pool to top-100. Marginal gain.

**v7 (0.939, WORSE):** Union of 3 different bi-encoders as candidate source. Hurt performance -- adding zero-shot bi-encoder candidates introduced irrelevant docs that confused the CE.

## Phase 3: BM25 Hybrid + Hard Negatives (v8-v12)

**v8 (0.951):** Added BM25 retrieval via RRF fusion with dense. Single biggest improvement: +0.010. BM25 captures lexical matches (crop names, chemical terms) that semantic models miss.

**v9 (0.957):** Enriched CE training with hard negatives -- top-30 unjudged docs from bi-encoder retrieval added as label=0. +0.016 improvement. The CE learned to distinguish near-miss documents.

**v10 (0.962):** Combined hard neg CE with BM25 hybrid candidates. Gains stack: +0.005.

**v11 (0.964):** Added stemmed BM25 (Porter stemmer + stopword removal) as third retrieval signal. Triple BM25 = dense + BM25-basic + BM25-stemmed, fused via RRF. +0.002.

**v12 (0.961, WORSE):** Tried quad retrieval adding zero-shot bi-encoder candidates. Hurt LB -- confirms that more retrievers is not better when recall is already saturated.

## Phase 4: Cross-Encoder Ensembles (v13-v14)

**v13 (0.968 public / 0.955 private, WINNING):** 5-fold cross-topic CE ensemble. Each of 5 ettin-68m models trained on 80% of topics (different fold held out). Averaged scores at inference. +0.004 over single CE. This was the submission selected for private LB evaluation.

**v14 (0.970 public):** Extended to 10-fold CE ensemble (each model sees 90% of topics). +0.002 over 5-fold on public LB. However, v13 was already locked in for final evaluation.

## Phase 5: Failed Diversification Attempts (v15)

**v15 (0.956, WORSE):** Blended 10-fold finetuned CE with zero-shot BAAI/bge-reranker-v2-m3. Despite the pretrained reranker adding an independent signal, it hurt LB by -0.015. The zero-shot model adds noise on topics it has never seen.

## LLM Listwise Reranking (not submitted)

Tested LLM-based listwise reranking on validation set:

| Provider | Model | Val nDCG@5 | Delta vs CE |
|---|---|---|---|
| Gemini | gemini-3.5-flash-lite | 0.9756 | -0.019 |
| Together AI | Llama-3.3-70B-Instruct-Turbo | 0.9469 | -0.048 |
| RRF(CE+LLM) | Together AI blend | 0.9785 | -0.016 |

LLMs consistently underperformed the fine-tuned CE. They tend to prefer verbosely relevant documents over precisely relevant ones.

## Cohere Rerank (not submitted)

Attempted Cohere rerank-english-v3.0 via API. Free tier rate limits caused ~70% query failures (HTTP 429). Even with retry + exponential backoff, too many queries failed to produce reliable scores. Abandoned.

---

## Error Analysis

### Recall is NOT the bottleneck

Triple RRF@100 recall analysis on training set:
- **100%** of rel-3 documents appear in top-100
- **100%** of rel-2 documents appear in top-100
- **90.7%** of rel-1 documents appear in top-100

All highly relevant documents are retrieved. More retrievers, HyDE, or query expansion cannot help.

### CE ranking IS the bottleneck

- Oracle nDCG@5 (perfect ranking within top-100): **0.9895**
- Single CE nDCG@5: **0.9417**
- Gap: **0.048** -- entirely due to CE ranking errors

Failure modes:
- **Needle in haystack:** 1 rel-3 doc among 99 rel-0 docs. Small topics with few positive examples are hardest.
- **Small topics:** Topics with only 2 training queries provide minimal signal for the CE.
- 51 queries had CE nDCG < 0.95; 6 queries scored 0.000 (complete ranking failure).

---

## Summary: What Worked vs What Failed

### Worked

| Technique | LB Impact | Notes |
|---|---|---|
| Better CE model (MiniLM to ettin-68m) | +0.018 | Biggest single change |
| Hard negative CE training | +0.016 | Top-30 unjudged as neg examples |
| BM25 hybrid retrieval (RRF) | +0.010 | Lexical + semantic complementarity |
| Dense + BM25 combo stacking | +0.005 | Gains are additive |
| 5-fold cross-topic CE ensemble | +0.004 | Reduces variance on unseen topics |
| Triple BM25 (basic + stemmed) | +0.002 | Stemming captures morphological variants |
| 10-fold CE ensemble (over 5-fold) | +0.002 | More folds = more diversity |

### Failed

| Technique | LB Impact | Why |
|---|---|---|
| Bi-encoder ensembles/unions | -0.003 to -0.006 | Recall already saturated; noise from weak models |
| Larger CE models (bge-reranker-base) | hurt val | Overfitting on small training set |
| Quad retrieval (adding ZS bi-encoder) | -0.003 | More candidates != better candidates |
| LLM listwise reranking | -0.016 to -0.048 | LLMs prefer verbose over precise relevance |
| Zero-shot pretrained reranker blend | -0.015 | Adds noise on unseen topics |
| Dropout/regularization | hurt val | Already regularized by small model + fold split |
| 2-round CE (CE-mined hard negs) | slight hurt | Diminishing returns from recursive mining |
| BM25 score fusion at reranking | marginal | RRF for candidates is enough |
| Cohere rerank API | unusable | Free tier rate limits |

### Key Insight

With perfect recall in the candidate set, the entire competition reduced to a **cross-encoder ranking problem**. The winning strategy was not about retrieving more documents but about training a better ranker: hard negatives, cross-topic fold diversity, and the right model size.
