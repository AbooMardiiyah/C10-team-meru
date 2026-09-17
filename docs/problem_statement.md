# Problem Statement

**Team Meru -- AI6 | AI Saturdays Lagos Cohort 10**
**Agricultural Extension RAG: Smart Retrieval for Farmers**

---

## Background

Smallholder farmers across Sub-Saharan Africa rely on timely, accurate agricultural guidance to make decisions about planting, pest management, soil health, and post-harvest practices. A wealth of knowledge exists in the form of agricultural extension factsheets produced by organizations such as CGIAR, FAO, IITA, and national agricultural research systems. However, this information is scattered across institutional repositories, formatted as dense technical documents, and organized in ways that do not match how farmers actually seek help.

When a farmer asks a short, informal question -- for example, "Why are my tomato leaves turning yellow?" -- traditional keyword-based search systems struggle to connect that query to the correct factsheet on nutrient deficiency management in solanaceous crops. The vocabulary gap between a farmer's everyday language and the technical terminology used in extension documents means that the most relevant guidance is often buried beneath irrelevant results or missed entirely.

## Problem Definition

The core challenge is one of **semantic retrieval**: given a farmer's natural-language question, identify and rank the most relevant agricultural extension documents from a corpus of 695 factsheets spanning 13 crops, 7 institutional sources, and 21 countries. The system must return exactly 5 documents, ranked by relevance, and is evaluated using Normalized Discounted Cumulative Gain at rank 5 (nDCG@5) against expert-graded relevance labels on a scale of 0 to 3.

This is not a simple classification problem. The graded relevance scale means that partially relevant documents carry value, and the ranking order matters. A system that places a moderately relevant document at position 1 and a highly relevant document at position 5 will score lower than one that reverses that order.

Critically, the training and test query sets have **zero topic overlap**. A model that memorizes associations between specific query terms and specific documents will fail. The system must learn general patterns of semantic relevance that transfer across crops, regions, and question types.

## Project Aim

The aim of this project is to build a robust document retrieval pipeline that serves as the retrieval component of a Retrieval-Augmented Generation (RAG) system for agricultural extension. Specifically, we seek to:

1. Develop a retrieval system that accurately ranks agricultural factsheets by relevance to a given farmer query.
2. Ensure the system generalizes across unseen topics, crops, and regions.
3. Combine complementary retrieval signals -- dense semantic embeddings and sparse lexical matching -- to handle both meaning-based and term-based relevance.
4. Train cross-encoder rerankers that refine initial retrieval candidates into precise top-5 rankings.

## Guiding Values

The following values guided our design choices and evaluation throughout the project:

- **Accuracy**: The system must surface the most relevant documents, not merely plausible ones. Agricultural misinformation can lead to crop loss, wasted inputs, and food insecurity.
- **Relevance**: Rankings must reflect genuine utility to the farmer. A document about the right crop but the wrong practice is not helpful.
- **Inclusivity**: The system should perform well across all 13 crops and all represented regions, not only the most common ones. Farmers growing cowpea or sorghum deserve the same retrieval quality as those growing maize.
- **Fairness**: Retrieval quality should not degrade for underrepresented crops or regions in the training data. Cross-topic generalization is a fairness concern as much as a technical one.
- **Transparency**: The pipeline should be interpretable. Each stage -- bi-encoder retrieval, BM25 fusion, cross-encoder reranking -- has a clear purpose and can be evaluated independently.

## What Success Looks Like

Success means building a retrieval system that a downstream RAG pipeline can trust. When a farmer asks a question, the top-5 documents returned should contain the information needed to generate a correct, actionable answer. Quantitatively, success is measured by nDCG@5 on the held-out test set of 200 queries. Our final system achieved a private leaderboard score of **0.95478**, earning 1st place in the competition.

Beyond the leaderboard, success means demonstrating that careful engineering -- hard negative mining, reciprocal rank fusion, cross-topic ensemble reranking -- can close the gap between how farmers ask questions and how experts write answers.
