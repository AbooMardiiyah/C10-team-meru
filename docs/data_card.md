# Data Card

**Team Meru -- AI6 | AI Saturdays Lagos Cohort 10**
**Agricultural Extension RAG: Smart Retrieval for Farmers**

---

## 1. Dataset Overview

This data card describes the dataset used in the Agricultural Extension RAG competition, which tasks participants with building a retrieval system that ranks agricultural factsheets by relevance to farmer queries.

| Component | Count | Description |
|-----------|-------|-------------|
| Factsheets (corpus) | 695 | Agricultural extension documents from institutional sources |
| Training queries | 308 | Farmer-style questions with graded relevance labels |
| Test queries | 200 | Held-out queries with zero topic overlap to training |
| Crops covered | 13 | Maize, Tomato, Rice, Cassava, Cowpea, Sorghum, Millet, Groundnut, Soybean, Yam, Cocoa, Cotton, Wheat |
| Source organizations | 7 | CGIAR, FAO, IITA, and other agricultural research institutions |
| Countries represented | 21 | Across Sub-Saharan Africa |

## 2. Input and Output Specification

**Input**: A natural-language query representing a farmer's question. Queries are typically short (5--20 words), informal, and may use non-technical language. Examples include questions about pest identification, planting schedules, soil management, and post-harvest storage.

**Output**: A ranked list of exactly 5 document IDs from the factsheet corpus, ordered by predicted relevance. The system must assign a ranking position (1 through 5) to each selected document.

## 3. Task Formulation

This is a **learning-to-rank** problem with graded relevance judgments, not a binary classification task. Each query-document pair in the training set carries a relevance label from 0 to 3:

| Label | Meaning |
|-------|---------|
| 0 | Not relevant -- the document does not address the query |
| 1 | Marginally relevant -- the document touches on the topic but does not directly answer the question |
| 2 | Relevant -- the document addresses the query's topic with useful information |
| 3 | Highly relevant -- the document directly and substantively answers the farmer's question |

The evaluation metric, nDCG@5, rewards systems that place highly relevant documents at higher ranks. It is a position-aware metric: a relevance-3 document at rank 1 contributes more to the score than the same document at rank 5.

## 4. Labeling Process

Relevance labels were created through **expert relevance judgments**. Agricultural domain experts assessed each query-document pair and assigned graded relevance scores. This process reflects standard practice in information retrieval evaluation (following the TREC tradition), where human assessors with domain knowledge determine the degree to which a document satisfies an information need.

The graded scale (0--3) captures nuance that binary labels cannot. A document about maize pest management may be marginally relevant to a query about maize storage (both concern maize post-production) but not directly relevant. This distinction matters for training rerankers that must learn fine-grained relevance differences.

## 5. Data Creation and Provenance

### 5.1 Factsheet Corpus

The 695 factsheets were sourced from publicly available agricultural extension materials produced by international and national agricultural research organizations. These documents are published under Creative Commons Attribution (CC-BY) licenses or equivalent open-access terms, as is standard for publicly funded agricultural research outputs.

Each factsheet is a structured document covering a specific agricultural topic -- for example, integrated pest management for fall armyworm in maize, or best practices for cassava processing. Documents vary in length and technical depth but share the common purpose of conveying research-backed agricultural guidance to practitioners.

### 5.2 Query Construction

Training and test queries were constructed to simulate the types of questions smallholder farmers would ask when seeking agricultural guidance. Queries span diverse topics including:

- Pest and disease identification and management
- Planting and harvesting schedules
- Soil fertility and nutrient management
- Post-harvest handling and storage
- Seed selection and variety recommendations
- Water management and irrigation

The deliberate **zero topic overlap** between training and test splits ensures that models are evaluated on their ability to generalize, not memorize.

### 5.3 Web Collection and Ethics

The factsheet corpus consists of documents published by agricultural research institutions for the explicit purpose of broad dissemination. These organizations produce extension materials to reach farmers, extension officers, and policymakers. Using these materials for retrieval system development is consistent with their intended purpose.

No personally identifiable information about individual farmers is present in the dataset. Queries are synthetic representations of farmer information needs, not transcripts of real interactions.

## 6. Bias and Representation

### 6.1 Crop Distribution

The corpus exhibits uneven crop representation. Maize, as the most widely grown staple crop in Sub-Saharan Africa, is substantially over-represented relative to crops such as millet, cocoa, or cotton. This imbalance reflects the real-world distribution of agricultural research funding and extension material production, but it creates a risk that retrieval systems will perform better on well-represented crops.

**Implications for our pipeline**: We used cross-topic 5-fold splits during cross-encoder training specifically to prevent the model from over-fitting to well-represented crops. By ensuring that each fold's validation set contains topics not seen during training, we forced the reranker to learn generalizable relevance patterns.

### 6.2 Language

All documents and queries are in English. This is a significant limitation given that many smallholder farmers in Sub-Saharan Africa communicate primarily in local languages (Yoruba, Hausa, Swahili, Amharic, etc.). A production deployment would require multilingual support or translation layers.

### 6.3 Geographic Coverage

While 21 countries are represented, coverage is uneven. Countries with larger national agricultural research systems and stronger ties to international organizations like CGIAR tend to have more documents. Farming practices that are region-specific (e.g., flood-recession agriculture in the Sahel) may be underrepresented.

### 6.4 Temporal Coverage

Agricultural guidance evolves as new varieties are released, pest pressures shift, and climate patterns change. The factsheets in this corpus represent a snapshot in time. A deployed system would need mechanisms to incorporate updated guidance and deprecate outdated recommendations.

## 7. Values and Community Alignment

This dataset was created to support a competition aimed at improving information access for smallholder farmers -- a population that is underserved by existing digital information systems. The values embedded in the dataset design include:

- **Equity of access**: Agricultural knowledge produced with public funding should be accessible to the farmers it is meant to serve, not locked behind institutional barriers or poor search interfaces.
- **Respect for expertise**: The graded relevance labels reflect domain expert judgment, ensuring that models are trained against a meaningful standard of quality.
- **Generalization as a design principle**: The zero-overlap train/test split encodes the expectation that retrieval systems should work for new questions and new topics, not only for queries similar to those seen during training.
- **Transparency of provenance**: The use of CC-BY licensed materials ensures that the dataset can be shared, studied, and built upon without legal ambiguity.

The ultimate community this work aims to serve -- smallholder farmers in Sub-Saharan Africa -- was not directly involved in dataset creation. This is a limitation. Future iterations should incorporate farmer input on query formulation, relevance assessment, and information need prioritization.
