# Stakeholder Engagement

**Team Meru -- AI6 | AI Saturdays Lagos Cohort 10**
**Agricultural Extension RAG: Smart Retrieval for Farmers**

---

## 1. Overview

A retrieval system for agricultural extension does not exist in isolation. Its value depends on how well it serves the people who will interact with it -- directly or indirectly. This document identifies the primary stakeholder groups for our agricultural RAG retrieval system, analyzes the potential benefits and risks for each group, describes engagement methods appropriate to each group's context, and explains how stakeholder input would influence project design in a deployment scenario.

We identify two primary stakeholder groups: **smallholder farmers** and **agricultural extension officers and agronomists**. While other groups (policymakers, agricultural researchers, technology platform providers) also have stakes in this work, farmers and extension officers are the most directly affected and therefore receive focused attention here.

## 2. Stakeholder Group 1: Smallholder Farmers

### 2.1 Profile

Smallholder farmers in Sub-Saharan Africa typically cultivate plots of less than 2 hectares, grow a mix of staple and cash crops, and operate with limited access to formal agricultural advisory services. They face diverse challenges including pest and disease pressure, soil degradation, unpredictable rainfall, and limited market access. Their information needs are immediate and practical: "What is eating my cassava leaves?" or "When should I plant maize this season?"

Many smallholder farmers have limited formal education and may not be literate in English. They often access information through oral channels -- conversations with other farmers, radio broadcasts, and visits from extension officers -- rather than written documents.

### 2.2 Potential Benefits

- **Faster access to relevant guidance**: A well-performing retrieval system can surface the right factsheet within seconds, dramatically reducing the time between recognizing a problem and finding a solution.
- **Broader knowledge access**: Farmers currently depend on whichever extension officer is assigned to their area. A retrieval system can draw from 695 factsheets across 7 institutional sources, offering a breadth of knowledge that no single officer can match.
- **Reduced information asymmetry**: Farmers who can access authoritative guidance directly are better positioned to evaluate advice from input dealers, seed companies, and other actors who may have commercial interests.
- **Support for diverse cropping systems**: Farmers growing less common crops (cowpea, sorghum, groundnut) may find it especially difficult to get crop-specific advice from generalist extension officers. A retrieval system that covers 13 crops can fill this gap.

### 2.3 Potential Risks

- **Inaccessibility**: If the system requires English literacy and smartphone access, it excludes the farmers who need it most. The current system operates in English only and assumes text-based interaction.
- **Misleading retrievals**: A farmer who receives an incorrect document and acts on it may suffer crop losses. Unlike an extension officer, the system cannot observe the farmer's field conditions and adjust its advice accordingly.
- **Erosion of trust**: If early experiences with the system produce poor results, farmers may lose trust not only in the technology but in the institutions behind it. Rebuilding trust is far harder than building it.
- **Dependency and deskilling**: Over-reliance on a retrieval system could reduce farmers' engagement with local knowledge networks and traditional practices that are well-adapted to their specific conditions.

### 2.4 Engagement Method: Distributed Dialogue

**Why this method**: Smallholder farmers are geographically dispersed, have varying levels of literacy, and may be unfamiliar with technology design processes. Distributed Dialogue is an engagement method that meets people where they are, using structured conversations facilitated by trusted local intermediaries.

**How it would work**:

1. **Identify local facilitators**: Partner with farmer cooperatives, community-based organizations, and local NGOs who have established relationships with farming communities. Facilitators should speak the local language and understand local agricultural practices.

2. **Structured question sessions**: Facilitators would present farmers with example queries and retrieved documents (translated into local languages and read aloud if necessary), then ask structured questions:
   - "Does this document answer the question?"
   - "Would this information be useful to you?"
   - "How would you ask this question in your own words?"
   - "What questions do you have that are not covered here?"

3. **Feedback aggregation**: Responses would be collected across multiple communities and regions to identify patterns -- which types of queries work well, which fail, and what information needs are unmet by the current corpus.

4. **Iterative refinement**: Farmer feedback would directly inform query reformulation strategies, corpus expansion priorities, and relevance threshold calibration.

**Influence on project design**: Farmer engagement would reshape the project in several concrete ways. Query augmentation strategies would be informed by how farmers actually phrase questions, rather than how researchers imagine they do. Corpus gaps identified by farmers (e.g., missing guidance on a locally important practice) would drive targeted document collection. Relevance judgments from farmers could supplement expert labels, capturing dimensions of relevance (clarity, actionability, local applicability) that domain experts may weigh differently.

## 3. Stakeholder Group 2: Agricultural Extension Officers and Agronomists

### 3.1 Profile

Extension officers are the primary intermediaries between agricultural research and farming communities. They visit farmers, diagnose problems in the field, recommend practices, and sometimes distribute inputs. In many countries, extension services are severely under-resourced: a single officer may be responsible for thousands of farmers across a wide geographic area, covering crops and conditions outside their core expertise.

Agronomists working in research institutions, NGOs, and private sector advisory services face similar challenges at a different scale. They need to quickly find authoritative references to support recommendations, training materials, and policy briefs.

### 3.2 Potential Benefits

- **Rapid reference lookup**: An extension officer preparing for a field visit can quickly find the most relevant factsheets for the crops and conditions they will encounter, even outside their area of specialization.
- **Evidence-based recommendations**: The system surfaces peer-reviewed, institutionally validated documents, helping officers ground their advice in the best available evidence rather than memory or habit.
- **Training support**: New extension officers can use the system as a learning tool, exploring the corpus to build their knowledge base across unfamiliar crops and practices.
- **Workload management**: By automating the document search step, the system frees officers to spend more time on activities that require human judgment -- field observation, farmer interaction, and contextual adaptation of general guidance.

### 3.3 Potential Risks

- **Automation bias**: Officers who trust the system uncritically may recommend a retrieved document's guidance without considering whether it applies to the specific field conditions they observe. A factsheet on maize pest management in East Africa may not apply directly to West African conditions.
- **Skill atrophy**: If officers rely on the system for document retrieval, they may lose familiarity with the corpus and the ability to navigate it independently. This creates vulnerability if the system becomes unavailable.
- **Misaligned incentives**: If the system is positioned as a replacement for extension services rather than a complement to them, it could be used to justify further reductions in extension funding and staffing.
- **Quality assurance burden**: Officers may be expected to validate system outputs on top of their existing workload, creating additional burden without additional resources.

### 3.4 Engagement Method: Participatory Design

**Why this method**: Extension officers and agronomists are literate professionals who interact with information systems regularly. They can engage directly in system design, testing, and evaluation. Participatory Design treats them as co-designers rather than passive users.

**How it would work**:

1. **Needs assessment workshops**: Conduct workshops with groups of 8--12 extension officers to understand their current information-seeking practices. Key questions include:
   - "How do you currently find factsheets or technical documents?"
   - "What makes a document useful in the field versus useful in the office?"
   - "When has a search system failed you, and why?"

2. **Prototype evaluation sessions**: Present officers with the retrieval pipeline's outputs for queries relevant to their work. Ask them to evaluate not only relevance but also actionability, clarity, and appropriateness to their regional context. Capture cases where the system ranks documents in an order that does not match the officer's judgment.

3. **Comparative evaluation**: Have officers perform the same retrieval task using the system and their current methods (manual search, memory, colleague consultation). Compare outcomes in terms of time, accuracy, and satisfaction to establish whether the system provides genuine value.

4. **Co-design of interface requirements**: Work with officers to define how retrieved results should be presented. Should documents be shown with summaries? Should the system highlight which section of a long factsheet is most relevant to the query? Should confidence indicators be displayed?

5. **Ongoing feedback channels**: Establish a mechanism for officers to report retrieval failures, suggest new queries for evaluation, and flag documents that are outdated or inaccurate. This creates a continuous improvement loop grounded in professional use.

**Influence on project design**: Participatory Design with extension officers would affect the pipeline at multiple levels. Their assessment of retrieval failures would identify systematic weaknesses -- for instance, if the system consistently confuses pest management and disease management documents, or if it fails on queries that reference local crop varieties by name rather than scientific nomenclature. Interface requirements defined by officers would shape how the retrieval system is integrated into broader advisory workflows. Their expertise in distinguishing between generically relevant and practically actionable documents could inform a more nuanced relevance labeling scheme for future training data.

## 4. Cross-Stakeholder Considerations

Both stakeholder groups share a concern about **trust calibration** -- understanding when the system's outputs should be relied upon and when they should be questioned. For farmers, this manifests as a need for clear communication about what the system can and cannot do. For extension officers, it manifests as a need for confidence indicators and mechanisms to override or supplement system outputs.

Both groups also share a concern about **representation**. Farmers growing underrepresented crops and officers working in underrepresented regions will experience lower retrieval quality unless the system is actively monitored and improved for equity across crops and geographies.

Engagement with both groups should be ongoing, not a one-time consultation. Agricultural information needs shift with seasons, markets, and climate. A retrieval system that was well-calibrated at launch may drift out of alignment with stakeholder needs within a single growing season. Sustained engagement ensures the system remains useful to the people it is designed to serve.
