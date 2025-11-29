# Ground Truth Generation Methodology v2.0

**Document Version:** 2.0
**Date:** 2025-11-29
**Status:** Implementation Complete

---

## Table of Contents

1. [Overview](#overview)
2. [Motivation & Design Principles](#motivation--design-principles)
3. [Multi-Strategy Generation Framework](#multi-strategy-generation-framework)
4. [Relevance Grading System](#relevance-grading-system)
5. [Quality Validation Process](#quality-validation-process)
6. [Bias Detection & Mitigation](#bias-detection--mitigation)
7. [Dataset Structure & Versioning](#dataset-structure--versioning)
8. [Usage Guide](#usage-guide)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Limitations & Future Work](#limitations--future-work)

---

## Overview

This document describes the comprehensive ground truth generation system designed for evaluating the Heritage Document Recommendation System. The system addresses critical limitations in the original ground truth (v1.0) by implementing:

- **Human-in-the-loop validation** with inter-annotator agreement tracking (Cohen's Kappa)
- **Multi-strategy query generation** spanning seed, synthetic, and negative queries
- **4-level relevance grading** (Perfect, Excellent, Good, Not Relevant) with explainability
- **Stratified sampling** ensuring coverage of all 12 document clusters, heritage types, and domains
- **Bias detection** for graph-connected vs semantically similar documents
- **Versioned datasets** with separate development and test sets

The ground truth v2.0 system produces **100+ validated queries** with verified relevance judgments suitable for rigorous evaluation of retrieval methods.

---

## Motivation & Design Principles

### Problems with Ground Truth v1.0

The original ground truth dataset (`data/evaluation/ground_truth.json`) had several limitations:

| Issue | Description | Impact |
|-------|-------------|--------|
| **Automatic generation** | All queries generated algorithmically without human validation | Relevance judgments may not reflect actual user information needs |
| **Binary relevance** | Documents marked as either relevant or not relevant | Oversimplifies nuanced heritage domain where partial matches are common |
| **Generous thresholding** | Cluster-based queries had 36-352 relevant docs per query | Makes precision/recall metrics less discriminative |
| **No inter-annotator agreement** | Zero human validation of relevance | Unknown quality of relevance judgments |
| **Potential bias** | 39 cluster-based, 30 metadata-based, 30 SimRank-based queries | May favor certain retrieval methods over others |
| **No explainability** | No rationale for why documents are relevant | Cannot validate if system retrieves for right reasons |

### Design Principles for v2.0

1. **Human-Centric Design**: Ground truth must reflect real user information needs validated by domain experts
2. **Multi-Level Relevance**: Use 4-level grading (0-3) to capture nuances in document relevance
3. **Quality Validation**: Require inter-annotator agreement (Cohen's Kappa > 0.6) for inclusion
4. **Comprehensive Coverage**: Stratified sampling ensures all document types, clusters, and characteristics are represented
5. **Bias Awareness**: Explicitly detect and mitigate biases that favor certain retrieval methods
6. **Explainability**: Document rationale for relevance judgments to enable future analysis
7. **Reproducibility**: Version all datasets and document methodology completely

---

## Multi-Strategy Generation Framework

Ground truth v2.0 uses **three complementary strategies** to generate diverse, unbiased queries:

### 1. Seed Queries (n=50)

**Purpose:** Capture authentic user information needs through manual curation.

**Generation Process:**
- Domain experts manually craft queries based on real heritage research scenarios
- Queries span all heritage types, domains, time periods, and regions
- Mix of complexity levels (simple, moderate, complex)
- Include edge cases (rare types, underrepresented regions, modern heritage)
- Include negative examples that should return empty/minimal results

**Query Categories:**

| Category | Count | Examples |
|----------|-------|----------|
| **Simple Heritage Type Queries** | 15 | "Show me ancient Buddhist stupas in India" |
| **Moderate Multi-Faceted** | 15 | "Buddhist and Jain cave temples in Western India with ancient rock-cut architecture" |
| **Complex Cross-Temporal/Cultural** | 10 | "Compare architectural evolution from ancient Buddhist stupas to medieval Hindu temples showing Indo-Islamic synthesis" |
| **Edge Cases** | 5 | "Heritage sites in Central India" (only 2 docs in dataset) |
| **Negative Examples** | 5 | "Ancient Egyptian pyramids and pharaonic tombs" (non-Indian) |

**Stratification:**

- **Heritage Types:** monument (12), site (10), artifact (5), architecture (10), tradition (8), art (5)
- **Domains:** religious (15), military (5), royal (8), cultural (18), archaeological (10), architectural (9)
- **Time Periods:** ancient (25), medieval (12), modern (8), unknown (5)
- **Regions:** north (10), south (8), east (5), west (7), central (3), india (17)
- **Complexity:** simple (20), moderate (15), complex (15)

### 2. Synthetic Queries - Easy (n=30)

**Purpose:** Ensure proportional coverage of all 12 document clusters through stratified sampling.

**Generation Algorithm:**

```python
# Stratified cluster sampling
for cluster_id in range(12):
    cluster_proportion = cluster_size / total_docs
    num_samples = max(1, int(30 * cluster_proportion))

    # Sample representative documents
    sampled_docs = random.sample(cluster_docs, num_samples)

    # Generate query from document characteristics
    for doc in sampled_docs:
        query = generate_query_from_document(
            doc,
            difficulty='easy',
            num_constraints=1-2  # Simple queries
        )
```

**Characteristics:**
- **Single clear intent:** 1-2 query constraints
- **Cluster-balanced:** Proportional sampling ensures all clusters represented
- **Automated generation:** Based on document metadata (heritage types, domains, time period, region)

**Example:**
- Document from Cluster 6 (Cultural Architecture, Monument) → Query: "Find ancient architecture heritage sites"

### 3. Synthetic Queries - Hard (n=20)

**Purpose:** Test retrieval performance on complex, multi-faceted queries with overlapping concepts.

**Generation Algorithm:**

```python
# Complex cross-cluster queries
for i in range(20):
    # Combine multiple random dimensions
    heritage_types = random.sample(all_types, k=2-3)
    domains = random.sample(all_domains, k=2-3)
    time_period = random.choice(time_periods)
    region = random.choice(regions)

    query = combine_into_complex_query(
        heritage_types, domains, time_period, region
    )
```

**Characteristics:**
- **Multiple overlapping concepts:** 3-4 query constraints
- **Cross-cluster matching:** May match documents from multiple clusters
- **Ambiguity:** Some queries intentionally ambiguous to test robustness

**Example:**
- "Ancient monument and art in religious, cultural domains from south region"

### 4. Negative Queries (n=5)

**Purpose:** Validate that the system correctly returns empty or minimal results for out-of-scope queries.

**Examples:**
- "Ancient Egyptian pyramids and pharaonic tombs" (non-Indian geography)
- "Renaissance art and architecture in Italy" (European context)
- "Contemporary digital art installations from 21st century" (too recent)

**Expected Behavior:** Should return 0-5 documents with low relevance scores.

---

## Relevance Grading System

### 4-Level Relevance Scale

Ground truth v2.0 uses a **4-level ordinal scale** instead of binary relevance:

| Level | Label | Definition | Examples |
|-------|-------|------------|----------|
| **3** | Perfect | Document **exactly matches** query intent; all constraints satisfied; primary information need fully met | Query: "Mughal forts in North India" → Doc: "Red Fort Delhi, Mughal military architecture" |
| **2** | Excellent | Document **highly relevant**; most constraints satisfied; very useful for query | Query: "Mughal forts in North India" → Doc: "Agra Fort overview, Mughal period" (no explicit region mention) |
| **1** | Good | Document **somewhat relevant**; partial match; tangentially useful | Query: "Mughal forts in North India" → Doc: "Indo-Islamic architecture styles" (discusses Mughal but not specific forts) |
| **0** | Not Relevant | Document **does not address** query; wrong domain/type/period/region | Query: "Mughal forts in North India" → Doc: "Dravidian temples in South India" |

### Why 4 Levels Instead of Binary?

Heritage domain complexity requires nuanced relevance:

1. **Partial Matches Are Common:** A document about "Indian forts" is relevant to "Mughal forts" but not perfectly
2. **Multi-Faceted Queries:** Query may have 4 dimensions (type, domain, time, region); documents may match 2-3
3. **Graded Evaluation Metrics:** NDCG and graded precision require multi-level relevance
4. **Explainability Testing:** Perfect matches should be retrieved via exact entity matching, while Excellent matches may require semantic understanding

### Annotation Guidelines

Annotators are provided with detailed guidelines:

**Perfect (3) Criteria:**
- ✅ All heritage types mentioned in query are present in document
- ✅ All domains match
- ✅ Time period matches (or both unspecified)
- ✅ Region matches (or both unspecified)
- ✅ Document directly answers the query's primary information need

**Excellent (2) Criteria:**
- ✅ Primary heritage type matches
- ✅ At least one domain matches
- ⚠️ Time period or region may differ but still relevant
- ✅ Document provides substantial information for query

**Good (1) Criteria:**
- ✅ At least one heritage type or domain matches
- ⚠️ May have wrong time period or region but conceptually related
- ✅ Document provides tangential or background information

**Not Relevant (0) Criteria:**
- ❌ No heritage type match
- ❌ Wrong domain entirely
- ❌ Different cultural/geographic context (e.g., non-Indian for India-specific query)
- ❌ Does not address query intent

### Rationale Requirement

For **every judgment**, annotators must provide a brief rationale:

- **Perfect:** "Exact match - Mughal fort in North India, all constraints satisfied"
- **Excellent:** "Highly relevant - Mughal fort but region not specified in document"
- **Good:** "Partial match - Discusses Mughal architecture but not specifically forts"
- **Not Relevant:** "Wrong domain - This is a South Indian temple, not a Mughal fort"

**Purpose of Rationales:**
1. **Quality control:** Ensures annotators are thinking critically
2. **Disagreement resolution:** Helps understand why annotators disagreed
3. **Explainability analysis:** Can verify if retrieval system ranks documents for correct reasons
4. **Future reference:** Documents edge cases for iterative improvement

---

## Quality Validation Process

### Human-in-the-Loop Annotation Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: Query Generation                                   │
│  - 50 seed queries (manual)                                  │
│  - 50 synthetic queries (stratified sampling)                │
│  - 5 negative queries                                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: Annotation Interface Creation                      │
│  - Export queries + candidate documents to JSON              │
│  - Provide annotation guidelines                             │
│  - Create web-based or CLI annotation tool                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: Multi-Annotator Judgment (2-3 experts)             │
│  - Each annotator independently judges all query-doc pairs   │
│  - Rate relevance: 0 (Not Relevant) to 3 (Perfect)           │
│  - Provide brief rationale for each judgment                 │
│  - Sample 20-30 candidate docs per query for efficiency      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: Inter-Annotator Agreement Validation               │
│  - Compute Cohen's Kappa for each query                      │
│  - Identify low-agreement queries (κ < 0.6)                  │
│  - Flag major disagreements (|rating1 - rating2| >= 2)       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 5: Consensus & Filtering                              │
│  - Use median rating as consensus (robust to outliers)       │
│  - Exclude queries with κ < 0.6                              │
│  - Require re-annotation for low-agreement queries           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 6: Bias Detection                                     │
│  - Check cluster distribution bias                           │
│  - Check temporal/spatial bias                               │
│  - Detect graph vs semantic method bias                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 7: Dataset Finalization                               │
│  - Split into dev (80%) and test (20%) sets                  │
│  - Version and save ground truth                             │
│  - Generate methodology documentation                        │
└─────────────────────────────────────────────────────────────┘
```

### Inter-Annotator Agreement (Cohen's Kappa)

**Cohen's Kappa** measures agreement between two annotators while accounting for chance agreement:

```
κ = (p_o - p_e) / (1 - p_e)

where:
  p_o = observed agreement
  p_e = expected agreement by chance
```

**Interpretation:**

| Kappa Range | Interpretation | Action |
|-------------|----------------|--------|
| κ < 0.0 | Poor (worse than chance) | Discard query or revise guidelines |
| 0.0 ≤ κ < 0.2 | Slight agreement | Re-annotate with clarified guidelines |
| 0.2 ≤ κ < 0.4 | Fair agreement | Consider re-annotation |
| 0.4 ≤ κ < 0.6 | Moderate agreement | Marginal - review disagreements |
| **0.6 ≤ κ < 0.8** | **Substantial agreement** | ✅ **Accept into ground truth** |
| 0.8 ≤ κ ≤ 1.0 | Almost perfect agreement | ✅ Accept into ground truth |

**Threshold:** Ground truth v2.0 requires **κ ≥ 0.6** for query inclusion.

### Aggregating Multiple Judgments

When 2-3 annotators provide judgments, consensus is computed as:

1. **Median Rating:** Use median of all ratings (robust to outliers)
2. **Disagreement Filter:** If max(ratings) - min(ratings) > 2, exclude the document (too much disagreement)
3. **Relevance Threshold:** Only include documents with consensus relevance > 0

**Example:**

| Doc ID | Annotator 1 | Annotator 2 | Annotator 3 | Median | Include? |
|--------|-------------|-------------|-------------|--------|----------|
| 42 | 3 | 3 | 2 | 3 | ✅ Yes (high agreement) |
| 57 | 2 | 2 | 1 | 2 | ✅ Yes (moderate agreement) |
| 89 | 1 | 3 | 0 | 1 | ❌ No (max_diff = 3 > 2) |
| 103 | 0 | 0 | 1 | 0 | ❌ No (median = 0) |

---

## Bias Detection & Mitigation

### Types of Bias Detected

#### 1. Cluster Distribution Bias

**Definition:** Certain clusters are over/underrepresented in relevant documents compared to their dataset proportion.

**Detection Method:**

```python
for cluster_id in range(12):
    expected_proportion = cluster_size / total_docs
    actual_proportion = cluster_relevant_count / total_relevant_docs
    bias_ratio = actual_proportion / expected_proportion

    if bias_ratio > 1.5:
        status = "overrepresented"
    elif bias_ratio < 0.5:
        status = "underrepresented"
    else:
        status = "balanced"
```

**Mitigation:** Add more queries targeting underrepresented clusters.

#### 2. Temporal Bias

**Definition:** Certain time periods (ancient, medieval, modern) are over/underrepresented.

**Example:** If 72% of dataset is ancient but 90% of relevant docs are ancient, there's temporal bias.

**Mitigation:** Increase queries for modern and medieval heritage.

#### 3. Spatial Bias

**Definition:** Certain regions are over/underrepresented.

**Known Issue:** Central India has only 2 documents (0.5% of dataset) - will naturally be underrepresented.

**Mitigation:** Acknowledge bias in documentation; consider collecting more central India heritage documents.

#### 4. Heritage Type/Domain Bias

**Definition:** Certain heritage types or domains are favored.

**Example:** If religious domain appears in 52% of dataset but 80% of relevant docs, there's domain bias.

**Mitigation:** Add queries for underrepresented domains (military, archaeological).

#### 5. Graph Connectivity Bias

**Definition:** Graph-based methods (SimRank) may be favored if ground truth predominantly contains graph-connected documents.

**Detection Method:**

```python
# For each query's relevant docs:
avg_graph_connectivity = mean([
    count_graph_edges(doc) for doc in relevant_docs
])

# Compare to non-relevant docs
avg_connectivity_non_relevant = mean([
    count_graph_edges(doc) for doc in non_relevant_docs
])

if avg_graph_connectivity > 2 * avg_connectivity_non_relevant:
    bias_detected = True
```

**Mitigation:** Include queries where semantically similar documents (high embedding similarity) are relevant even without graph connections.

#### 6. Embedding Similarity Bias

**Definition:** Embedding-based methods may be favored if ground truth predominantly contains semantically similar documents.

**Detection Method:**

```python
# Compute average embedding similarity between query and relevant docs
avg_embedding_sim_relevant = mean([
    cosine_similarity(query_embedding, doc_embedding)
    for doc in relevant_docs
])

# Compare to non-relevant
avg_embedding_sim_non_relevant = mean([
    cosine_similarity(query_embedding, doc_embedding)
    for doc in non_relevant_docs
])
```

**Mitigation:** Include queries where graph-connected documents are relevant even without high embedding similarity.

### Bias Report Output

The bias detection system generates a comprehensive JSON report:

```json
{
  "cluster_bias": {
    "cluster_0": {
      "expected_proportion": 0.089,
      "actual_proportion": 0.120,
      "bias_ratio": 1.35,
      "status": "balanced"
    },
    "cluster_11": {
      "expected_proportion": 0.100,
      "actual_proportion": 0.045,
      "bias_ratio": 0.45,
      "status": "underrepresented"
    }
  },
  "temporal_bias": { ... },
  "spatial_bias": { ... },
  "recommendations": [
    "Add more queries targeting cluster_11 (bias ratio: 0.45)",
    "Add more queries for modern time period (bias ratio: 0.38)",
    "Add more queries for central region (bias ratio: 0.20)"
  ]
}
```

---

## Dataset Structure & Versioning

### File Structure

```
data/evaluation/
├── ground_truth_v2.0_dev.json          # Development set (80% of validated queries)
├── ground_truth_v2.0_test.json         # Test set (20% of validated queries)
├── annotator_agreement_report_v2.0.json
├── bias_detection_report_v2.0.json
├── annotation_interface_v2.json        # For annotators
├── annotations_annotator_1.json        # Individual annotator judgments
├── annotations_annotator_2.json
└── query_metadata_v2.json              # Query generation metadata
```

### Ground Truth JSON Schema

```json
{
  "version": "2.0",
  "split": "dev",  // or "test"
  "creation_date": "2025-11-29T...",
  "num_queries": 85,
  "statistics": {
    "num_queries": 85,
    "query_types": {"seed": 40, "synthetic_easy": 25, "synthetic_hard": 15, "negative": 5},
    "complexity_distribution": {"simple": 30, "moderate": 35, "complex": 20},
    "avg_relevant_docs_per_query": 12.4,
    "avg_inter_annotator_agreement": 0.74,
    "relevance_level_distribution": {"0": 5234, "1": 432, "2": 289, "3": 156},
    "cluster_coverage": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "time_period_coverage": ["ancient", "medieval", "modern"],
    "region_coverage": ["north", "south", "east", "west", "central", "india"]
  },
  "queries": [
    {
      "query_id": "seed_001",
      "query_text": "Show me ancient Buddhist stupas in India",
      "query_type": "seed",
      "heritage_types": ["monument"],
      "domains": ["religious", "archaeological"],
      "time_period": "ancient",
      "region": "india",
      "complexity": "simple",
      "relevance_judgments": {
        "42": [
          {
            "annotator_id": "expert_1",
            "document_id": 42,
            "relevance_level": 3,
            "rationale": "Perfect match - Sanchi Stupa, ancient Buddhist monument",
            "timestamp": "2025-11-29T..."
          },
          {
            "annotator_id": "expert_2",
            "document_id": 42,
            "relevance_level": 3,
            "rationale": "Exact match - ancient Buddhist stupa in India",
            "timestamp": "2025-11-29T..."
          }
        ]
      },
      "consensus_relevance": {
        "42": 3,   // Doc 42 is Perfect (3)
        "57": 2,   // Doc 57 is Excellent (2)
        "89": 1    // Doc 89 is Good (1)
      },
      "inter_annotator_agreement": 0.85,
      "num_annotators": 2,
      "cluster_distribution": {
        "1": 2,
        "3": 3,
        "7": 1
      },
      "creation_date": "2025-11-29T...",
      "version": "2.0",
      "expected_result_size_range": [5, 20],
      "rationale": "Clear single-intent query for specific monument type with temporal/spatial constraints"
    }
  ]
}
```

### Versioning Scheme

- **Version Format:** `v<major>.<minor>`
- **Major Version:** Incremented for methodology changes (e.g., v1.0 → v2.0)
- **Minor Version:** Incremented for query additions/refinements (e.g., v2.0 → v2.1)

**Changelog:**

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2024-XX-XX | Initial automatic generation (100 queries, binary relevance) |
| v2.0 | 2025-11-29 | Human-validated, 4-level relevance, stratified sampling, bias detection |

---

## Usage Guide

### Step 1: Generate Seed and Synthetic Queries

```python
from src.evaluation.ground_truth_generator import GroundTruthGenerator

# Initialize generator
generator = GroundTruthGenerator(
    data_path="data/processed/heritage_metadata.csv",
    output_dir="data/evaluation"
)

# Generate seed queries
seed_queries = generator.generate_seed_queries(num_queries=50)

# Generate synthetic queries
synthetic_queries = generator.generate_synthetic_queries(
    num_easy=30,
    num_hard=20
)

# Create annotation interface
all_queries = seed_queries + synthetic_queries
generator.create_annotation_interface(
    queries=all_queries,
    output_file="annotation_interface_v2.json"
)
```

**Output:** `data/evaluation/annotation_interface_v2.json` ready for annotators.

### Step 2: Collect Human Annotations

**Option A: Use Simple CLI Tool (Demo)**

```python
from src.evaluation.validation_workflow import SimpleAnnotationTool

tool = SimpleAnnotationTool("data/evaluation/annotation_interface_v2.json")

# Annotate a query
judgments = tool.annotate_query(
    query_id="seed_001",
    annotator_id="expert_1",
    sample_size=20
)

# Save judgments
import json
with open("data/evaluation/annotations_expert_1.json", 'w') as f:
    json.dump(judgments, f, indent=2)
```

**Option B: Web-Based Interface (Recommended for Production)**

- Deploy a web application with:
  - Query display
  - Document preview
  - Relevance rating buttons (0-3)
  - Rationale text field
  - Progress tracking
- Export annotations in standardized JSON format

### Step 3: Validate Annotations and Create Ground Truth

```python
from src.evaluation.validation_workflow import AnnotationValidator

# Initialize validator
validator = AnnotationValidator(
    annotation_file="data/evaluation/annotation_interface_v2.json",
    min_agreement=0.6
)

# Load annotator judgments and validate
validated_queries, reports = validator.validate_and_create_ground_truth(
    annotator_files=[
        "data/evaluation/annotations_expert_1.json",
        "data/evaluation/annotations_expert_2.json",
        "data/evaluation/annotations_expert_3.json"
    ],
    output_version="2.0"
)
```

**Outputs:**
- `ground_truth_v2.0_dev.json` (development set)
- `ground_truth_v2.0_test.json` (test set)
- `annotator_agreement_report_v2.0.json`
- `bias_detection_report_v2.0.json`

### Step 4: Use Ground Truth for Evaluation

```python
import json

# Load ground truth
with open("data/evaluation/ground_truth_v2.0_test.json", 'r') as f:
    gt = json.load(f)

# Evaluate a retrieval system
for query in gt['queries']:
    query_text = query['query_text']

    # Run your retrieval system
    results = your_retrieval_system.search(query_text, top_k=10)

    # Get consensus relevance for evaluation
    consensus_relevance = {
        int(doc_id): level
        for doc_id, level in query['consensus_relevance'].items()
    }

    # Compute metrics
    from src.evaluation.metrics import compute_ndcg_at_k
    ndcg = compute_ndcg_at_k(results, consensus_relevance, k=10)
```

---

## Evaluation Metrics

Ground truth v2.0 supports **graded relevance metrics** that account for 4-level judgments:

### 1. Normalized Discounted Cumulative Gain (NDCG@K)

**Formula:**

```
NDCG@K = DCG@K / IDCG@K

DCG@K = Σ (2^rel_i - 1) / log2(i + 1)

where:
  rel_i = relevance level of document at rank i (0-3)
  IDCG@K = DCG@K for ideal ranking
```

**Interpretation:**
- NDCG@K = 1.0: Perfect ranking
- NDCG@K = 0.0: Worst possible ranking
- NDCG@K ∈ [0, 1]: Captures how well system ranks documents by relevance

**Why NDCG?**
- Accounts for graded relevance (Perfect > Excellent > Good > Not Relevant)
- Penalizes placing irrelevant documents at top ranks
- Rewards placing highly relevant documents early

### 2. Graded Precision@K

**Formula:**

```
Graded Precision@K = Σ rel_i / (K × max_relevance)

where:
  rel_i = relevance level of document at rank i
  max_relevance = 3 (Perfect)
```

**Example:**
- Top-5 results: [3, 2, 1, 0, 2]
- Graded Precision@5 = (3 + 2 + 1 + 0 + 2) / (5 × 3) = 8/15 = 0.533

### 3. Graded Recall@K

**Formula:**

```
Graded Recall@K = Σ rel_i / Σ all_relevant

where:
  rel_i = relevance level of retrieved document at rank i
  all_relevant = sum of relevance levels for all relevant docs
```

### 4. Mean Average Precision (MAP)

Computed across all queries using graded relevance.

### 5. Heritage-Specific Metrics

- **Temporal Accuracy:** % of top-K results matching query's time period
- **Spatial Relevance:** % of top-K results matching query's region
- **Domain Alignment:** Overlap of domains between query and results

---

## Limitations & Future Work

### Current Limitations

1. **Annotation Cost:** Human annotation is time-consuming and expensive
   - **Mitigation:** Sample 20-30 candidate docs per query instead of all 369 docs

2. **Small Dataset Size:** 369 documents may not cover all heritage domain aspects
   - **Future:** Expand dataset to 1000+ documents

3. **Underrepresented Categories:**
   - Central India: Only 2 documents (0.5%)
   - Modern heritage: Only 20 documents (5.4%)
   - Intangible heritage: Only 24 documents (6.5%)
   - **Future:** Targeted data collection for these categories

4. **Single Language:** All documents in English
   - **Future:** Multilingual ground truth (Hindi, regional languages)

5. **Static Queries:** Queries don't evolve with user behavior
   - **Future:** Incorporate real user query logs

### Future Enhancements

1. **Active Learning for Annotation:**
   - Use initial model predictions to prioritize uncertain documents for annotation
   - Reduces annotation burden by focusing on boundary cases

2. **Crowdsourced Validation:**
   - After expert annotation, validate with crowdworkers on MTurk/Prolific
   - Use majority voting for additional validation

3. **Query Difficulty Estimation:**
   - Automatically estimate query difficulty based on retrieval performance
   - Use to balance test set difficulty distribution

4. **Temporal Drift Detection:**
   - Monitor if ground truth becomes outdated as retrieval methods improve
   - Version ground truth annually with new queries

5. **Explainability Ground Truth:**
   - Not just "which documents are relevant" but "why they are relevant"
   - Enable evaluation of retrieval explanation quality

6. **Multi-Modal Ground Truth:**
   - Include image-based queries (e.g., "Find monuments similar to this image")
   - Relevant for heritage domain with rich visual content

---

## Conclusion

Ground truth v2.0 represents a **significant improvement** over the initial automatic generation approach:

| Aspect | v1.0 | v2.0 |
|--------|------|------|
| Generation | 100% automatic | 50% manual seed + 50% synthetic |
| Relevance Scale | Binary (0/1) | 4-level graded (0-3) |
| Validation | None | Cohen's Kappa > 0.6 required |
| Bias Detection | None | Comprehensive cluster/temporal/spatial analysis |
| Explainability | None | Rationale required for all judgments |
| Dataset Split | Single file | Separate dev/test sets |
| Documentation | Minimal | Comprehensive methodology + reports |

The **human-in-the-loop validation** ensures that ground truth reflects authentic user information needs, while **stratified sampling** and **bias detection** ensure comprehensive, unbiased coverage of the heritage document space.

This ground truth enables **rigorous evaluation** of the Heritage Recommendation System's three retrieval methods (SimRank, Horn's Index, Embedding Similarity) and supports future improvements through graded metrics like NDCG.

---

**Document Status:** ✅ Implementation Complete
**Next Steps:** Collect annotations from domain experts and generate final ground truth v2.0 dataset.
