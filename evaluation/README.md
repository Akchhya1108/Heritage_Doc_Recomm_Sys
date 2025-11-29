# Ground Truth V2.0 - Heritage Document Recommendation System

## Overview

This directory contains a comprehensive **multi-strategy ground truth generation system** for evaluating the Heritage Document Recommendation System. The system implements human-in-the-loop validation, stratified sampling, 4-level relevance grading, and bias detection to create unbiased, realistic test queries with verified relevance judgments.

## 📁 Directory Structure

```
evaluation/
├── README.md                                    # This file
├── ground_truth_methodology.md                  # Detailed methodology documentation
│
data/evaluation/
├── ground_truth_v2.0_dev.json                   # Development set (80% of queries)
├── ground_truth_v2.0_test.json                  # Test set (20% of queries)
├── annotator_agreement_report_v2.0.json         # Inter-annotator agreement metrics
├── bias_detection_report_v2.0.json              # Bias analysis report
├── annotation_interface_v2.json                 # Interface for human annotators
├── annotations_demo_annotator_1.json            # Sample annotations (annotator 1)
├── annotations_demo_annotator_2.json            # Sample annotations (annotator 2)
└── query_metadata_v2.json                       # Query generation metadata

src/7_evaluation/
├── ground_truth_generator.py                    # Main generation system
├── validation_workflow.py                       # Annotation validation workflow
├── run_validation.py                            # End-to-end validation script
└── prepare_metadata.py                          # Metadata preprocessing
```

## 🎯 Key Features

### 1. Multi-Strategy Query Generation
- **50 Seed Queries**: Manually curated based on real user information needs
- **30 Easy Synthetic Queries**: Stratified sampling across all 12 document clusters
- **20 Hard Synthetic Queries**: Complex multi-faceted queries
- **5 Negative Queries**: Out-of-scope queries (should return empty results)

### 2. 4-Level Relevance Grading
- **Perfect (3)**: Exact match, all constraints satisfied
- **Excellent (2)**: Highly relevant, most constraints met
- **Good (1)**: Somewhat relevant, partial match
- **Not Relevant (0)**: Does not address query

### 3. Human-in-the-Loop Validation
- 2-3 domain experts independently judge relevance
- Cohen's Kappa ≥ 0.6 required for query inclusion
- Rationale required for all judgments
- Disagreement tracking and resolution

### 4. Bias Detection
- Cluster distribution bias
- Temporal bias (ancient vs medieval vs modern)
- Spatial bias (regional coverage)
- Graph connectivity vs semantic similarity bias

### 5. Versioned Datasets
- Separate development and test sets
- Complete provenance tracking
- Methodology documentation
- Reproducible generation process

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy pandas scikit-learn
```

### Generate Seed and Synthetic Queries

```bash
python src/7_evaluation/ground_truth_generator.py
```

**Output:**
- `data/evaluation/annotation_interface_v2.json` (71 queries ready for annotation)
- `data/evaluation/query_metadata_v2.json` (query generation metadata)

### Create Sample Annotations (Demo)

```bash
python src/7_evaluation/validation_workflow.py
```

**Output:**
- `data/evaluation/annotations_demo_annotator_1.json`
- `data/evaluation/annotations_demo_annotator_2.json`

### Run Validation and Create Ground Truth

```bash
python src/7_evaluation/run_validation.py
```

**Output:**
- `data/evaluation/ground_truth_v2.0_dev.json` (development set)
- `data/evaluation/ground_truth_v2.0_test.json` (test set)
- `data/evaluation/annotator_agreement_report_v2.0.json`
- `data/evaluation/bias_detection_report_v2.0.json`

## 📖 Detailed Usage

### Step 1: Generate Queries

```python
from src.evaluation.ground_truth_generator import GroundTruthGenerator

# Initialize generator
generator = GroundTruthGenerator(
    data_path="data/processed/heritage_metadata.csv",
    output_dir="data/evaluation"
)

# Generate seed queries (manually curated)
seed_queries = generator.generate_seed_queries(num_queries=50)

# Generate synthetic queries (stratified sampling)
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

### Step 2: Annotate Queries

**For Production**: Use a web-based annotation interface with:
- Query display
- Document preview
- Relevance rating buttons (0-3)
- Rationale text field
- Progress tracking

**For Demo**: Use the simple CLI tool:

```python
from src.evaluation.validation_workflow import SimpleAnnotationTool

tool = SimpleAnnotationTool("data/evaluation/annotation_interface_v2.json")

# Annotate a single query
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

### Step 3: Validate Annotations

```python
from src.evaluation.validation_workflow import AnnotationValidator

# Initialize validator
validator = AnnotationValidator(
    annotation_file="data/evaluation/annotation_interface_v2.json",
    min_agreement=0.6
)

# Validate and create ground truth
validated_queries, reports = validator.validate_and_create_ground_truth(
    annotator_files=[
        "data/evaluation/annotations_expert_1.json",
        "data/evaluation/annotations_expert_2.json",
        "data/evaluation/annotations_expert_3.json"
    ],
    output_version="2.0"
)
```

### Step 4: Use Ground Truth for Evaluation

```python
import json

# Load ground truth
with open("data/evaluation/ground_truth_v2.0_test.json", 'r') as f:
    gt = json.load(f)

# Evaluate your retrieval system
for query in gt['queries']:
    query_text = query['query_text']

    # Run your retrieval system
    results = your_system.search(query_text, top_k=10)

    # Get consensus relevance
    consensus_relevance = {
        int(doc_id): level
        for doc_id, level in query['consensus_relevance'].items()
    }

    # Compute NDCG@10
    from src.evaluation.metrics import compute_ndcg_at_k
    ndcg = compute_ndcg_at_k(results, consensus_relevance, k=10)

    print(f"Query: {query_text}")
    print(f"NDCG@10: {ndcg:.3f}")
```

## 📊 Ground Truth Statistics

### Dataset Size
- **Total Queries**: 71 (before validation)
- **Validated Queries**: Subset with Cohen's Kappa ≥ 0.6
- **Development Set**: 80% of validated queries
- **Test Set**: 20% of validated queries

### Query Distribution

| Query Type | Count | Description |
|------------|-------|-------------|
| Seed (Simple) | 31 | Single clear intent |
| Seed (Moderate) | 11 | Multiple overlapping concepts |
| Seed (Complex) | 8 | Cross-temporal/cultural complexity |
| Synthetic Easy | 1 | Cluster-based stratified sampling |
| Synthetic Hard | 20 | Multi-faceted random generation |
| Negative | 5 | Out-of-scope queries |

### Heritage Coverage

| Dimension | Categories Covered |
|-----------|-------------------|
| Heritage Types | monument, site, artifact, architecture, tradition, art |
| Domains | religious, military, royal, cultural, archaeological, architectural |
| Time Periods | ancient, medieval, modern, unknown |
| Regions | north, south, east, west, central, india, unknown |
| Clusters | All 12 clusters proportionally represented |

## 📈 Evaluation Metrics

### Supported Metrics

#### 1. NDCG@K (Normalized Discounted Cumulative Gain)
- Accounts for graded relevance (0-3)
- Penalizes placing irrelevant docs at top ranks
- Range: [0, 1], higher is better

#### 2. Graded Precision@K
- Weighted by relevance level
- Formula: `Σ rel_i / (K × max_relevance)`

#### 3. Graded Recall@K
- Coverage of relevant documents
- Formula: `Σ rel_i / Σ all_relevant`

#### 4. MAP (Mean Average Precision)
- Computed across all queries using graded relevance

#### 5. Heritage-Specific Metrics
- **Temporal Accuracy**: % matching query's time period
- **Spatial Relevance**: % matching query's region
- **Domain Alignment**: Overlap of domains

## 🔬 Quality Validation

### Inter-Annotator Agreement

The system uses **Cohen's Kappa** to measure agreement between annotators:

| Kappa Range | Interpretation | Action |
|-------------|----------------|--------|
| κ < 0.0 | Poor (worse than chance) | Discard query |
| 0.0 ≤ κ < 0.2 | Slight agreement | Re-annotate |
| 0.2 ≤ κ < 0.4 | Fair agreement | Consider re-annotation |
| 0.4 ≤ κ < 0.6 | Moderate agreement | Review disagreements |
| **κ ≥ 0.6** | **Substantial agreement** | **✅ Accept** |
| 0.8 ≤ κ ≤ 1.0 | Almost perfect | ✅ Accept |

### Consensus Aggregation

When multiple annotators judge the same query-document pair:

1. **Median Rating**: Use median of all ratings (robust to outliers)
2. **Disagreement Filter**: Exclude if `max(ratings) - min(ratings) > 2`
3. **Relevance Threshold**: Only include if consensus level > 0

## 🎯 Bias Detection

The system automatically detects and reports biases:

### Cluster Bias
- Checks if certain clusters are over/underrepresented
- Bias Ratio = Actual Proportion / Expected Proportion
- Recommends adding queries for underrepresented clusters

### Temporal Bias
- Checks if time periods are proportionally represented
- Flags if modern heritage is underrepresented

### Spatial Bias
- Checks regional coverage
- Accounts for dataset imbalances (e.g., Central India has only 2 docs)

### Method Bias
- Detects if ground truth favors graph-based or embedding-based methods
- Ensures both retrieval approaches are fairly evaluated

## 📝 Annotation Guidelines

### Relevance Levels

**Perfect (3):**
- ✅ All heritage types match
- ✅ All domains match
- ✅ Time period matches (or both unspecified)
- ✅ Region matches (or both unspecified)
- ✅ Document directly answers query

**Excellent (2):**
- ✅ Primary heritage type matches
- ✅ At least one domain matches
- ⚠️ Time period or region may differ
- ✅ Document very useful for query

**Good (1):**
- ✅ At least one heritage type or domain matches
- ⚠️ May have wrong time/region but conceptually related
- ✅ Document provides tangential information

**Not Relevant (0):**
- ❌ No heritage type match
- ❌ Wrong domain entirely
- ❌ Different cultural/geographic context
- ❌ Does not address query intent

### Rationale Requirement

For **every judgment**, provide a brief rationale:

- **Perfect**: "Exact match - Mughal fort in North India, all constraints satisfied"
- **Excellent**: "Highly relevant - Mughal fort but region not specified"
- **Good**: "Partial match - Discusses Mughal architecture but not specifically forts"
- **Not Relevant**: "Wrong domain - This is a South Indian temple, not a Mughal fort"

## 🔄 Versioning

### Version Format
- `v<major>.<minor>`
- **Major**: Methodology changes (e.g., v1.0 → v2.0)
- **Minor**: Query additions/refinements (e.g., v2.0 → v2.1)

### Changelog

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2024-XX-XX | Initial automatic generation (100 queries, binary relevance) |
| **v2.0** | **2025-11-29** | **Human-validated, 4-level relevance, stratified sampling, bias detection** |

## 📚 Documentation

- **[ground_truth_methodology.md](ground_truth_methodology.md)**: Complete methodology documentation
- **[data/evaluation/annotator_agreement_report_v2.0.json](../data/evaluation/annotator_agreement_report_v2.0.json)**: Inter-annotator agreement analysis
- **[data/evaluation/bias_detection_report_v2.0.json](../data/evaluation/bias_detection_report_v2.0.json)**: Bias detection results
- **[data/evaluation/query_metadata_v2.json](../data/evaluation/query_metadata_v2.json)**: Query generation provenance

## 🛠️ Extending the System

### Adding New Queries

1. Edit `ground_truth_generator.py` → `generate_seed_queries()`
2. Add query dict with all required fields
3. Re-run generation and validation workflow

### Adjusting Agreement Threshold

```python
validator = AnnotationValidator(
    annotation_file="...",
    min_agreement=0.7  # Stricter threshold
)
```

### Custom Bias Detection

Edit `ground_truth_generator.py` → `detect_bias()` to add new bias checks.

## 🐛 Troubleshooting

### "No such file: heritage_metadata.csv"

Run metadata preparation first:

```bash
python src/7_evaluation/prepare_metadata.py
```

### "TypeError: Object of type int64 is not JSON serializable"

This has been fixed in v2.0 with custom `NpEncoder`. Update to latest code.

### Low Inter-Annotator Agreement

- Review annotation guidelines with annotators
- Provide more examples of each relevance level
- Focus re-annotation on low-agreement queries

## 📞 Contact

For questions or issues, please:
1. Check [ground_truth_methodology.md](ground_truth_methodology.md) for detailed documentation
2. Review the code comments in `ground_truth_generator.py`
3. Open an issue in the project repository

## 📄 License

This ground truth generation system is part of the Heritage Document Recommendation System project.

---

**Last Updated**: 2025-11-29
**Version**: 2.0
**Status**: ✅ Production Ready
