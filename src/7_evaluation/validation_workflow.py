"""
Human-in-the-Loop Validation Workflow

This module provides tools for annotators to judge relevance and for
validation of ground truth quality through inter-annotator agreement.
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from ground_truth_generator import (
    GroundTruthGenerator,
    GroundTruthQuery,
    RelevanceJudgment,
    NpEncoder
)


class AnnotationValidator:
    """
    Validates annotation quality and creates final ground truth.

    Responsibilities:
    1. Load and validate annotator judgments
    2. Compute inter-annotator agreement (Cohen's Kappa)
    3. Identify low-agreement queries that need re-annotation
    4. Aggregate judgments into consensus ground truth
    5. Filter queries based on agreement threshold (>0.6)
    """

    def __init__(self, annotation_file: str, min_agreement: float = 0.6):
        self.annotation_file = Path(annotation_file)
        self.min_agreement = min_agreement
        self.generator = GroundTruthGenerator()

        # Load annotation data
        with open(self.annotation_file, 'r') as f:
            self.annotation_data = json.load(f)

        self.queries = self.annotation_data.get('queries', [])
        self.documents = self.annotation_data.get('documents', [])

    def load_annotator_judgments(self, annotator_files: List[str]) -> Dict:
        """
        Load judgments from multiple annotators.

        Args:
            annotator_files: List of JSON files with annotator judgments

        Returns:
            Dict[query_id -> Dict[doc_id -> List[RelevanceJudgment]]]
        """

        all_judgments = defaultdict(lambda: defaultdict(list))

        for annotator_file in annotator_files:
            with open(annotator_file, 'r') as f:
                annotator_data = json.load(f)

            annotator_id = annotator_data.get('annotator_id', 'unknown')
            timestamp = annotator_data.get('timestamp', datetime.now().isoformat())

            for query in annotator_data.get('queries', []):
                query_id = query['query_id']

                for doc_id_str, judgment in query.get('judgments', {}).items():
                    doc_id = int(doc_id_str)
                    relevance_level = judgment.get('relevance_level', 0)
                    rationale = judgment.get('rationale', '')

                    rel_judgment = RelevanceJudgment(
                        annotator_id=annotator_id,
                        document_id=doc_id,
                        relevance_level=relevance_level,
                        rationale=rationale,
                        timestamp=timestamp
                    )

                    all_judgments[query_id][doc_id].append(rel_judgment)

        return all_judgments

    def validate_and_create_ground_truth(self, annotator_files: List[str],
                                          output_version: str = "2.0") -> Tuple[List[GroundTruthQuery], Dict]:
        """
        Main validation workflow:
        1. Load all annotator judgments
        2. Compute inter-annotator agreement
        3. Filter queries with agreement < threshold
        4. Create final ground truth with consensus judgments

        Args:
            annotator_files: List of annotator judgment files
            output_version: Version string for output

        Returns:
            Tuple of (validated_queries, agreement_report)
        """

        print("=" * 80)
        print("ANNOTATION VALIDATION WORKFLOW")
        print("=" * 80)

        # Step 1: Load judgments
        print("\n[1/5] Loading annotator judgments...")
        all_judgments = self.load_annotator_judgments(annotator_files)
        print(f"  ✓ Loaded judgments from {len(annotator_files)} annotators")
        print(f"  ✓ {len(all_judgments)} queries annotated")

        # Step 2: Compute agreement
        print("\n[2/5] Computing inter-annotator agreement...")
        agreement_report = self.generator.compute_inter_annotator_agreement(
            self._convert_judgments_for_agreement(all_judgments)
        )
        print(f"  ✓ Overall Cohen's Kappa: {agreement_report['overall_kappa']:.3f}")
        print(f"  ✓ Kappa range: [{agreement_report.get('kappa_min', 0):.3f}, {agreement_report.get('kappa_max', 0):.3f}]")
        print(f"  ✓ Disagreement cases: {len(agreement_report['disagreement_cases'])}")

        # Step 3: Filter queries by agreement
        print(f"\n[3/5] Filtering queries (min agreement: {self.min_agreement})...")
        validated_queries = []
        low_agreement_queries = []

        for query_dict in self.queries:
            query_id = query_dict['query_id']
            query_kappa = agreement_report['query_level_kappa'].get(query_id, 0.0)

            if query_kappa >= self.min_agreement:
                # Create GroundTruthQuery object
                gt_query = self._create_ground_truth_query(
                    query_dict,
                    all_judgments.get(query_id, {}),
                    query_kappa,
                    output_version
                )
                validated_queries.append(gt_query)
            else:
                low_agreement_queries.append({
                    'query_id': query_id,
                    'kappa': query_kappa,
                    'query_text': query_dict['query_text']
                })

        print(f"  ✓ Validated: {len(validated_queries)} queries")
        print(f"  ✓ Low agreement (needs re-annotation): {len(low_agreement_queries)}")

        if low_agreement_queries:
            print(f"\n  Low agreement queries:")
            for laq in low_agreement_queries[:5]:
                print(f"    - {laq['query_id']} (κ={laq['kappa']:.3f}): {laq['query_text'][:60]}...")

        # Step 4: Detect bias
        print(f"\n[4/5] Detecting bias in ground truth...")
        bias_report = self.generator.detect_bias(validated_queries)
        print(f"  ✓ Cluster bias analyzed: {len(bias_report['cluster_bias'])} clusters")
        print(f"  ✓ Recommendations: {len(bias_report['recommendations'])}")

        if bias_report['recommendations']:
            print(f"\n  Top bias recommendations:")
            for rec in bias_report['recommendations'][:3]:
                print(f"    - {rec}")

        # Step 5: Save results
        print(f"\n[5/5] Saving ground truth and reports...")

        # Split into dev and test sets (80/20)
        np.random.shuffle(validated_queries)
        split_idx = int(0.8 * len(validated_queries))
        dev_queries = validated_queries[:split_idx]
        test_queries = validated_queries[split_idx:]

        dev_file = self.generator.save_ground_truth(dev_queries, split='dev', version=output_version)
        test_file = self.generator.save_ground_truth(test_queries, split='test', version=output_version)

        # Save agreement report
        agreement_file = self.generator.output_dir / f"annotator_agreement_report_v{output_version}.json"
        with open(agreement_file, 'w') as f:
            json.dump({
                'overall_statistics': {
                    'overall_kappa': agreement_report['overall_kappa'],
                    'kappa_std': agreement_report.get('kappa_std', 0),
                    'kappa_range': [agreement_report.get('kappa_min', 0), agreement_report.get('kappa_max', 0)],
                    'num_annotators': len(annotator_files),
                    'num_queries': len(all_judgments),
                    'validated_queries': len(validated_queries),
                    'low_agreement_queries': len(low_agreement_queries)
                },
                'annotator_pairs': agreement_report['annotator_pairs'],
                'query_level_kappa': agreement_report['query_level_kappa'],
                'disagreement_cases': agreement_report['disagreement_cases'][:50],  # First 50
                'low_agreement_queries': low_agreement_queries,
                'validation_date': datetime.now().isoformat()
            }, f, indent=2, cls=NpEncoder)

        # Save bias report
        bias_file = self.generator.output_dir / f"bias_detection_report_v{output_version}.json"
        with open(bias_file, 'w') as f:
            json.dump(bias_report, f, indent=2, cls=NpEncoder)

        print(f"\n  ✓ Development set: {dev_file} ({len(dev_queries)} queries)")
        print(f"  ✓ Test set: {test_file} ({len(test_queries)} queries)")
        print(f"  ✓ Agreement report: {agreement_file}")
        print(f"  ✓ Bias report: {bias_file}")

        print("\n" + "=" * 80)
        print("VALIDATION COMPLETE")
        print("=" * 80)

        return validated_queries, {
            'agreement_report': agreement_report,
            'bias_report': bias_report,
            'low_agreement_queries': low_agreement_queries
        }

    def _convert_judgments_for_agreement(self, all_judgments: Dict) -> Dict:
        """Convert judgments to format for agreement calculation"""

        converted = {}
        for query_id, doc_judgments in all_judgments.items():
            converted[query_id] = {}
            for doc_id, judgments in doc_judgments.items():
                converted[query_id][doc_id] = [j.relevance_level for j in judgments]

        return converted

    def _create_ground_truth_query(self, query_dict: Dict, doc_judgments: Dict,
                                    kappa: float, version: str) -> GroundTruthQuery:
        """Create GroundTruthQuery object from validated annotations"""

        # Aggregate judgments
        consensus_relevance = self.generator.aggregate_relevance_judgments(
            query_dict['query_id'],
            doc_judgments
        )

        # Compute cluster distribution
        cluster_distribution = defaultdict(int)
        for doc_id in consensus_relevance.keys():
            if doc_id < len(self.generator.documents):
                doc = self.generator.documents.iloc[doc_id]
                if 'cluster' in doc:
                    cluster_distribution[int(doc['cluster'])] += 1

        # Count annotators
        num_annotators = 0
        for judgments in doc_judgments.values():
            num_annotators = max(num_annotators, len(judgments))

        return GroundTruthQuery(
            query_id=query_dict['query_id'],
            query_text=query_dict['query_text'],
            query_type=query_dict['query_type'],
            heritage_types=query_dict['expected_characteristics']['heritage_types'],
            domains=query_dict['expected_characteristics']['domains'],
            time_period=query_dict['expected_characteristics'].get('time_period'),
            region=query_dict['expected_characteristics'].get('region'),
            complexity=query_dict.get('complexity', 'moderate'),
            relevance_judgments=doc_judgments,
            consensus_relevance=consensus_relevance,
            inter_annotator_agreement=kappa,
            num_annotators=num_annotators,
            cluster_distribution=dict(cluster_distribution),
            creation_date=datetime.now().isoformat(),
            version=version,
            expected_result_size_range=query_dict.get('expected_result_size_range', (0, 100)),
            rationale=query_dict.get('rationale', '')
        )


class SimpleAnnotationTool:
    """
    Simple command-line annotation tool for quick manual judgment.

    For full annotation, use a web-based interface.
    This is useful for demonstration or small-scale annotation.
    """

    def __init__(self, annotation_interface_file: str):
        self.interface_file = Path(annotation_interface_file)

        with open(self.interface_file, 'r') as f:
            self.data = json.load(f)

        self.queries = self.data['queries']
        self.documents = pd.DataFrame(self.data['documents'])

    def annotate_query(self, query_id: str, annotator_id: str,
                        sample_size: int = 20) -> Dict:
        """
        Annotate a single query with sampled candidate documents.

        Args:
            query_id: Query to annotate
            annotator_id: Annotator identifier
            sample_size: Number of documents to sample for annotation

        Returns:
            Dictionary of judgments
        """

        query = next((q for q in self.queries if q['query_id'] == query_id), None)
        if not query:
            raise ValueError(f"Query {query_id} not found")

        print("=" * 80)
        print(f"QUERY: {query['query_text']}")
        print(f"Expected: {query['expected_characteristics']}")
        print("=" * 80)

        # Sample documents (mix of likely relevant and random)
        candidate_docs = query['candidate_documents']
        sampled_docs = np.random.choice(candidate_docs, min(sample_size, len(candidate_docs)), replace=False)

        judgments = {}

        for doc_id in sampled_docs:
            doc = self.documents.iloc[doc_id]

            print(f"\n--- Document {doc_id} ---")
            print(f"Title: {doc.get('title', 'N/A')}")
            print(f"Heritage Types: {doc.get('heritage_types', [])}")
            print(f"Domains: {doc.get('domains', [])}")
            print(f"Time Period: {doc.get('time_period', 'N/A')}")
            print(f"Region: {doc.get('region', 'N/A')}")

            print("\nRelevance levels:")
            print("  0 = Not Relevant")
            print("  1 = Good (partial match)")
            print("  2 = Excellent (strong match)")
            print("  3 = Perfect (exact match)")

            # Get judgment
            while True:
                try:
                    level = int(input("Relevance level (0-3): "))
                    if level not in [0, 1, 2, 3]:
                        raise ValueError
                    break
                except (ValueError, EOFError):
                    print("Invalid input. Please enter 0, 1, 2, or 3.")

            if level > 0:
                rationale = input("Brief rationale: ")
            else:
                rationale = "Not relevant"

            judgments[doc_id] = {
                'relevance_level': level,
                'rationale': rationale
            }

            print(f"✓ Recorded: Level {level}")

        return {
            'annotator_id': annotator_id,
            'timestamp': datetime.now().isoformat(),
            'queries': [{
                'query_id': query_id,
                'judgments': {str(k): v for k, v in judgments.items()}
            }]
        }


def create_sample_annotations(num_queries: int = 10):
    """
    Create sample annotations for demonstration purposes.

    This simulates 2 annotators with reasonable agreement.
    """

    print("=" * 80)
    print("CREATING SAMPLE ANNOTATIONS (DEMONSTRATION)")
    print("=" * 80)

    interface_file = "data/evaluation/annotation_interface_v2.json"

    if not Path(interface_file).exists():
        print(f"Error: {interface_file} not found. Run ground_truth_generator.py first.")
        return

    with open(interface_file, 'r') as f:
        data = json.load(f)

    queries = data['queries'][:num_queries]
    documents = pd.DataFrame(data['documents'])

    # Simulate 2 annotators
    for annotator_num in [1, 2]:
        annotator_id = f"demo_annotator_{annotator_num}"
        output_file = f"data/evaluation/annotations_{annotator_id}.json"

        annotation_data = {
            'annotator_id': annotator_id,
            'timestamp': datetime.now().isoformat(),
            'queries': []
        }

        for query in queries:
            query_id = query['query_id']
            expected = query['expected_characteristics']

            # Sample 30 documents per query
            candidate_docs = np.random.choice(
                query['candidate_documents'],
                min(30, len(query['candidate_documents'])),
                replace=False
            )

            judgments = {}

            for doc_id_raw in candidate_docs:
                doc_id = int(doc_id_raw)  # Convert to Python int
                doc = documents.iloc[doc_id]

                # Simulate relevance judgment based on feature matching
                relevance_score = 0

                # Heritage type match
                doc_types = doc.get('heritage_types', [])
                if isinstance(doc_types, str):
                    doc_types = eval(doc_types) if doc_types.startswith('[') else []

                type_match = len(set(doc_types) & set(expected['heritage_types'])) > 0
                if type_match:
                    relevance_score += 1

                # Domain match
                doc_domains = doc.get('domains', [])
                if isinstance(doc_domains, str):
                    doc_domains = eval(doc_domains) if doc_domains.startswith('[') else []

                domain_match = len(set(doc_domains) & set(expected['domains'])) > 0
                if domain_match:
                    relevance_score += 1

                # Time period match
                if expected.get('time_period') and doc.get('time_period') == expected['time_period']:
                    relevance_score += 1

                # Region match
                if expected.get('region') and doc.get('region') == expected['region']:
                    relevance_score += 1

                # Add some noise for realism
                noise = np.random.randint(-1, 2)  # -1, 0, or 1
                final_level = max(0, min(3, relevance_score + noise))

                # Annotator 2 is slightly more generous
                if annotator_num == 2 and final_level > 0:
                    final_level = min(3, final_level + np.random.choice([0, 1], p=[0.7, 0.3]))

                judgments[str(doc_id)] = {
                    'relevance_level': int(final_level),  # Convert to Python int
                    'rationale': f"Matches {relevance_score}/4 expected characteristics" if final_level > 0 else "No significant match"
                }

            annotation_data['queries'].append({
                'query_id': query_id,
                'judgments': judgments
            })

        # Save
        with open(output_file, 'w') as f:
            json.dump(annotation_data, f, indent=2, cls=NpEncoder)

        print(f"✓ Created sample annotations: {output_file}")
        print(f"  Annotator: {annotator_id}")
        print(f"  Queries: {len(queries)}")
        print(f"  Avg judgments per query: {np.mean([len(q['judgments']) for q in annotation_data['queries']]):.1f}")

    print("\n" + "=" * 80)
    print("SAMPLE ANNOTATIONS CREATED")
    print("=" * 80)
    print("\nNext: Run validation workflow with:")
    print("  validator = AnnotationValidator('data/evaluation/annotation_interface_v2.json')")
    print("  validator.validate_and_create_ground_truth([")
    print("      'data/evaluation/annotations_demo_annotator_1.json',")
    print("      'data/evaluation/annotations_demo_annotator_2.json'")
    print("  ])")


if __name__ == "__main__":
    # Demo workflow
    create_sample_annotations(num_queries=15)
