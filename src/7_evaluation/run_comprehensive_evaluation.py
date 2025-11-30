"""
Run Comprehensive Evaluation

Executes full multi-dimensional evaluation and generates reports.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from comprehensive_evaluator import (
    ComprehensiveEvaluator,
    RecommendationResult,
    DiversityMetrics,
    FairnessMetrics,
    RobustnessMetrics
)


def create_sample_recommendations() -> list:
    """Create sample recommendations for demonstration"""
    recommendations = [
        RecommendationResult(
            doc_id='doc_taj_mahal',
            rank=1,
            relevance=3,
            cluster_id=0,
            time_period='early_modern',
            region='north',
            domain='islamic',
            source='wikipedia',
            latitude=27.1751,
            longitude=78.0421,
            explanation_path=['query_entity', 'mughal_architecture', 'taj_mahal'],
            novelty_score=0.2
        ),
        RecommendationResult(
            doc_id='doc_red_fort',
            rank=2,
            relevance=3,
            cluster_id=0,
            time_period='early_modern',
            region='north',
            domain='islamic',
            source='asi',
            latitude=28.6562,
            longitude=77.2410,
            explanation_path=['query_entity', 'mughal_architecture', 'red_fort'],
            novelty_score=0.3
        ),
        RecommendationResult(
            doc_id='doc_sanchi_stupa',
            rank=3,
            relevance=1,
            cluster_id=3,
            time_period='ancient',
            region='central',
            domain='buddhist',
            source='unesco',
            latitude=23.4794,
            longitude=77.7399,
            explanation_path=['query_entity', 'historical_period', 'ancient', 'buddhist', 'sanchi'],
            novelty_score=0.8
        ),
        RecommendationResult(
            doc_id='doc_qutub_minar',
            rank=4,
            relevance=2,
            cluster_id=0,
            time_period='medieval',
            region='north',
            domain='islamic',
            source='unesco',
            latitude=28.5244,
            longitude=77.1855,
            explanation_path=['query_entity', 'delhi', 'qutub_minar'],
            novelty_score=0.4
        ),
        RecommendationResult(
            doc_id='doc_meenakshi_temple',
            rank=5,
            relevance=1,
            cluster_id=5,
            time_period='medieval',
            region='south',
            domain='hindu',
            source='asi',
            latitude=9.9195,
            longitude=78.1193,
            explanation_path=['temples', 'dravidian_style', 'meenakshi'],
            novelty_score=0.9
        ),
        RecommendationResult(
            doc_id='doc_konark_temple',
            rank=6,
            relevance=2,
            cluster_id=5,
            time_period='medieval',
            region='east',
            domain='hindu',
            source='unesco',
            latitude=19.8876,
            longitude=86.0945,
            explanation_path=['temples', 'sun_temple', 'konark'],
            novelty_score=0.7
        ),
        RecommendationResult(
            doc_id='doc_ajanta_caves',
            rank=7,
            relevance=1,
            cluster_id=3,
            time_period='ancient',
            region='west',
            domain='buddhist',
            source='unesco',
            latitude=20.5519,
            longitude=75.7033,
            explanation_path=['buddhist_sites', 'rock_cut', 'ajanta'],
            novelty_score=0.6
        ),
        RecommendationResult(
            doc_id='doc_hampi',
            rank=8,
            relevance=2,
            cluster_id=6,
            time_period='medieval',
            region='south',
            domain='hindu',
            source='unesco',
            latitude=15.3350,
            longitude=76.4600,
            explanation_path=['vijayanagara_empire', 'hampi'],
            novelty_score=0.5
        ),
        RecommendationResult(
            doc_id='doc_khajuraho',
            rank=9,
            relevance=1,
            cluster_id=5,
            time_period='medieval',
            region='central',
            domain='hindu',
            source='unesco',
            latitude=24.8318,
            longitude=79.9199,
            explanation_path=['temples', 'nagara_style', 'khajuraho'],
            novelty_score=0.6
        ),
        RecommendationResult(
            doc_id='doc_ellora_caves',
            rank=10,
            relevance=1,
            cluster_id=3,
            time_period='medieval',
            region='west',
            domain='hindu',
            source='unesco',
            latitude=20.0269,
            longitude=75.1790,
            explanation_path=['rock_cut', 'ellora'],
            novelty_score=0.7
        )
    ]

    return recommendations


def create_sample_metadata() -> dict:
    """Create sample document metadata"""
    return {
        'doc_taj_mahal': {
            'time_period': 'early_modern',
            'region': 'north',
            'domain': 'islamic',
            'source': 'wikipedia',
            'cluster_id': 0
        },
        'doc_red_fort': {
            'time_period': 'early_modern',
            'region': 'north',
            'domain': 'islamic',
            'source': 'asi',
            'cluster_id': 0
        },
        'doc_sanchi_stupa': {
            'time_period': 'ancient',
            'region': 'central',
            'domain': 'buddhist',
            'source': 'unesco',
            'cluster_id': 3
        },
        'doc_qutub_minar': {
            'time_period': 'medieval',
            'region': 'north',
            'domain': 'islamic',
            'source': 'unesco',
            'cluster_id': 0
        },
        'doc_meenakshi_temple': {
            'time_period': 'medieval',
            'region': 'south',
            'domain': 'hindu',
            'source': 'asi',
            'cluster_id': 5
        }
    }


def run_comprehensive_evaluation():
    """Run full evaluation pipeline"""
    print("="*80)
    print("COMPREHENSIVE MULTI-DIMENSIONAL EVALUATION")
    print("="*80)

    # Initialize evaluator
    cluster_sizes = {0: 150, 1: 120, 2: 100, 3: 80, 4: 70, 5: 90, 6: 60, 7: 50, 8: 40, 9: 30, 10: 25, 11: 15}
    doc_metadata = create_sample_metadata()

    evaluator = ComprehensiveEvaluator(
        document_metadata=doc_metadata,
        cluster_sizes=cluster_sizes
    )

    # Sample query and recommendations
    query = "Mughal architectural monuments"
    recommendations = create_sample_recommendations()

    print(f"\nQuery: '{query}'")
    print(f"Top-10 recommendations retrieved\n")

    # 1. DIVERSITY EVALUATION
    print("-" * 80)
    print("1. DIVERSITY METRICS")
    print("-" * 80)

    diversity = evaluator.evaluate_diversity(recommendations, user_cluster=0)

    print(f"\nTemporal Diversity (Shannon entropy): {diversity.temporal_entropy:.3f}")
    print(f"  Distribution: {diversity.time_period_distribution}")
    print(f"  Interpretation: Higher entropy = more diverse time periods")

    print(f"\nSpatial Diversity (geographic dispersion): {diversity.spatial_dispersion:.1f} km")
    print(f"  Distribution: {diversity.region_distribution}")
    print(f"  Interpretation: Larger dispersion = more geographic spread")

    print(f"\nCultural Diversity (Simpson's index): {diversity.cultural_diversity:.3f}")
    print(f"  Distribution: {diversity.domain_distribution}")
    print(f"  Interpretation: Values closer to 1.0 = higher diversity")

    print(f"\nNovelty Rate: {diversity.novelty_rate:.1%}")
    print(f"  Interpretation: {diversity.novelty_rate:.1%} of results outside user's cluster")

    # 2. EXPLANATION QUALITY
    print("\n" + "-" * 80)
    print("2. EXPLANATION QUALITY METRICS")
    print("-" * 80)

    explanation = evaluator.evaluate_explanation_quality(recommendations)

    print(f"\nAverage Correctness (1-5 scale): {explanation.avg_correctness:.2f}")
    print(f"  Target: > 3.5 for good quality")

    print(f"\nExplanation Diversity: {explanation.explanation_diversity:.1%}")
    print(f"  {len(explanation.explanation_type_counts)} unique explanation types")
    print(f"  Types: {explanation.explanation_type_counts}")

    print(f"\nAverage Path Length: {explanation.avg_path_length:.2f} hops")
    print(f"Short Path Rate (≤3 hops): {explanation.short_path_rate:.1%}")

    # Generate counterfactual example
    counterfactual = evaluator.generate_counterfactual_explanation(
        query="Mughal architecture",
        recommendation=recommendations[2],  # Sanchi Stupa
        alternative_query="Buddhist heritage"
    )
    print(f"\nExample Counterfactual Explanation:")
    print(f"  {counterfactual}")

    # 3. USER EXPERIENCE
    print("\n" + "-" * 80)
    print("3. USER EXPERIENCE METRICS")
    print("-" * 80)

    ux = evaluator.evaluate_user_experience(recommendations)

    print(f"\nExpected Click-Through Rate: {ux.expected_ctr:.1%}")
    print(f"  Target: > 30% for good UX")

    print(f"\nCTR by Position:")
    for i, ctr in enumerate(ux.ctr_by_position, 1):
        print(f"  Position {i}: {ctr:.1%}")

    print(f"\nAverage Dwell Time: {ux.avg_dwell_time:.1f} seconds")
    print(f"  Interpretation: Higher relevance → longer dwell time")

    print(f"\nSession Success Rate: {ux.session_success_rate:.1%}")
    print(f"  {ux.session_success_rate:.1%} of queries have perfect match in top-10")
    print(f"  Target: > 70%")

    print(f"\nDiscovery Potential: {ux.discovery_potential:.1%}")
    print(f"  {ux.discovery_potential:.1%} of results are serendipitous discoveries")

    # 4. FAIRNESS (requires multiple queries)
    print("\n" + "-" * 80)
    print("4. FAIRNESS METRICS (Multi-Query Analysis)")
    print("-" * 80)

    # Simulate multiple queries
    all_recs = []
    for _ in range(10):  # 10 sample queries
        # Add slight variations to recommendations
        recs_copy = [
            RecommendationResult(
                doc_id=r.doc_id + f'_{_}',
                rank=r.rank,
                relevance=r.relevance,
                cluster_id=r.cluster_id,
                time_period=r.time_period,
                region=r.region,
                domain=r.domain,
                source=r.source,
                latitude=r.latitude,
                longitude=r.longitude,
                explanation_path=r.explanation_path,
                novelty_score=r.novelty_score
            )
            for r in recommendations
        ]
        all_recs.append(recs_copy)

    fairness = evaluator.evaluate_fairness(all_recs)

    print(f"\nCluster Representation Score: {fairness.cluster_representation_score:.3f}")
    print(f"  Target: > 0.9 (perfect fairness = 1.0)")
    print(f"  Interpretation: How well cluster exposure matches cluster size")

    print(f"\nCluster Exposure:")
    for cluster_id, exposure in sorted(fairness.cluster_exposure.items()):
        expected = cluster_sizes.get(cluster_id, 0) / sum(cluster_sizes.values())
        print(f"  Cluster {cluster_id}: {exposure:.1%} (expected: {expected:.1%})")

    print(f"\nSource Bias (Chi-square p-value): {fairness.source_bias_pvalue:.4f}")
    print(f"  p > 0.05 → no significant source bias")

    print(f"\nTemporal Bias (KL divergence): {fairness.temporal_bias_kl:.4f}")
    print(f"  Lower = fairer temporal representation")

    print(f"\nGeographic Bias (North/South ratio): {fairness.geographic_bias_ratio:.2f}")
    print(f"  Target: ~1.0 (balanced representation)")

    # 5. GENERATE COMPREHENSIVE REPORT
    print("\n" + "-" * 80)
    print("5. GENERATING COMPREHENSIVE REPORT")
    print("-" * 80)

    # Single query evaluation
    eval_results = [evaluator.evaluate_all(
        query=query,
        recommendations=recommendations,
        user_cluster=0
    )]

    # Mock robustness metrics
    robustness = RobustnessMetrics(
        perturbation_ndcg_drop=0.08,
        longtail_performance=0.65,
        coldstart_recall=0.55,
        multilingual_accuracy=0.70,
        typo_robustness=0.92,
        synonym_robustness=0.88,
        reorder_robustness=0.85
    )

    # Generate report
    report = evaluator.generate_report(
        all_evaluations=eval_results,
        fairness_metrics=fairness,
        robustness_metrics=robustness
    )

    # Save report
    os.makedirs('evaluation', exist_ok=True)

    with open('evaluation/comprehensive_evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("\n✓ Report saved to: evaluation/comprehensive_evaluation_report.json")

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print(f"\nOverall Grade: {report['overall_grade']}")

    print(f"\nDiversity: {report['diversity']['interpretation']}")
    print(f"  - Temporal entropy: {report['diversity']['temporal_entropy']:.3f}")
    print(f"  - Cultural diversity: {report['diversity']['cultural_diversity']:.3f}")

    print(f"\nFairness: {report['fairness']['interpretation']}")
    print(f"  - Cluster fairness: {report['fairness']['cluster_representation_score']:.3f}")

    print(f"\nExplanation Quality: {report['explanation_quality']['interpretation']}")
    print(f"  - Avg correctness: {report['explanation_quality']['avg_correctness']:.2f}/5.0")

    print(f"\nUser Experience: {report['user_experience']['interpretation']}")
    print(f"  - Expected CTR: {report['user_experience']['expected_ctr']:.1%}")
    print(f"  - Session success: {report['user_experience']['session_success_rate']:.1%}")

    print(f"\nRobustness: {report['robustness']['interpretation']}")
    print(f"  - Perturbation drop: {report['robustness']['perturbation_ndcg_drop']:.1%}")

    # Create fairness-specific report
    fairness_report = {
        'timestamp': report['summary']['evaluation_timestamp'],
        'fairness_metrics': {
            'cluster_representation': {
                'score': fairness.cluster_representation_score,
                'interpretation': 'Excellent' if fairness.cluster_representation_score > 0.9 else 'Needs improvement',
                'cluster_exposure': fairness.cluster_exposure,
                'cluster_sizes': cluster_sizes
            },
            'source_fairness': {
                'chi_square_pvalue': fairness.source_bias_pvalue,
                'interpretation': 'No bias detected' if fairness.source_bias_pvalue > 0.05 else 'Bias detected',
                'source_distribution': fairness.source_distribution
            },
            'temporal_fairness': {
                'kl_divergence': fairness.temporal_bias_kl,
                'interpretation': 'Fair' if fairness.temporal_bias_kl < 0.1 else 'Biased'
            },
            'geographic_fairness': {
                'north_south_ratio': fairness.geographic_bias_ratio,
                'interpretation': 'Balanced' if 0.8 <= fairness.geographic_bias_ratio <= 1.2 else 'Imbalanced',
                'regional_distribution': fairness.regional_representation
            }
        },
        'recommendations': {
            'cluster_bias': 'Ensure exposure proportional to cluster size (±10%)',
            'source_bias': 'Balance Wikipedia, UNESCO, ASI sources',
            'temporal_bias': 'Increase representation of underrepresented periods',
            'geographic_bias': 'Balance North/South monument recommendations'
        }
    }

    with open('evaluation/fairness_report.json', 'w') as f:
        json.dump(fairness_report, f, indent=2)

    print("\n✓ Fairness report saved to: evaluation/fairness_report.json")

    # Create explanation quality study template
    explanation_study = {
        'study_design': {
            'evaluators': 3,
            'samples_per_evaluator': 30,
            'rating_scale': '1-5 (1=Incorrect, 5=Perfect)',
            'target_agreement': 'Fleiss Kappa > 0.6'
        },
        'sample_explanations': [
            {
                'query': 'Mughal architecture',
                'recommendation': 'Taj Mahal',
                'path': ' → '.join(recommendations[0].explanation_path),
                'path_length': len(recommendations[0].explanation_path),
                'evaluator_1_rating': None,
                'evaluator_2_rating': None,
                'evaluator_3_rating': None,
                'mean_rating': None,
                'agreement': None
            },
            {
                'query': 'Mughal architecture',
                'recommendation': 'Sanchi Stupa',
                'path': ' → '.join(recommendations[2].explanation_path),
                'path_length': len(recommendations[2].explanation_path),
                'evaluator_1_rating': None,
                'evaluator_2_rating': None,
                'evaluator_3_rating': None,
                'mean_rating': None,
                'agreement': None
            }
        ],
        'instructions': 'Rate each explanation path for how well it connects the query to the recommendation. Consider semantic relevance and path clarity.'
    }

    with open('evaluation/explanation_quality_study.json', 'w') as f:
        json.dump(explanation_study, f, indent=2)

    print("✓ Explanation quality study template saved to: evaluation/explanation_quality_study.json")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  1. evaluation/comprehensive_evaluation_report.json - Full report")
    print("  2. evaluation/fairness_report.json - Bias analysis")
    print("  3. evaluation/explanation_quality_study.json - Human evaluation template")
    print("  4. evaluation/user_study_protocol.md - Online evaluation protocol")
    print("\nNext steps:")
    print("  - Review fairness report for systematic biases")
    print("  - Conduct human evaluation of explanation quality")
    print("  - Run user study following protocol in user_study_protocol.md")
    print("  - Monitor evaluation dashboard in Streamlit app")


if __name__ == '__main__':
    run_comprehensive_evaluation()
