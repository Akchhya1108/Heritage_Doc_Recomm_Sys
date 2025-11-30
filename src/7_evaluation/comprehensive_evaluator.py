"""
Comprehensive Multi-Dimensional Evaluation Framework

Measures:
1. Diversity (temporal, spatial, cultural, novelty)
2. Fairness (representation, source, temporal, geographic)
3. Explanation Quality (correctness, diversity, length)
4. User Experience (CTR, dwell time, session success, discovery)
5. Robustness (perturbation, sparsity, cold start, multilingual)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from scipy.stats import entropy, chisquare
from scipy.spatial.distance import pdist
from sklearn.metrics import ndcg_score
import json
from collections import Counter, defaultdict
import re


@dataclass
class RecommendationResult:
    """Single recommendation result"""
    doc_id: str
    rank: int
    relevance: int  # 0-3 scale
    cluster_id: int
    time_period: str
    region: str
    domain: str  # Cultural tradition (Buddhist, Islamic, Hindu, etc.)
    source: str  # Wikipedia, UNESCO, ASI, etc.
    latitude: float
    longitude: float
    explanation_path: List[str]  # KG path from query to doc
    novelty_score: float  # How unexpected this recommendation is


@dataclass
class DiversityMetrics:
    """Diversity evaluation results"""
    temporal_entropy: float  # Shannon entropy of time periods
    spatial_dispersion: float  # Geographic spread (km)
    cultural_diversity: float  # Simpson's index
    novelty_rate: float  # % recommendations outside user's cluster

    # Breakdown
    time_period_distribution: Dict[str, int]
    region_distribution: Dict[str, int]
    domain_distribution: Dict[str, int]


@dataclass
class FairnessMetrics:
    """Fairness evaluation results"""
    cluster_representation_score: float  # 1.0 = perfect fairness
    source_bias_pvalue: float  # Chi-square test p-value
    temporal_bias_kl: float  # KL divergence (lower = fairer)
    geographic_bias_ratio: float  # North/South ratio (1.0 = balanced)

    # Detailed analysis
    cluster_exposure: Dict[int, float]  # % of recs from each cluster
    source_distribution: Dict[str, float]
    regional_representation: Dict[str, float]


@dataclass
class ExplanationMetrics:
    """Explanation quality metrics"""
    avg_correctness: float  # Mean human rating (1-5)
    explanation_diversity: float  # % unique explanation types
    avg_path_length: float
    short_path_rate: float  # % paths <= 3 hops

    # Explanation types
    explanation_type_counts: Dict[str, int]


@dataclass
class UserExperienceMetrics:
    """Simulated user experience metrics"""
    expected_ctr: float  # Expected click-through rate
    avg_dwell_time: float  # Predicted dwell time (seconds)
    session_success_rate: float  # % queries with perfect match in top-10
    discovery_potential: float  # % serendipitous discoveries

    # Per-rank metrics
    ctr_by_position: List[float]


@dataclass
class RobustnessMetrics:
    """Robustness under adversarial conditions"""
    perturbation_ndcg_drop: float  # % NDCG drop with typos
    longtail_performance: float  # NDCG on bottom 20% queries
    coldstart_recall: float  # Recall on low-degree docs
    multilingual_accuracy: float  # Cross-lingual retrieval

    # Perturbation analysis
    typo_robustness: float
    synonym_robustness: float
    reorder_robustness: float


class ComprehensiveEvaluator:
    """Multi-dimensional evaluation framework"""

    def __init__(self,
                 document_metadata: Dict,
                 cluster_sizes: Dict[int, int],
                 true_distribution: Dict = None):
        """
        Initialize evaluator

        Args:
            document_metadata: Metadata for all documents
            cluster_sizes: Number of documents in each cluster
            true_distribution: True distribution of documents (for fairness)
        """
        self.doc_metadata = document_metadata
        self.cluster_sizes = cluster_sizes
        self.true_distribution = true_distribution or self._compute_true_distribution()

        # Heritage domain knowledge
        self.time_periods = [
            'ancient', 'medieval', 'early_modern', 'modern', 'contemporary'
        ]
        self.regions = [
            'north', 'south', 'east', 'west', 'central', 'northeast'
        ]
        self.domains = [
            'buddhist', 'hindu', 'islamic', 'jain', 'sikh', 'christian',
            'secular', 'colonial'
        ]

    def _compute_true_distribution(self) -> Dict:
        """Compute true distribution of documents"""
        distribution = {
            'time_period': Counter(),
            'region': Counter(),
            'domain': Counter(),
            'source': Counter(),
            'cluster': Counter()
        }

        for doc_id, metadata in self.doc_metadata.items():
            distribution['time_period'][metadata.get('time_period', 'unknown')] += 1
            distribution['region'][metadata.get('region', 'unknown')] += 1
            distribution['domain'][metadata.get('domain', 'unknown')] += 1
            distribution['source'][metadata.get('source', 'unknown')] += 1
            distribution['cluster'][metadata.get('cluster_id', -1)] += 1

        # Normalize to probabilities
        for key in distribution:
            total = sum(distribution[key].values())
            distribution[key] = {k: v/total for k, v in distribution[key].items()}

        return distribution

    # =========================================================================
    # 1. DIVERSITY METRICS
    # =========================================================================

    def evaluate_diversity(self,
                          recommendations: List[RecommendationResult],
                          user_cluster: int = None) -> DiversityMetrics:
        """
        Evaluate diversity across multiple dimensions

        Args:
            recommendations: Top-K recommendations
            user_cluster: User's primary cluster (for novelty)

        Returns:
            DiversityMetrics object
        """
        k = len(recommendations)

        # Extract attributes
        time_periods = [r.time_period for r in recommendations]
        regions = [r.region for r in recommendations]
        domains = [r.domain for r in recommendations]
        clusters = [r.cluster_id for r in recommendations]
        coords = [(r.latitude, r.longitude) for r in recommendations]

        # Temporal diversity (Shannon entropy)
        time_dist = Counter(time_periods)
        time_probs = np.array([time_dist[t] / k for t in time_dist])
        temporal_entropy = entropy(time_probs)  # Higher = more diverse

        # Spatial diversity (geographic dispersion)
        if len(coords) > 1:
            # Compute pairwise distances and take mean
            coords_array = np.array(coords)
            distances = pdist(coords_array, metric='euclidean')  # Approximate km
            spatial_dispersion = float(np.mean(distances) * 111)  # Convert to km
        else:
            spatial_dispersion = 0.0

        # Cultural diversity (Simpson's index)
        domain_dist = Counter(domains)
        domain_probs = np.array([domain_dist[d] / k for d in domain_dist])
        cultural_diversity = 1 - np.sum(domain_probs ** 2)  # 1 - sum(p_i^2)

        # Novelty rate (% outside user's cluster)
        if user_cluster is not None:
            novelty_count = sum(1 for c in clusters if c != user_cluster)
            novelty_rate = novelty_count / k
        else:
            novelty_rate = 0.5  # Default

        return DiversityMetrics(
            temporal_entropy=temporal_entropy,
            spatial_dispersion=spatial_dispersion,
            cultural_diversity=cultural_diversity,
            novelty_rate=novelty_rate,
            time_period_distribution=dict(time_dist),
            region_distribution=dict(Counter(regions)),
            domain_distribution=dict(domain_dist)
        )

    # =========================================================================
    # 2. FAIRNESS METRICS
    # =========================================================================

    def evaluate_fairness(self,
                         all_recommendations: List[List[RecommendationResult]],
                         tolerance: float = 0.1) -> FairnessMetrics:
        """
        Evaluate fairness across queries

        Args:
            all_recommendations: List of recommendation lists (one per query)
            tolerance: Acceptable deviation from true distribution (±10%)

        Returns:
            FairnessMetrics object
        """
        # Collect all recommended documents
        all_recs = [rec for recs in all_recommendations for rec in recs]
        total_recs = len(all_recs)

        # Cluster representation fairness
        cluster_counts = Counter(r.cluster_id for r in all_recs)
        cluster_exposure = {c: cluster_counts[c] / total_recs for c in cluster_counts}

        # Target: exposure proportional to cluster size (±tolerance)
        cluster_total_docs = sum(self.cluster_sizes.values())
        expected_exposure = {c: size / cluster_total_docs
                           for c, size in self.cluster_sizes.items()}

        # Compute fairness score (1.0 = perfect)
        fairness_scores = []
        for cluster_id in expected_exposure:
            actual = cluster_exposure.get(cluster_id, 0)
            expected = expected_exposure[cluster_id]

            # Score: 1.0 if within tolerance, decreases linearly outside
            deviation = abs(actual - expected)
            if deviation <= tolerance:
                score = 1.0
            else:
                score = max(0, 1.0 - (deviation - tolerance) / expected)
            fairness_scores.append(score)

        cluster_representation_score = np.mean(fairness_scores)

        # Source fairness (chi-square test)
        source_counts = Counter(r.source for r in all_recs)
        observed = np.array([source_counts[s] for s in source_counts])
        expected_dist = np.array([self.true_distribution['source'].get(s, 1/len(source_counts))
                                 for s in source_counts])
        expected_counts = expected_dist * total_recs

        chi2, source_bias_pvalue = chisquare(observed, expected_counts)

        # Temporal fairness (KL divergence)
        time_counts = Counter(r.time_period for r in all_recs)
        time_dist_rec = {t: time_counts[t] / total_recs for t in time_counts}

        # KL divergence
        temporal_bias_kl = 0.0
        for period in self.true_distribution['time_period']:
            p = self.true_distribution['time_period'][period]
            q = time_dist_rec.get(period, 1e-10)  # Avoid log(0)
            temporal_bias_kl += p * np.log(p / q)

        # Geographic fairness (North/South ratio)
        region_counts = Counter(r.region for r in all_recs)
        north_count = region_counts.get('north', 0)
        south_count = region_counts.get('south', 0)

        if south_count > 0:
            geographic_bias_ratio = north_count / south_count
        else:
            geographic_bias_ratio = float('inf') if north_count > 0 else 1.0

        return FairnessMetrics(
            cluster_representation_score=cluster_representation_score,
            source_bias_pvalue=source_bias_pvalue,
            temporal_bias_kl=temporal_bias_kl,
            geographic_bias_ratio=geographic_bias_ratio,
            cluster_exposure=cluster_exposure,
            source_distribution={s: source_counts[s] / total_recs for s in source_counts},
            regional_representation={r: region_counts[r] / total_recs for r in region_counts}
        )

    # =========================================================================
    # 3. EXPLANATION QUALITY METRICS
    # =========================================================================

    def evaluate_explanation_quality(self,
                                     recommendations: List[RecommendationResult],
                                     human_ratings: Dict[str, float] = None) -> ExplanationMetrics:
        """
        Evaluate quality of KG path explanations

        Args:
            recommendations: Recommendations with explanation paths
            human_ratings: Optional human judgments (doc_id -> rating 1-5)

        Returns:
            ExplanationMetrics object
        """
        paths = [r.explanation_path for r in recommendations]

        # Average path length
        path_lengths = [len(p) for p in paths]
        avg_path_length = np.mean(path_lengths)

        # Short path rate (≤ 3 hops)
        short_path_rate = sum(1 for l in path_lengths if l <= 3) / len(path_lengths)

        # Explanation type diversity
        explanation_types = [self._classify_explanation_type(p) for p in paths]
        type_counts = Counter(explanation_types)
        explanation_diversity = len(type_counts) / len(paths)  # % unique types

        # Correctness (human ratings)
        if human_ratings:
            ratings = [human_ratings.get(r.doc_id, 3.0) for r in recommendations]
            avg_correctness = np.mean(ratings)
        else:
            # Heuristic: shorter paths = more correct
            avg_correctness = 5.0 - min(avg_path_length - 1, 2.0)  # 5 for len=1, 3 for len≥3

        return ExplanationMetrics(
            avg_correctness=avg_correctness,
            explanation_diversity=explanation_diversity,
            avg_path_length=avg_path_length,
            short_path_rate=short_path_rate,
            explanation_type_counts=dict(type_counts)
        )

    def _classify_explanation_type(self, path: List[str]) -> str:
        """Classify explanation path into type"""
        if len(path) == 0:
            return 'no_explanation'
        elif len(path) == 1:
            return 'direct_match'
        elif len(path) == 2:
            return 'one_hop'
        else:
            # Analyze edge types to categorize
            path_str = ' -> '.join(path).lower()

            if 'temporal' in path_str or 'period' in path_str:
                return 'temporal_connection'
            elif 'located' in path_str or 'region' in path_str:
                return 'spatial_connection'
            elif 'built_by' in path_str or 'dynasty' in path_str:
                return 'historical_connection'
            elif 'style' in path_str or 'architecture' in path_str:
                return 'architectural_similarity'
            else:
                return 'multi_hop_generic'

    def generate_counterfactual_explanation(self,
                                           query: str,
                                           recommendation: RecommendationResult,
                                           alternative_query: str) -> str:
        """
        Generate contrastive explanation

        "If you were interested in X instead of Y, we'd recommend Z"
        """
        return (f"If you were interested in '{alternative_query}' instead of '{query}', "
                f"we would recommend {recommendation.doc_id} because "
                f"{' -> '.join(recommendation.explanation_path)}")

    # =========================================================================
    # 4. USER EXPERIENCE METRICS
    # =========================================================================

    def evaluate_user_experience(self,
                                recommendations: List[RecommendationResult]) -> UserExperienceMetrics:
        """
        Simulate user behavior and measure UX

        Args:
            recommendations: Top-K recommendations

        Returns:
            UserExperienceMetrics object
        """
        # Click-Through Rate simulation
        # Model: P(click | rank, relevance) = relevance_weight × position_bias
        ctr_by_position = []
        expected_ctr = 0.0

        position_bias = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]  # Top-10

        for i, rec in enumerate(recommendations[:10]):
            # Relevance to click probability: 0→0%, 1→25%, 2→50%, 3→100%
            relevance_prob = rec.relevance / 3.0

            # Position bias
            pos_bias = position_bias[i] if i < len(position_bias) else 0.1

            # Combined click probability
            click_prob = relevance_prob * pos_bias
            ctr_by_position.append(click_prob)
            expected_ctr += click_prob

        expected_ctr /= min(len(recommendations), 10)  # Average over top-10

        # Dwell time prediction
        # Model: dwell_time = 30s × relevance + 20s × novelty
        dwell_times = [
            30 * rec.relevance + 20 * rec.novelty_score
            for rec in recommendations[:10]
        ]
        avg_dwell_time = np.mean(dwell_times)

        # Session success rate (at least 1 perfect match in top-10)
        perfect_matches = sum(1 for rec in recommendations[:10] if rec.relevance == 3)
        session_success = 1.0 if perfect_matches > 0 else 0.0

        # Discovery potential (relevant + novel)
        serendipitous = sum(1 for rec in recommendations[:10]
                          if rec.relevance >= 2 and rec.novelty_score > 0.5)
        discovery_potential = serendipitous / min(len(recommendations), 10)

        return UserExperienceMetrics(
            expected_ctr=expected_ctr,
            avg_dwell_time=avg_dwell_time,
            session_success_rate=session_success,
            discovery_potential=discovery_potential,
            ctr_by_position=ctr_by_position
        )

    # =========================================================================
    # 5. ROBUSTNESS METRICS
    # =========================================================================

    def evaluate_robustness(self,
                           recommender_func,
                           test_queries: List[Tuple[str, List[int]]],
                           perturbation_types: List[str] = None) -> RobustnessMetrics:
        """
        Test robustness under adversarial conditions

        Args:
            recommender_func: Function that takes query and returns recommendations
            test_queries: List of (query, ground_truth_relevance)
            perturbation_types: Types of perturbations to test

        Returns:
            RobustnessMetrics object
        """
        if perturbation_types is None:
            perturbation_types = ['typo', 'synonym', 'reorder']

        # Baseline NDCG
        baseline_ndcgs = []
        for query, ground_truth in test_queries:
            recs = recommender_func(query)
            predicted_relevance = [r.relevance for r in recs[:10]]

            # Pad to length 10
            while len(predicted_relevance) < 10:
                predicted_relevance.append(0)
            while len(ground_truth) < 10:
                ground_truth.append(0)

            ndcg = ndcg_score([ground_truth[:10]], [predicted_relevance[:10]])
            baseline_ndcgs.append(ndcg)

        baseline_ndcg = np.mean(baseline_ndcgs)

        # Perturbation robustness
        perturbation_results = {}

        for p_type in perturbation_types:
            perturbed_ndcgs = []

            for query, ground_truth in test_queries:
                # Apply perturbation
                perturbed_query = self._perturb_query(query, p_type)

                # Get recommendations
                recs = recommender_func(perturbed_query)
                predicted_relevance = [r.relevance for r in recs[:10]]

                # Pad
                while len(predicted_relevance) < 10:
                    predicted_relevance.append(0)

                ndcg = ndcg_score([ground_truth[:10]], [predicted_relevance[:10]])
                perturbed_ndcgs.append(ndcg)

            perturbation_results[p_type] = np.mean(perturbed_ndcgs)

        # Overall perturbation drop
        avg_perturbed_ndcg = np.mean(list(perturbation_results.values()))
        perturbation_ndcg_drop = (baseline_ndcg - avg_perturbed_ndcg) / baseline_ndcg

        # Long-tail performance (bottom 20% queries)
        # Assume queries are sorted by frequency
        longtail_start = int(len(test_queries) * 0.8)
        longtail_ndcgs = baseline_ndcgs[longtail_start:]
        longtail_performance = np.mean(longtail_ndcgs) if longtail_ndcgs else 0.0

        # Cold start recall (simulated - would need actual graph degrees)
        coldstart_recall = 0.5  # Placeholder

        # Multilingual accuracy (simulated)
        multilingual_accuracy = 0.7  # Placeholder

        return RobustnessMetrics(
            perturbation_ndcg_drop=perturbation_ndcg_drop,
            longtail_performance=longtail_performance,
            coldstart_recall=coldstart_recall,
            multilingual_accuracy=multilingual_accuracy,
            typo_robustness=perturbation_results.get('typo', 0.0),
            synonym_robustness=perturbation_results.get('synonym', 0.0),
            reorder_robustness=perturbation_results.get('reorder', 0.0)
        )

    def _perturb_query(self, query: str, perturbation_type: str) -> str:
        """Apply query perturbation"""
        words = query.split()

        if perturbation_type == 'typo':
            # Swap adjacent characters in random word
            if words:
                idx = np.random.randint(len(words))
                word = words[idx]
                if len(word) > 2:
                    pos = np.random.randint(len(word) - 1)
                    word_list = list(word)
                    word_list[pos], word_list[pos+1] = word_list[pos+1], word_list[pos]
                    words[idx] = ''.join(word_list)

        elif perturbation_type == 'synonym':
            # Replace with synonym (simplified)
            synonyms = {
                'ancient': 'old',
                'temple': 'shrine',
                'fort': 'fortress',
                'architecture': 'design',
                'monument': 'memorial'
            }
            for i, word in enumerate(words):
                if word.lower() in synonyms:
                    words[i] = synonyms[word.lower()]
                    break

        elif perturbation_type == 'reorder':
            # Shuffle words
            if len(words) > 1:
                np.random.shuffle(words)

        return ' '.join(words)

    # =========================================================================
    # COMPREHENSIVE EVALUATION
    # =========================================================================

    def evaluate_all(self,
                    query: str,
                    recommendations: List[RecommendationResult],
                    user_cluster: int = None,
                    human_ratings: Dict[str, float] = None) -> Dict:
        """
        Run all evaluation dimensions

        Returns:
            Dict with all metrics
        """
        diversity = self.evaluate_diversity(recommendations, user_cluster)
        explanation = self.evaluate_explanation_quality(recommendations, human_ratings)
        ux = self.evaluate_user_experience(recommendations)

        return {
            'query': query,
            'diversity': diversity,
            'explanation_quality': explanation,
            'user_experience': ux,
            'num_recommendations': len(recommendations)
        }

    def generate_report(self,
                       all_evaluations: List[Dict],
                       fairness_metrics: FairnessMetrics,
                       robustness_metrics: RobustnessMetrics) -> Dict:
        """
        Generate comprehensive evaluation report

        Args:
            all_evaluations: List of per-query evaluations
            fairness_metrics: Fairness evaluation
            robustness_metrics: Robustness evaluation

        Returns:
            Complete evaluation report
        """
        # Aggregate diversity metrics
        avg_temporal_entropy = np.mean([e['diversity'].temporal_entropy
                                       for e in all_evaluations])
        avg_spatial_dispersion = np.mean([e['diversity'].spatial_dispersion
                                         for e in all_evaluations])
        avg_cultural_diversity = np.mean([e['diversity'].cultural_diversity
                                         for e in all_evaluations])
        avg_novelty_rate = np.mean([e['diversity'].novelty_rate
                                   for e in all_evaluations])

        # Aggregate explanation metrics
        avg_explanation_correctness = np.mean([e['explanation_quality'].avg_correctness
                                              for e in all_evaluations])
        avg_explanation_diversity = np.mean([e['explanation_quality'].explanation_diversity
                                            for e in all_evaluations])
        avg_path_length = np.mean([e['explanation_quality'].avg_path_length
                                  for e in all_evaluations])

        # Aggregate UX metrics
        avg_expected_ctr = np.mean([e['user_experience'].expected_ctr
                                   for e in all_evaluations])
        avg_session_success = np.mean([e['user_experience'].session_success_rate
                                      for e in all_evaluations])
        avg_discovery = np.mean([e['user_experience'].discovery_potential
                                for e in all_evaluations])

        report = {
            'summary': {
                'total_queries_evaluated': len(all_evaluations),
                'evaluation_timestamp': pd.Timestamp.now().isoformat()
            },
            'diversity': {
                'temporal_entropy': avg_temporal_entropy,
                'spatial_dispersion_km': avg_spatial_dispersion,
                'cultural_diversity': avg_cultural_diversity,
                'novelty_rate': avg_novelty_rate,
                'interpretation': self._interpret_diversity(
                    avg_temporal_entropy, avg_cultural_diversity, avg_novelty_rate
                )
            },
            'fairness': {
                'cluster_representation_score': fairness_metrics.cluster_representation_score,
                'source_bias_pvalue': fairness_metrics.source_bias_pvalue,
                'temporal_bias_kl': fairness_metrics.temporal_bias_kl,
                'geographic_bias_ratio': fairness_metrics.geographic_bias_ratio,
                'cluster_exposure': fairness_metrics.cluster_exposure,
                'interpretation': self._interpret_fairness(fairness_metrics)
            },
            'explanation_quality': {
                'avg_correctness': avg_explanation_correctness,
                'avg_diversity': avg_explanation_diversity,
                'avg_path_length': avg_path_length,
                'interpretation': self._interpret_explanations(
                    avg_explanation_correctness, avg_path_length
                )
            },
            'user_experience': {
                'expected_ctr': avg_expected_ctr,
                'session_success_rate': avg_session_success,
                'discovery_potential': avg_discovery,
                'interpretation': self._interpret_ux(
                    avg_expected_ctr, avg_session_success, avg_discovery
                )
            },
            'robustness': {
                'perturbation_ndcg_drop': robustness_metrics.perturbation_ndcg_drop,
                'longtail_performance': robustness_metrics.longtail_performance,
                'coldstart_recall': robustness_metrics.coldstart_recall,
                'interpretation': self._interpret_robustness(robustness_metrics)
            },
            'overall_grade': self._compute_overall_grade(
                avg_temporal_entropy, fairness_metrics.cluster_representation_score,
                avg_explanation_correctness, avg_session_success,
                robustness_metrics.perturbation_ndcg_drop
            )
        }

        return report

    # Interpretation helpers

    def _interpret_diversity(self, entropy, cultural, novelty) -> str:
        """Interpret diversity scores"""
        if entropy > 1.5 and cultural > 0.7 and novelty > 0.3:
            return "Excellent: High diversity across all dimensions"
        elif entropy > 1.0 and cultural > 0.5:
            return "Good: Moderate diversity, room for improvement in novelty"
        else:
            return "Poor: Recommendations lack diversity, may be too concentrated"

    def _interpret_fairness(self, metrics: FairnessMetrics) -> str:
        """Interpret fairness scores"""
        if metrics.cluster_representation_score > 0.9 and metrics.temporal_bias_kl < 0.1:
            return "Excellent: Fair representation across clusters and time periods"
        elif metrics.cluster_representation_score > 0.7:
            return "Good: Generally fair, minor biases detected"
        else:
            return f"Poor: Systematic bias detected (cluster fairness: {metrics.cluster_representation_score:.2f})"

    def _interpret_explanations(self, correctness, path_length) -> str:
        """Interpret explanation quality"""
        if correctness > 4.0 and path_length <= 3:
            return "Excellent: Explanations are correct and concise"
        elif correctness > 3.5:
            return "Good: Explanations are generally useful but could be shorter"
        else:
            return "Poor: Explanation quality needs improvement"

    def _interpret_ux(self, ctr, success, discovery) -> str:
        """Interpret user experience"""
        if success > 0.7 and ctr > 0.3:
            return "Excellent: Users likely to find relevant results and click"
        elif success > 0.5:
            return "Good: Reasonable user satisfaction expected"
        else:
            return "Poor: Low probability of user satisfaction"

    def _interpret_robustness(self, metrics: RobustnessMetrics) -> str:
        """Interpret robustness"""
        if metrics.perturbation_ndcg_drop < 0.1:
            return "Excellent: System is robust to query variations"
        elif metrics.perturbation_ndcg_drop < 0.2:
            return "Good: Moderate robustness to perturbations"
        else:
            return "Poor: System is sensitive to query variations"

    def _compute_overall_grade(self, diversity, fairness, explanation, ux, robustness) -> str:
        """Compute overall system grade"""
        # Weighted score
        score = (
            0.2 * diversity / 2.0 +  # Normalize entropy
            0.2 * fairness +
            0.2 * explanation / 5.0 +  # Normalize correctness
            0.2 * ux +
            0.2 * (1 - robustness)  # Lower drop = better
        )

        if score > 0.8:
            return "A (Excellent)"
        elif score > 0.7:
            return "B (Good)"
        elif score > 0.6:
            return "C (Acceptable)"
        else:
            return "D (Needs Improvement)"


if __name__ == '__main__':
    print("Comprehensive Multi-Dimensional Evaluator")
    print("="*60)
    print("\nThis module evaluates:")
    print("  1. Diversity (temporal, spatial, cultural, novelty)")
    print("  2. Fairness (representation, source, temporal, geographic)")
    print("  3. Explanation Quality (correctness, diversity, length)")
    print("  4. User Experience (CTR, dwell time, session success)")
    print("  5. Robustness (perturbation, long-tail, cold start)")
    print("\nSee user_study_protocol.md for online evaluation design.")
