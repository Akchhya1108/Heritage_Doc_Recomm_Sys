"""
Training Script for Learning to Rank Models

Trains LTR models on ground truth data and compares performance.
"""

import sys
import os
import json
import pickle
import numpy as np
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_extractor import FeatureExtractor, QueryDocFeatures, create_training_dataset
from query_classifier import QueryTypeClassifier, create_synthetic_training_data
from learned_ranker import LearnedRanker, RankingWeights
from ensemble_ranker import compare_fusion_methods, RankedDocument


def train_query_classifier(output_dir: str = 'models/ranker'):
    """Train and save query type classifier"""
    print("\n" + "="*80)
    print("TRAINING QUERY TYPE CLASSIFIER")
    print("="*80)

    # Create synthetic training data
    queries, entities, labels = create_synthetic_training_data()

    # Train classifier
    classifier = QueryTypeClassifier()
    classifier.train(queries, entities, labels)

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    classifier.save(f'{output_dir}/query_classifier.pkl')

    return classifier


def train_ltr_models(training_features: List[QueryDocFeatures],
                     output_dir: str = 'models/ranker') -> Dict[str, LearnedRanker]:
    """
    Train all LTR models and compare

    Args:
        training_features: List of training samples
        output_dir: Directory to save models

    Returns:
        Dict of trained models
    """
    print("\n" + "="*80)
    print("TRAINING LTR MODELS")
    print("="*80)

    models = {}

    # Train LambdaMART
    print("\n" + "-"*80)
    lambdamart = LearnedRanker(model_type='lambdamart')
    lambdamart.train(training_features, n_folds=5, optimize_per_query_type=True)
    lambdamart.save(f'{output_dir}/lambdamart_model.pkl')
    models['lambdamart'] = lambdamart

    # Train RankNet
    print("\n" + "-"*80)
    ranknet = LearnedRanker(model_type='ranknet')
    ranknet.train(training_features, n_folds=5, optimize_per_query_type=True)
    ranknet.save(f'{output_dir}/ranknet_model.pkl')
    models['ranknet'] = ranknet

    # Train ListNet
    print("\n" + "-"*80)
    listnet = LearnedRanker(model_type='listnet')
    listnet.train(training_features, n_folds=5, optimize_per_query_type=True)
    listnet.save(f'{output_dir}/listnet_model.pkl')
    models['listnet'] = listnet

    return models


def export_learned_weights(models: Dict[str, LearnedRanker],
                          output_file: str = 'models/ranker/learned_weights.json'):
    """
    Export learned weights to JSON for easy inspection

    Args:
        models: Trained LTR models
        output_file: Output JSON file
    """
    weights_data = {}

    for model_name, model in models.items():
        weights_data[model_name] = {}

        for query_type, weights in model.query_type_weights.items():
            weights_data[model_name][query_type] = {
                'simrank_weight': weights.simrank_weight,
                'horn_index_weight': weights.horn_index_weight,
                'embedding_weight': weights.embedding_weight,
                'confidence': weights.confidence
            }

    # Save to JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(weights_data, f, indent=2)

    print(f"\nLearned weights exported to: {output_file}")


def evaluate_ensemble_methods(training_features: List[QueryDocFeatures],
                              output_file: str = 'evaluation/ltr_comparison.json'):
    """
    Evaluate different ensemble fusion methods

    Args:
        training_features: Training data with ground truth
        output_file: Output JSON file with comparison results
    """
    print("\n" + "="*80)
    print("EVALUATING ENSEMBLE METHODS")
    print("="*80)

    # Group features by query
    query_groups = {}
    for feature in training_features:
        if feature.query_id not in query_groups:
            query_groups[feature.query_id] = []
        query_groups[feature.query_id].append(feature)

    # Evaluate each fusion method
    fusion_results = {
        'cascade': [],
        'rrf': [],
        'borda': [],
        'combmnz': []
    }

    for query_id, features in query_groups.items():
        # Create RankedDocument objects
        documents = [
            RankedDocument(
                doc_id=f.doc_id,
                simrank_score=f.simrank_score,
                horn_score=f.horn_index_score,
                embedding_score=f.embedding_similarity
            )
            for f in features
        ]

        # Ground truth relevance
        ground_truth = {f.doc_id: f.relevance_label for f in features}

        # Compare fusion methods
        ndcg_scores = compare_fusion_methods(documents, ground_truth)

        for method, ndcg in ndcg_scores.items():
            fusion_results[method].append(ndcg)

    # Compute average NDCG for each method
    comparison_results = {}
    for method, scores in fusion_results.items():
        if scores:
            comparison_results[method] = {
                'mean_ndcg': float(np.mean(scores)),
                'std_ndcg': float(np.std(scores)),
                'num_queries': len(scores)
            }

    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)

    print("\nEnsemble Method Comparison:")
    print("-"*60)
    for method, results in sorted(comparison_results.items(), key=lambda x: x[1]['mean_ndcg'], reverse=True):
        print(f"{method.upper():15} NDCG@10: {results['mean_ndcg']:.4f} (+/- {results['std_ndcg']:.4f})")

    print(f"\nResults saved to: {output_file}")


def generate_training_summary(models: Dict[str, LearnedRanker],
                             training_features: List[QueryDocFeatures],
                             output_file: str = 'models/ranker/training_summary.md'):
    """
    Generate comprehensive training summary report

    Args:
        models: Trained LTR models
        training_features: Training data
        output_file: Output markdown file
    """
    print("\n" + "="*80)
    print("GENERATING TRAINING SUMMARY")
    print("="*80)

    # Collect statistics
    num_samples = len(training_features)
    num_queries = len(set(f.query_id for f in training_features))
    query_types = {}
    for f in training_features:
        qtype = f.query_type_encoding
        query_types[qtype] = query_types.get(qtype, 0) + 1

    # Write summary
    with open(output_file, 'w') as f:
        f.write("# Learning to Rank - Training Summary\n\n")

        # Dataset statistics
        f.write("## Dataset Statistics\n\n")
        f.write(f"- **Total training samples**: {num_samples}\n")
        f.write(f"- **Unique queries**: {num_queries}\n")
        f.write(f"- **Average samples per query**: {num_samples / num_queries:.1f}\n\n")

        f.write("### Query Type Distribution\n\n")
        type_names = {0: 'simple_keyword', 1: 'entity_focused', 2: 'concept_focused', 3: 'complex_nlp'}
        for qtype, count in sorted(query_types.items()):
            f.write(f"- **{type_names[qtype]}**: {count} samples ({100*count/num_samples:.1f}%)\n")

        # Model performance
        f.write("\n## Model Performance\n\n")
        f.write("Cross-validation NDCG@10 scores:\n\n")
        f.write("| Model | Mean NDCG@10 | Std Dev |\n")
        f.write("|-------|--------------|----------|\n")

        # Note: Would need to extract CV scores from training
        for model_name in models:
            f.write(f"| {model_name.upper()} | See training logs | - |\n")

        # Learned weights
        f.write("\n## Learned Weights\n\n")
        for model_name, model in models.items():
            f.write(f"### {model_name.upper()}\n\n")
            f.write("| Query Type | SimRank | Horn's Index | Embedding | Confidence |\n")
            f.write("|------------|---------|--------------|-----------|------------|\n")

            for qtype, weights in model.query_type_weights.items():
                f.write(f"| {qtype} | {weights.simrank_weight:.3f} | "
                       f"{weights.horn_index_weight:.3f} | {weights.embedding_weight:.3f} | "
                       f"{weights.confidence:.3f} |\n")
            f.write("\n")

        # Feature importance
        f.write("## Feature Importance\n\n")
        f.write("Top features from LambdaMART model:\n\n")

        if 'lambdamart' in models:
            importances = models['lambdamart'].model.feature_importances_
            feature_names = QueryDocFeatures.feature_names()

            top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]

            f.write("| Rank | Feature | Importance |\n")
            f.write("|------|---------|------------|\n")
            for rank, (name, imp) in enumerate(top_features, 1):
                f.write(f"| {rank} | {name} | {imp:.4f} |\n")

        # Recommendations
        f.write("\n## Recommendations\n\n")
        f.write("Based on learned weights:\n\n")

        # Analyze weight patterns
        f.write("1. **SimRank effectiveness**: Graph structure is ")
        avg_simrank = np.mean([w.simrank_weight for m in models.values() for w in m.query_type_weights.values()])
        if avg_simrank > 0.4:
            f.write("highly important (avg weight: {:.2f})\n".format(avg_simrank))
        else:
            f.write("moderately important (avg weight: {:.2f})\n".format(avg_simrank))

        f.write("2. **Horn's Index effectiveness**: Entity importance is ")
        avg_horn = np.mean([w.horn_index_weight for m in models.values() for w in m.query_type_weights.values()])
        if avg_horn > 0.3:
            f.write("significant (avg weight: {:.2f})\n".format(avg_horn))
        else:
            f.write("minor (avg weight: {:.2f})\n".format(avg_horn))

        f.write("3. **Embedding effectiveness**: Semantic similarity is ")
        avg_emb = np.mean([w.embedding_weight for m in models.values() for w in m.query_type_weights.values()])
        if avg_emb > 0.4:
            f.write("critical (avg weight: {:.2f})\n".format(avg_emb))
        else:
            f.write("supplementary (avg weight: {:.2f})\n".format(avg_emb))

        f.write("\n## Integration Guide\n\n")
        f.write("To use learned weights in recommender:\n\n")
        f.write("```python\n")
        f.write("from src.5_ranking.learned_ranker import LearnedRanker\n")
        f.write("from src.5_ranking.query_classifier import QueryTypeClassifier\n\n")
        f.write("# Load models\n")
        f.write("classifier = QueryTypeClassifier()\n")
        f.write("classifier.load('models/ranker/query_classifier.pkl')\n\n")
        f.write("ranker = LearnedRanker(model_type='lambdamart')\n")
        f.write("ranker.load('models/ranker/lambdamart_model.pkl')\n\n")
        f.write("# Classify query\n")
        f.write("query_type_id, query_type, confidence = classifier.predict(query_text, entities)\n\n")
        f.write("# Get adaptive weights\n")
        f.write("weights = ranker.get_weights_for_query_type(query_type)\n\n")
        f.write("# Apply to ranking\n")
        f.write("final_score = (weights.simrank_weight * simrank_score +\n")
        f.write("              weights.horn_index_weight * horn_score +\n")
        f.write("              weights.embedding_weight * embedding_score)\n")
        f.write("```\n")

    print(f"Training summary saved to: {output_file}")


def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print("LEARNING TO RANK - TRAINING PIPELINE")
    print("="*80)

    # Step 1: Train query classifier
    classifier = train_query_classifier()

    # Step 2: Create training dataset
    # Note: This requires ground truth annotations and recommender output
    # For now, we'll create a placeholder
    print("\n" + "="*80)
    print("CREATING TRAINING DATASET")
    print("="*80)
    print("\nNote: This step requires:")
    print("  1. Ground truth annotations: data/evaluation/ground_truth_v2.0_dev.json")
    print("  2. Recommender output: data/evaluation/recommender_results.json")
    print("\nTo generate recommender results, run:")
    print("  python src/1_recommender/recommender.py --evaluate")

    # Check if files exist
    gt_file = 'data/evaluation/ground_truth_v2.0_dev.json'
    rec_file = 'data/evaluation/recommender_results.json'

    if os.path.exists(gt_file) and os.path.exists(rec_file):
        print("\nFiles found! Creating training dataset...")

        # Initialize feature extractor
        extractor = FeatureExtractor(
            kg_file='data/knowledge_graph/heritage_kg.gpickle',
            document_metadata_file='data/processed/document_metadata.json',
            entity_importance_file='data/entity_importance/computed_scores.json'
        )

        # Create training dataset
        training_file = 'data/evaluation/ltr_training_features.pkl'
        training_features = create_training_dataset(
            ground_truth_file=gt_file,
            recommender_results_file=rec_file,
            feature_extractor=extractor,
            output_file=training_file
        )

        # Step 3: Train LTR models
        if len(training_features) > 0:
            models = train_ltr_models(training_features)

            # Step 4: Export learned weights
            export_learned_weights(models)

            # Step 5: Evaluate ensemble methods
            evaluate_ensemble_methods(training_features)

            # Step 6: Generate summary
            generate_training_summary(models, training_features)

            print("\n" + "="*80)
            print("TRAINING COMPLETE")
            print("="*80)
            print("\nOutput files:")
            print("  - models/ranker/query_classifier.pkl")
            print("  - models/ranker/lambdamart_model.pkl")
            print("  - models/ranker/ranknet_model.pkl")
            print("  - models/ranker/listnet_model.pkl")
            print("  - models/ranker/learned_weights.json")
            print("  - models/ranker/training_summary.md")
            print("  - evaluation/ltr_comparison.json")

        else:
            print("\nError: No training features extracted")

    else:
        print("\nSkipping LTR training (missing required files)")
        print("\nQuery classifier has been trained and saved.")
        print("\nTo complete LTR training:")
        print("  1. Generate ground truth: python src/7_evaluation/ground_truth_generator.py")
        print("  2. Run recommender with evaluation mode")
        print("  3. Re-run this script")


if __name__ == '__main__':
    main()
