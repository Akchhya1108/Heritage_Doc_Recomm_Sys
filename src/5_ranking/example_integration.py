"""
Example Integration: Adaptive Ranking in Heritage Document Recommender

This file shows how to integrate the LTR framework into the existing
heritage document recommender system.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_recommender import AdaptiveRecommender


class EnhancedHeritageRecommender:
    """
    Example of heritage recommender with adaptive LTR ranking

    This is a simplified example showing how to integrate the adaptive
    ranking framework into your existing recommender.
    """

    def __init__(self):
        """Initialize recommender with adaptive ranking"""
        # Initialize existing components
        # self.faiss_index = ...
        # self.kg = ...
        # self.embeddings = ...

        # Add adaptive ranker
        self.adaptive_ranker = AdaptiveRecommender(
            classifier_path='models/ranker/query_classifier.pkl',
            ranker_path='models/ranker/lambdamart_model.pkl',  # Optional
            use_ensemble=True,
            ensemble_method='adaptive'  # Auto-select best method
        )

        print("✓ Heritage recommender initialized with adaptive ranking")

    def search(self, query_text: str, top_k: int = 10):
        """
        Search for heritage documents

        Args:
            query_text: User query
            top_k: Number of results to return

        Returns:
            List of ranked documents
        """
        print(f"\nQuery: '{query_text}'")

        # Step 1: Extract entities (existing code)
        query_entities = self._extract_entities(query_text)
        print(f"Extracted entities: {query_entities}")

        # Step 2: Compute query complexity (new)
        query_complexity = self._compute_query_complexity(query_text)

        # Step 3: Get candidate documents (existing code)
        # - FAISS retrieves top-100 from embeddings
        # - Filter by metadata if needed
        candidates = self._get_candidates(query_text, top_n=100)
        print(f"Retrieved {len(candidates)} candidates")

        # Step 4: Compute component scores for each candidate
        for doc in candidates:
            # Existing score computations
            doc['simrank_score'] = self._compute_simrank(query_entities, doc)
            doc['horn_score'] = self._compute_horn_weighted(query_entities, doc)
            doc['embedding_score'] = self._compute_embedding_similarity(query_text, doc)

        # Step 5: Use adaptive ranking (NEW!)
        ranked_documents = self.adaptive_ranker.rank_documents(
            documents=candidates,
            query_text=query_text,
            query_entities=query_entities,
            query_complexity=query_complexity
        )

        # Step 6: Return top-k results
        return ranked_documents[:top_k]

    def search_with_explanation(self, query_text: str, top_k: int = 10):
        """
        Search with ranking explanations

        Returns:
            (ranked_documents, explanation)
        """
        # Get ranked results
        results = self.search(query_text, top_k)

        # Get weights used
        weights, query_type = self.adaptive_ranker.get_adaptive_weights(
            query_text=query_text,
            query_entities=self._extract_entities(query_text)
        )

        # Generate explanation
        explanation = {
            'query_type': query_type,
            'weights': {
                'simrank': weights.simrank_weight,
                'horn_index': weights.horn_index_weight,
                'embedding': weights.embedding_weight
            },
            'confidence': weights.confidence,
            'top_doc_explanation': self.adaptive_ranker.explain_ranking(results[0], weights)
        }

        return results, explanation

    # Placeholder methods (implement based on your existing code)

    def _extract_entities(self, query_text: str):
        """Extract entities from query (implement with your NER)"""
        # Placeholder: simple keyword extraction
        keywords = query_text.lower().split()
        return [kw for kw in keywords if len(kw) > 3]

    def _compute_query_complexity(self, query_text: str) -> float:
        """Compute linguistic complexity of query"""
        words = query_text.split()

        # Length-based complexity
        length_score = min(len(words) / 20.0, 1.0)

        # Question words indicate complexity
        question_words = ['how', 'why', 'what', 'which', 'where', 'when', 'who']
        has_question = any(qw in query_text.lower() for qw in question_words)

        complexity = (length_score + (0.5 if has_question else 0.0)) / 1.5
        return complexity

    def _get_candidates(self, query_text: str, top_n: int = 100):
        """Get candidate documents (implement with your FAISS index)"""
        # Placeholder: return dummy candidates
        return [
            {
                'doc_id': f'doc_{i:03d}',
                'title': f'Heritage Document {i}',
                'content': 'Sample content...',
                'entities': [],
                'cluster_id': i % 12
            }
            for i in range(top_n)
        ]

    def _compute_simrank(self, query_entities, doc):
        """Compute SimRank score (implement with your KG)"""
        # Placeholder: random score
        import random
        return random.uniform(0.5, 0.95)

    def _compute_horn_weighted(self, query_entities, doc):
        """Compute Horn's Index weighted score (implement with your importance scores)"""
        # Placeholder: random score
        import random
        return random.uniform(0.5, 0.95)

    def _compute_embedding_similarity(self, query_text, doc):
        """Compute embedding similarity (implement with your embeddings)"""
        # Placeholder: random score
        import random
        return random.uniform(0.5, 0.95)


def demo_basic_search():
    """Demo: Basic search with adaptive ranking"""
    print("="*80)
    print("DEMO 1: Basic Search with Adaptive Ranking")
    print("="*80)

    # Initialize recommender
    recommender = EnhancedHeritageRecommender()

    # Test queries
    queries = [
        "mughal architecture",
        "what are the main features of indo-islamic architecture",
        "taj mahal history",
        "ancient buddhist monuments in eastern india"
    ]

    for query in queries:
        results = recommender.search(query, top_k=5)

        print(f"\nTop 3 results:")
        for doc in results[:3]:
            print(f"  {doc['rank']}. {doc['title']} (score: {doc['final_score']:.4f})")
        print()


def demo_search_with_explanation():
    """Demo: Search with ranking explanations"""
    print("\n" + "="*80)
    print("DEMO 2: Search with Explanations")
    print("="*80)

    recommender = EnhancedHeritageRecommender()

    query = "what are the architectural features of mughal monuments"
    results, explanation = recommender.search_with_explanation(query, top_k=5)

    print(f"\nQuery: '{query}'")
    print(f"\nQuery Type: {explanation['query_type']}")
    print(f"Confidence: {explanation['confidence']:.3f}")
    print(f"\nWeights Used:")
    print(f"  SimRank:   {explanation['weights']['simrank']:.3f}")
    print(f"  Horn:      {explanation['weights']['horn_index']:.3f}")
    print(f"  Embedding: {explanation['weights']['embedding']:.3f}")

    print(f"\nTop Result Explanation:")
    print(explanation['top_doc_explanation'])

    print(f"\nAll Results:")
    for doc in results[:5]:
        print(f"  {doc['rank']}. {doc['title']} (score: {doc['final_score']:.4f})")


def demo_weight_comparison():
    """Demo: Compare fixed vs. adaptive weights"""
    print("\n" + "="*80)
    print("DEMO 3: Fixed vs. Adaptive Weights Comparison")
    print("="*80)

    from adaptive_recommender import AdaptiveRecommender

    # Test query
    query_text = "buddhist architecture"
    query_entities = ['buddhist architecture']

    # Get adaptive weights
    recommender = AdaptiveRecommender()
    weights, query_type = recommender.get_adaptive_weights(query_text, query_entities)

    print(f"\nQuery: '{query_text}'")
    print(f"Detected Type: {query_type}")

    print(f"\nFixed Weights (one-size-fits-all):")
    print(f"  SimRank:   0.400")
    print(f"  Horn:      0.300")
    print(f"  Embedding: 0.300")

    print(f"\nAdaptive Weights (type-specific):")
    print(f"  SimRank:   {weights.simrank_weight:.3f}")
    print(f"  Horn:      {weights.horn_index_weight:.3f}")
    print(f"  Embedding: {weights.embedding_weight:.3f}")

    print(f"\nDifference:")
    print(f"  SimRank:   {weights.simrank_weight - 0.4:+.3f}")
    print(f"  Horn:      {weights.horn_index_weight - 0.3:+.3f}")
    print(f"  Embedding: {weights.embedding_weight - 0.3:+.3f}")

    print(f"\nWhy this matters:")
    if query_type == 'simple_keyword':
        print("  → Simple keyword queries benefit more from semantic similarity (embeddings)")
    elif query_type == 'entity_focused':
        print("  → Entity queries benefit from graph structure (SimRank) and importance (Horn)")
    elif query_type == 'concept_focused':
        print("  → Concept queries need balanced graph structure and semantic similarity")
    else:
        print("  → Complex queries benefit from all three components equally")


def demo_ensemble_comparison():
    """Demo: Compare different ensemble methods"""
    print("\n" + "="*80)
    print("DEMO 4: Ensemble Method Comparison")
    print("="*80)

    from ensemble_ranker import EnsembleRanker, RankedDocument

    # Sample documents with scores
    docs = [
        RankedDocument('doc_1', simrank_score=0.9, horn_score=0.7, embedding_score=0.6),
        RankedDocument('doc_2', simrank_score=0.7, horn_score=0.9, embedding_score=0.8),
        RankedDocument('doc_3', simrank_score=0.6, horn_score=0.8, embedding_score=0.9),
    ]

    methods = ['rrf', 'borda', 'combmnz']

    for method in methods:
        ranker = EnsembleRanker(fusion_method=method)
        ranked = ranker.rank(docs.copy())

        print(f"\n{method.upper()} Fusion:")
        for doc in ranked:
            print(f"  {doc.rank_position}. {doc.doc_id} (final_score: {doc.final_score:.4f})")


if __name__ == '__main__':
    # Run all demos
    demo_basic_search()
    demo_search_with_explanation()
    demo_weight_comparison()
    demo_ensemble_comparison()

    print("\n" + "="*80)
    print("INTEGRATION COMPLETE")
    print("="*80)
    print("\nTo integrate into your recommender:")
    print("  1. Copy the AdaptiveRecommender initialization from this file")
    print("  2. Add adaptive ranking after computing component scores")
    print("  3. Optionally add ranking explanations for debugging")
    print("\nSee EnhancedHeritageRecommender class for full example.")
