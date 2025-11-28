"""
SimRank Algorithm Implementation
Compute semantic similarity between documents using KG structure
"""

import pickle
import os
import sys
import json
import numpy as np
import networkx as nx
from datetime import datetime
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.config_loader import get_config
from utils.logger import get_logger

# Initialize
config = get_config()
logger = get_logger(__name__)

def load_knowledge_graph():
    """Load the knowledge graph"""
    kg_file = config.get_path('knowledge_graph', 'kg_file')
    
    logger.info(f"Loading knowledge graph from: {kg_file}")
    
    if not os.path.exists(kg_file):
        logger.error(f"Knowledge graph not found: {kg_file}")
        logger.error("Please run 5_build_knowledge_graph.py first!")
        return None
    
    with open(kg_file, 'rb') as f:
        G = pickle.load(f)
    
    logger.info(f"✓ Loaded KG with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G

def get_document_nodes(G):
    """Extract only document nodes from KG"""
    doc_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'document']
    
    logger.info(f"✓ Found {len(doc_nodes)} document nodes")
    
    return doc_nodes

def compute_simrank_optimized(G, doc_nodes, max_iter=None, decay=None, threshold=None):
    """
    Compute SimRank similarity between document nodes
    
    SimRank formula:
    sim(a,b) = C * avg(sim(I(a), I(b))) where I(x) are in-neighbors
    
    Optimized for document nodes only
    """
    # Get config parameters
    if max_iter is None:
        max_iter = config.get('models', 'simrank', 'max_iterations', default=10)
    if decay is None:
        decay = config.get('models', 'simrank', 'decay_factor', default=0.8)
    if threshold is None:
        threshold = config.get('models', 'simrank', 'convergence_threshold', default=0.001)
    
    logger.info(f"\n[SimRank] Configuration:")
    logger.info(f"   Max iterations: {max_iter}")
    logger.info(f"   Decay factor (C): {decay}")
    logger.info(f"   Convergence threshold: {threshold}")
    logger.info(f"   Document nodes: {len(doc_nodes)}")
    
    # Initialize similarity matrix
    n = len(doc_nodes)
    node_to_idx = {node: i for i, node in enumerate(doc_nodes)}
    
    # Current and previous similarity matrices
    sim = np.eye(n, dtype=np.float32)  # Identity matrix (each doc similar to itself)
    sim_prev = np.zeros((n, n), dtype=np.float32)
    
    # Get neighbors for each document node (only within doc nodes)
    logger.info("\n[Phase 1] Building neighbor index...")
    neighbors = {}
    for node in tqdm(doc_nodes, desc="Indexing neighbors"):
        # Get all neighbors (both in and out edges)
        all_neighbors = set(G.predecessors(node)) | set(G.successors(node))
        # Filter to only document nodes
        doc_neighbors = [n for n in all_neighbors if n in node_to_idx]
        neighbors[node] = doc_neighbors
    
    logger.info(f"✓ Indexed neighbors for {len(neighbors)} nodes")
    
    # Iterative SimRank computation
    logger.info(f"\n[Phase 2] Computing SimRank...")
    
    for iteration in range(max_iter):
        sim_prev = sim.copy()
        
        # Compute new similarities
        for i, node_a in enumerate(tqdm(doc_nodes, desc=f"Iter {iteration+1}/{max_iter}")):
            for j, node_b in enumerate(doc_nodes):
                if i == j:
                    sim[i, j] = 1.0  # Self-similarity
                    continue
                
                # Get neighbors
                neighbors_a = neighbors[node_a]
                neighbors_b = neighbors[node_b]
                
                if len(neighbors_a) == 0 or len(neighbors_b) == 0:
                    sim[i, j] = 0.0
                    continue
                
                # Compute average similarity of all neighbor pairs
                total_sim = 0.0
                count = 0
                
                for na in neighbors_a:
                    for nb in neighbors_b:
                        if na in node_to_idx and nb in node_to_idx:
                            na_idx = node_to_idx[na]
                            nb_idx = node_to_idx[nb]
                            total_sim += sim_prev[na_idx, nb_idx]
                            count += 1
                
                if count > 0:
                    sim[i, j] = decay * (total_sim / count)
                else:
                    sim[i, j] = 0.0
        
        # Check convergence
        diff = np.abs(sim - sim_prev).max()
        logger.info(f"   Iteration {iteration+1}: max difference = {diff:.6f}")
        
        if diff < threshold:
            logger.info(f"   ✓ Converged after {iteration+1} iterations!")
            break
    
    logger.info(f"\n✓ SimRank computation complete")
    
    return sim, node_to_idx

def compute_simrank_approximate(G, doc_nodes, max_iter=5, decay=0.8, sample_size=50):
    """
    Faster approximate SimRank using sampling
    Good for large graphs (1000+ documents)
    """
    logger.info(f"\n[SimRank Approximate] Using sampling for speed...")
    logger.info(f"   Sample size: {sample_size} neighbor pairs")
    
    n = len(doc_nodes)
    node_to_idx = {node: i for i, node in enumerate(doc_nodes)}
    
    sim = np.eye(n, dtype=np.float32)
    
    # Build neighbor index
    logger.info("\n[Phase 1] Building neighbor index...")
    neighbors = {}
    for node in tqdm(doc_nodes, desc="Indexing"):
        all_neighbors = set(G.predecessors(node)) | set(G.successors(node))
        doc_neighbors = [n for n in all_neighbors if n in node_to_idx]
        neighbors[node] = doc_neighbors
    
    # Approximate SimRank
    logger.info(f"\n[Phase 2] Computing approximate SimRank...")
    
    for iteration in range(max_iter):
        sim_prev = sim.copy()
        
        for i, node_a in enumerate(tqdm(doc_nodes, desc=f"Iter {iteration+1}/{max_iter}")):
            for j in range(i+1, n):  # Only compute upper triangle
                node_b = doc_nodes[j]
                
                neighbors_a = neighbors[node_a]
                neighbors_b = neighbors[node_b]
                
                if len(neighbors_a) == 0 or len(neighbors_b) == 0:
                    sim[i, j] = sim[j, i] = 0.0
                    continue
                
                # Sample neighbor pairs instead of all pairs
                num_samples = min(sample_size, len(neighbors_a) * len(neighbors_b))
                
                total_sim = 0.0
                for _ in range(num_samples):
                    na = np.random.choice(neighbors_a)
                    nb = np.random.choice(neighbors_b)
                    
                    if na in node_to_idx and nb in node_to_idx:
                        na_idx = node_to_idx[na]
                        nb_idx = node_to_idx[nb]
                        total_sim += sim_prev[na_idx, nb_idx]
                
                avg_sim = decay * (total_sim / num_samples)
                sim[i, j] = sim[j, i] = avg_sim
        
        diff = np.abs(sim - sim_prev).max()
        logger.info(f"   Iteration {iteration+1}: max difference = {diff:.6f}")
    
    logger.info(f"\n✓ Approximate SimRank complete")
    
    return sim, node_to_idx

def analyze_simrank_scores(sim_matrix, doc_nodes, top_k=10):
    """Analyze SimRank results"""
    logger.info(f"\n[Analysis] SimRank Statistics:")
    
    # Get non-diagonal values
    n = sim_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    non_diag = sim_matrix[mask]
    
    logger.info(f"   Mean similarity: {non_diag.mean():.4f}")
    logger.info(f"   Std similarity: {non_diag.std():.4f}")
    logger.info(f"   Min similarity: {non_diag.min():.4f}")
    logger.info(f"   Max similarity: {non_diag.max():.4f}")
    logger.info(f"   Median similarity: {np.median(non_diag):.4f}")
    
    # Find most similar pairs
    logger.info(f"\n   Top {top_k} most similar document pairs:")
    
    # Get upper triangle indices (avoid duplicates)
    triu_indices = np.triu_indices(n, k=1)
    triu_values = sim_matrix[triu_indices]
    
    # Get top-k
    top_k_idx = np.argsort(triu_values)[-top_k:][::-1]
    
    for rank, idx in enumerate(top_k_idx, 1):
        i = triu_indices[0][idx]
        j = triu_indices[1][idx]
        score = sim_matrix[i, j]
        
        # Get document titles (truncate for display)
        node_i = doc_nodes[i]
        node_j = doc_nodes[j]
        
        logger.info(f"      {rank}. Score: {score:.4f}")
        logger.info(f"         Doc {i} ↔ Doc {j}")

def save_simrank_results(sim_matrix, node_to_idx, doc_nodes, G):
    """Save SimRank results"""
    logger.info(f"\n[Saving] Storing SimRank results...")
    
    output_file = config.get_path('knowledge_graph', 'simrank_scores')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Prepare data for storage
    simrank_data = {
        'similarity_matrix': sim_matrix,
        'node_to_idx': node_to_idx,
        'idx_to_node': {v: k for k, v in node_to_idx.items()},
        'doc_nodes': doc_nodes,
        'metadata': {
            'n_documents': len(doc_nodes),
            'matrix_shape': sim_matrix.shape,
            'mean_similarity': float(sim_matrix[~np.eye(len(doc_nodes), dtype=bool)].mean()),
            'computed_at': datetime.now().isoformat(),
            'algorithm': 'SimRank',
            'parameters': {
                'max_iterations': config.get('models', 'simrank', 'max_iterations'),
                'decay_factor': config.get('models', 'simrank', 'decay_factor'),
                'convergence_threshold': config.get('models', 'simrank', 'convergence_threshold')
            }
        }
    }
    
    # Save with pickle (preserves numpy arrays)
    with open(output_file, 'wb') as f:
        pickle.dump(simrank_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    logger.info(f"   ✓ Saved to: {output_file}")
    
    # Also save metadata as JSON for inspection
    json_file = output_file.replace('.pkl', '_metadata.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(simrank_data['metadata'], f, indent=2)
    
    logger.info(f"   ✓ Metadata saved to: {json_file}")
    
    # Save top-k similar pairs for quick inspection
    save_top_similar_pairs(sim_matrix, doc_nodes, G, top_k=100)

def save_top_similar_pairs(sim_matrix, doc_nodes, G, top_k=100):
    """Save top-k most similar document pairs"""
    logger.info(f"\n[Exporting] Top {top_k} similar pairs...")
    
    n = sim_matrix.shape[0]
    
    # Get all pairs with similarities
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append({
                'doc_idx_1': i,
                'doc_idx_2': j,
                'doc_id_1': doc_nodes[i],
                'doc_id_2': doc_nodes[j],
                'title_1': G.nodes[doc_nodes[i]].get('title', 'Unknown')[:60],
                'title_2': G.nodes[doc_nodes[j]].get('title', 'Unknown')[:60],
                'similarity': float(sim_matrix[i, j])
            })
    
    # Sort by similarity
    pairs.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Save top-k
    output_file = config.get_path('knowledge_graph', 'base') + '/top_similar_pairs.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'top_k': top_k,
            'total_pairs': len(pairs),
            'pairs': pairs[:top_k]
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"   ✓ Saved top pairs to: {output_file}")

def main():
    logger.info("="*70)
    logger.info("SIMRANK COMPUTATION")
    logger.info("="*70)
    
    # Load KG
    G = load_knowledge_graph()
    if G is None:
        return
    
    # Get document nodes
    doc_nodes = get_document_nodes(G)
    
    if len(doc_nodes) == 0:
        logger.error("No document nodes found in KG!")
        return
    
    # Choose algorithm based on graph size
    if len(doc_nodes) > 500:
        logger.warning(f"Large graph detected ({len(doc_nodes)} docs)")
        logger.warning("Using approximate SimRank for speed...")
        sim_matrix, node_to_idx = compute_simrank_approximate(G, doc_nodes)
    else:
        sim_matrix, node_to_idx = compute_simrank_optimized(G, doc_nodes)
    
    # Analyze results
    analyze_simrank_scores(sim_matrix, doc_nodes)
    
    # Save results
    save_simrank_results(sim_matrix, node_to_idx, doc_nodes, G)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("SIMRANK COMPUTATION COMPLETE")
    logger.info("="*70)
    logger.info(f"✅ Computed similarity for {len(doc_nodes)} documents")
    logger.info(f"✅ Matrix size: {sim_matrix.shape}")
    logger.info(f"✅ Sparsity: {np.sum(sim_matrix > 0.01) / sim_matrix.size * 100:.1f}% non-zero")
    logger.info(f"\n📂 Output files:")
    logger.info(f"   - {config.get_path('knowledge_graph', 'simrank_scores')}")
    logger.info(f"   - knowledge_graph/simrank_scores_metadata.json")
    logger.info(f"   - knowledge_graph/top_similar_pairs.json")
    logger.info("="*70)

if __name__ == "__main__":
    main()