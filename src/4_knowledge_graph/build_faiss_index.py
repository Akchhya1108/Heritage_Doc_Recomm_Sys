"""
FAISS Index Builder
Build fast similarity search index for billion-scale retrieval
"""

import os
import sys
import json
import numpy as np
import faiss
import pickle
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.config_loader import get_config
from utils.logger import get_logger

# Initialize
config = get_config()
logger = get_logger(__name__)

def load_balanced_embeddings():
    """Load balanced document embeddings"""
    embeddings_file = config.get_path('data', 'balanced') + '/balanced_embeddings.npy'
    metadata_file = config.get_path('data', 'balanced') + '/balanced_documents.json'
    
    logger.info(f"Loading embeddings from: {embeddings_file}")
    
    if not os.path.exists(embeddings_file):
        logger.error(f"Embeddings not found: {embeddings_file}")
        logger.error("Please run balance_data.py first!")
        return None, None
    
    embeddings = np.load(embeddings_file)
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    logger.info(f"✓ Loaded {len(embeddings)} embeddings (dim: {embeddings.shape[1]})")
    logger.info(f"✓ Loaded metadata for {len(documents)} documents")
    
    return embeddings, documents

def build_flat_index(embeddings):
    """
    Build Flat (exact) FAISS index
    Best for < 1M vectors, exact search
    """
    logger.info("\n[Building] Flat Index (Exact Search)...")
    
    d = embeddings.shape[1]  # Dimension
    
    # Create flat L2 index
    index = faiss.IndexFlatL2(d)
    
    # Add vectors
    index.add(embeddings.astype('float32'))
    
    logger.info(f"   ✓ Added {index.ntotal} vectors")
    logger.info(f"   ✓ Index type: Flat (exact)")
    
    return index

def build_ivf_index(embeddings, nlist=100):
    """
    Build IVF (Inverted File) index
    Faster approximate search for 1M-1B vectors
    
    Args:
        nlist: Number of clusters (typically sqrt(n) to 4*sqrt(n))
    """
    logger.info(f"\n[Building] IVF Index (Approximate Search)...")
    logger.info(f"   Clusters (nlist): {nlist}")
    
    d = embeddings.shape[1]
    
    # Create quantizer (for clustering)
    quantizer = faiss.IndexFlatL2(d)
    
    # Create IVF index
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    
    # Train the index
    logger.info("   Training index...")
    index.train(embeddings.astype('float32'))
    
    # Add vectors
    logger.info("   Adding vectors...")
    index.add(embeddings.astype('float32'))
    
    # Set number of clusters to probe during search (speed vs accuracy tradeoff)
    index.nprobe = min(10, nlist // 10)  # Probe 10% of clusters
    
    logger.info(f"   ✓ Added {index.ntotal} vectors")
    logger.info(f"   ✓ Index type: IVF{nlist}")
    logger.info(f"   ✓ Probe clusters: {index.nprobe}")
    
    return index

def build_hnsw_index(embeddings, M=32, efConstruction=200):
    """
    Build HNSW (Hierarchical Navigable Small World) index
    Best for fast approximate search with high recall
    
    Args:
        M: Number of connections per layer (16-64, higher = better recall)
        efConstruction: Size of dynamic candidate list (100-500)
    """
    logger.info(f"\n[Building] HNSW Index (Graph-based Search)...")
    logger.info(f"   M (connections): {M}")
    logger.info(f"   efConstruction: {efConstruction}")
    
    d = embeddings.shape[1]
    
    # Create HNSW index
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = efConstruction
    
    # Add vectors
    logger.info("   Adding vectors...")
    index.add(embeddings.astype('float32'))
    
    # Set search parameter (speed vs accuracy)
    index.hnsw.efSearch = 64  # Higher = more accurate but slower
    
    logger.info(f"   ✓ Added {index.ntotal} vectors")
    logger.info(f"   ✓ Index type: HNSW{M}")
    logger.info(f"   ✓ efSearch: {index.hnsw.efSearch}")
    
    return index

def test_index_performance(index, embeddings, k=10, num_queries=100):
    """Test index search performance"""
    logger.info(f"\n[Testing] Index Performance...")
    logger.info(f"   Queries: {num_queries}")
    logger.info(f"   Top-K: {k}")
    
    # Select random queries
    np.random.seed(42)
    query_indices = np.random.choice(len(embeddings), size=num_queries, replace=False)
    queries = embeddings[query_indices].astype('float32')
    
    # Measure search time
    import time
    start = time.time()
    
    distances, indices = index.search(queries, k)
    
    elapsed = time.time() - start
    
    logger.info(f"\n   Results:")
    logger.info(f"   ✓ Total time: {elapsed:.3f}s")
    logger.info(f"   ✓ Queries/second: {num_queries/elapsed:.1f}")
    logger.info(f"   ✓ Latency per query: {elapsed/num_queries*1000:.2f}ms")
    
    # Analyze results
    avg_distance = distances.mean()
    logger.info(f"   ✓ Average distance to top-1: {avg_distance:.4f}")
    
    return distances, indices

def save_index(index, index_type, embeddings, documents):
    """Save FAISS index and metadata"""
    logger.info(f"\n[Saving] FAISS Index...")
    
    # Create output directory
    index_dir = os.path.join(config.get_path('models', 'ranker'), 'faiss')
    os.makedirs(index_dir, exist_ok=True)
    
    # Save index
    index_file = os.path.join(index_dir, f'{index_type}_index.faiss')
    faiss.write_index(index, index_file)
    logger.info(f"   ✓ Index saved: {index_file}")
    
    # Save metadata
    metadata = {
        'index_type': index_type,
        'n_vectors': index.ntotal,
        'dimension': embeddings.shape[1],
        'created_at': datetime.now().isoformat(),
        'index_params': {}
    }
    
    # Add type-specific params
    if index_type == 'ivf':
        metadata['index_params']['nlist'] = index.nlist
        metadata['index_params']['nprobe'] = index.nprobe
    elif index_type == 'hnsw':
        metadata['index_params']['M'] = index.hnsw.M
        metadata['index_params']['efConstruction'] = index.hnsw.efConstruction
        metadata['index_params']['efSearch'] = index.hnsw.efSearch
    
    metadata_file = os.path.join(index_dir, f'{index_type}_metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"   ✓ Metadata saved: {metadata_file}")
    
    # Save document ID mapping
    id_mapping = {
        i: {
            'doc_id': i,
            'title': doc.get('title', 'Unknown'),
            'cluster_id': doc.get('cluster_id', -1),
            'source': doc.get('source', 'Unknown')
        }
        for i, doc in enumerate(documents)
    }
    
    mapping_file = os.path.join(index_dir, 'document_mapping.json')
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(id_mapping, f, indent=2)
    
    logger.info(f"   ✓ Mapping saved: {mapping_file}")
    
    return index_file, metadata_file

def main():
    logger.info("="*70)
    logger.info("FAISS INDEX BUILDER")
    logger.info("="*70)
    
    # Load embeddings
    embeddings, documents = load_balanced_embeddings()
    
    if embeddings is None:
        return
    
    # Determine best index type based on dataset size
    n_docs = len(embeddings)
    
    logger.info(f"\n[Configuration]")
    logger.info(f"   Dataset size: {n_docs} documents")
    logger.info(f"   Embedding dim: {embeddings.shape[1]}")
    
    if n_docs < 1000:
        logger.info(f"   Recommended: Flat (exact) index")
        index_type = 'flat'
        index = build_flat_index(embeddings)
    
    elif n_docs < 100000:
        logger.info(f"   Recommended: HNSW (fast approximate) index")
        index_type = 'hnsw'
        index = build_hnsw_index(embeddings, M=32, efConstruction=200)
    
    else:
        logger.info(f"   Recommended: IVF (scalable) index")
        nlist = min(4096, int(np.sqrt(n_docs) * 4))
        index_type = 'ivf'
        index = build_ivf_index(embeddings, nlist=nlist)
    
    # Test performance
    test_index_performance(index, embeddings, k=10, num_queries=min(100, n_docs))
    
    # Save index
    index_file, metadata_file = save_index(index, index_type, embeddings, documents)
    
    # Build all index types for comparison (optional)
    logger.info(f"\n[Bonus] Building additional index types for comparison...")
    
    if n_docs >= 100:  # Only if we have enough docs
        # Build Flat for exact comparison
        if index_type != 'flat':
            flat_index = build_flat_index(embeddings)
            save_index(flat_index, 'flat', embeddings, documents)
        
        # Build HNSW for speed comparison
        if index_type != 'hnsw' and n_docs < 10000:
            hnsw_index = build_hnsw_index(embeddings)
            save_index(hnsw_index, 'hnsw', embeddings, documents)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("FAISS INDEX BUILD COMPLETE")
    logger.info("="*70)
    logger.info(f"✅ Built {index_type.upper()} index for {n_docs} documents")
    logger.info(f"✅ Index dimension: {embeddings.shape[1]}")
    logger.info(f"\n📂 Output files:")
    logger.info(f"   - {index_file}")
    logger.info(f"   - {metadata_file}")
    logger.info(f"   - models/ranker/faiss/document_mapping.json")
    logger.info("\n💡 Usage:")
    logger.info(f"   import faiss")
    logger.info(f"   index = faiss.read_index('{index_file}')")
    logger.info(f"   distances, indices = index.search(query_vector, k=10)")
    logger.info("="*70)

if __name__ == "__main__":
    main()