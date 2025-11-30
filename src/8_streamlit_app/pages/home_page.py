"""
Home Page / Dashboard
Modern 2025 UI with Heritage Color Palette - FIXED VERSION
"""

import streamlit as st
import json
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def load_css():
    """Load custom CSS styling."""
    css_path = Path(__file__).parent.parent / 'style.css'
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_system_stats():
    """Load REAL system statistics from files."""
    stats = {
        'total_documents': 0,
        'kg_nodes': 0,
        'kg_edges': 0,
        'clusters': 12,
        'best_precision': 0.0,
        'best_method': 'Unknown',
        'avg_latency': 0.0,
        'coverage': 0.0
    }

    # Try to load actual document count
    try:
        metadata_file = Path('data/metadata/enriched_metadata.json')
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                stats['total_documents'] = len(metadata)
    except Exception as e:
        st.warning(f"Could not load document metadata: {e}")

    # Try to load KG statistics
    try:
        kg_stats_file = Path('knowledge_graph/kg_statistics.json')
        if kg_stats_file.exists():
            with open(kg_stats_file, 'r') as f:
                kg_stats = json.load(f)
                stats['kg_nodes'] = kg_stats.get('total_nodes', 0)
                stats['kg_edges'] = kg_stats.get('total_edges', 0)
    except Exception as e:
        st.warning(f"Could not load KG statistics: {e}")

    # Try to load evaluation results
    try:
        comparison_file = Path('results/method_comparison.json')
        if comparison_file.exists():
            with open(comparison_file, 'r') as f:
                comparison = json.load(f)
                methods = comparison.get('methods', {})

                if methods:
                    best_precision = 0
                    best_method_name = ''

                    for method, metrics in methods.items():
                        p5 = metrics.get('precision@5', 0)
                        if p5 > best_precision:
                            best_precision = p5
                            best_method_name = method

                    stats['best_precision'] = best_precision
                    stats['best_method'] = best_method_name

                    latencies = [m.get('query_latency_ms', 0) for m in methods.values() if m.get('query_latency_ms')]
                    if latencies:
                        stats['avg_latency'] = sum(latencies) / len(latencies)
                    
                    coverages = [m.get('coverage', 0) for m in methods.values() if m.get('coverage')]
                    if coverages:
                        stats['coverage'] = sum(coverages) / len(coverages)
    except Exception as e:
        st.warning(f"Could not load evaluation results: {e}")

    return stats


def render():
    """Render modern homepage."""
    # Load CSS
    load_css()

    # Hero Section
    st.markdown("""
        <div class='hero-section fade-in'>
            <h1 class='hero-title'>
                🏛️ Heritage Recommender
            </h1>
            <p class='hero-subtitle'>
                AI-Powered Heritage Document Discovery
            </p>
            <p class='hero-description'>
                Knowledge Graph • Graph Algorithms • Deep Learning
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Load stats
    with st.spinner("Loading system statistics..."):
        stats = load_system_stats()

    # Check if we have actual data
    if stats['total_documents'] == 0:
        st.warning("""
        ⚠️ **System data not found**
        
        It looks like the system hasn't been fully set up yet. Please run:
        1. Data collection: `python src/1_data_collection/1_collect_all_sources.py`
        2. Preprocessing: `python src/2_preprocessing/clean_data.py`
        3. Knowledge graph: `python src/4_knowledge_graph/5_build_knowledge_graph.py`
        
        Using placeholder values for now.
        """)
        # Use placeholder values
        stats = {
            'total_documents': 369,
            'kg_nodes': 500,
            'kg_edges': 6500,
            'clusters': 12,
            'best_precision': 0.828,
            'best_method': 'Hybrid (50-50)',
            'avg_latency': 0.21,
            'coverage': 0.878
        }

    # Stats Overview
    st.markdown("### 📊 System Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
            <div class='metric-card slide-in-left' style='animation-delay: 0s;'>
                <div style='font-size: 3rem; margin-bottom: 12px;'>📚</div>
                <div style='font-size: 2.5rem; font-weight: 700; color: #2563eb; margin: 12px 0;'>{stats['total_documents']}</div>
                <div style='color: #94a3b8; font-size: 0.9rem; font-weight: 700; letter-spacing: 0.5px;'>DOCUMENTS</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class='metric-card slide-in-left' style='animation-delay: 0.1s;'>
                <div style='font-size: 3rem; margin-bottom: 12px;'>🕸️</div>
                <div style='font-size: 2.5rem; font-weight: 700; color: #0d9488; margin: 12px 0;'>{stats['kg_nodes']}</div>
                <div style='color: #94a3b8; font-size: 0.9rem; font-weight: 700; letter-spacing: 0.5px;'>KG NODES</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        precision_pct = stats['best_precision'] * 100 if stats['best_precision'] < 10 else stats['best_precision']
        st.markdown(f"""
            <div class='metric-card slide-in-left' style='animation-delay: 0.2s;'>
                <div style='font-size: 3rem; margin-bottom: 12px;'>🎯</div>
                <div style='font-size: 2.5rem; font-weight: 700; color: #10b981; margin: 12px 0;'>{precision_pct:.1f}%</div>
                <div style='color: #94a3b8; font-size: 0.9rem; font-weight: 700; letter-spacing: 0.5px;'>PRECISION@5</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class='metric-card slide-in-left' style='animation-delay: 0.3s;'>
                <div style='font-size: 3rem; margin-bottom: 12px;'>⚡</div>
                <div style='font-size: 2.5rem; font-weight: 700; color: #f59e0b; margin: 12px 0;'>{stats['avg_latency']:.2f}ms</div>
                <div style='color: #94a3b8; font-size: 0.9rem; font-weight: 700; letter-spacing: 0.5px;'>LATENCY</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature Cards
    st.markdown("### ✨ Key Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class='feature-card fade-in' style='animation-delay: 0s;'>
                <div class='feature-icon'>🔍</div>
                <h3 class='feature-title'>Smart Search</h3>
                <p class='feature-description'>
                    Natural language queries with entity extraction and semantic understanding powered by advanced NLP.
                </p>
                <div style='margin-top: 20px;'>
                    <span class='badge badge-primary'>SimRank</span>
                    <span class='badge badge-gold'>Horn's Index</span>
                    <span class='badge badge-accent'>Embeddings</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class='feature-card fade-in' style='animation-delay: 0.2s;'>
                <div class='feature-icon'>🕸️</div>
                <h3 class='feature-title'>Knowledge Graph</h3>
                <p class='feature-description'>
                    Interactive visualization of document relationships and entity connections with {stats['kg_edges']}+ edges.
                </p>
                <div style='margin-top: 20px;'>
                    <span class='badge badge-primary'>{stats['kg_nodes']}+ Nodes</span>
                    <span class='badge badge-secondary'>PyVis</span>
                    <span class='badge badge-accent'>Interactive</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class='feature-card fade-in' style='animation-delay: 0.4s;'>
                <div class='feature-icon'>📊</div>
                <h3 class='feature-title'>Evaluation</h3>
                <p class='feature-description'>
                    Comprehensive metrics for quality, fairness, diversity, and performance with real-time monitoring.
                </p>
                <div style='margin-top: 20px;'>
                    <span class='badge badge-accent'>NDCG@10</span>
                    <span class='badge badge-gold'>Coverage</span>
                    <span class='badge badge-primary'>Real-time</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Performance Visualizations
    st.markdown("### 📈 Performance Metrics")
    
    st.markdown("""
    <div class='info-box' style='margin-bottom: 20px;'>
        <strong>📊 What are these metrics?</strong><br>
        <strong>Precision@5:</strong> Percentage of top-5 recommended documents that are actually relevant to the query. 
        Higher is better (82.8% means 4 out of 5 recommendations are relevant).<br>
        <strong>Coverage:</strong> Percentage of documents in the collection that can be recommended. 
        Higher coverage means the system can find relevant documents across more diverse topics.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        precision_val = stats['best_precision'] * 100 if stats['best_precision'] < 10 else stats['best_precision']
        
        # Precision gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=precision_val,
            title={'text': "Precision@5 (%)", 'font': {'size': 20, 'color': '#f1f5f9', 'family': 'Inter'}},
            delta={'reference': 75, 'increasing': {'color': "#10b981"}},
            number={'suffix': "%", 'font': {'size': 40, 'color': '#2563eb'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#94a3b8"},
                'bar': {'color': "#2563eb", 'thickness': 0.75},
                'bgcolor': "#1e293b",
                'borderwidth': 2,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.1)'},
                    {'range': [50, 75], 'color': 'rgba(245, 158, 11, 0.2)'},
                    {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': "#2563eb", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'family': 'Inter', 'color': '#94a3b8'},
            height=300,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        coverage_val = stats['coverage'] * 100 if stats['coverage'] < 10 else stats['coverage']
        
        # Coverage gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=coverage_val,
            title={'text': "Coverage (%)", 'font': {'size': 20, 'color': '#f1f5f9', 'family': 'Inter'}},
            delta={'reference': 70, 'increasing': {'color': "#10b981"}},
            number={'suffix': "%", 'font': {'size': 40, 'color': '#0d9488'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#94a3b8"},
                'bar': {'color': "#0d9488", 'thickness': 0.75},
                'bgcolor': "#1e293b",
                'borderwidth': 2,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.1)'},
                    {'range': [50, 75], 'color': 'rgba(245, 158, 11, 0.2)'},
                    {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': "#0d9488", 'width': 4},
                    'thickness': 0.75,
                    'value': 75
                }
            }
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'family': 'Inter', 'color': '#94a3b8'},
            height=300,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Quick Start Guide with proper navigation
    st.markdown("### 🚀 Quick Start")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🔍 Search Documents", use_container_width=True, type="primary", key="nav_search"):
            st.switch_page("pages/search_page.py")

    with col2:
        if st.button("🕸️ Explore Graph", use_container_width=True, type="primary", key="nav_graph"):
            st.switch_page("pages/kg_viz_page.py")

    with col3:
        if st.button("📊 View Results", use_container_width=True, type="primary", key="nav_results"):
            if 'recommendations' in st.session_state and st.session_state.recommendations:
                st.switch_page("pages/results_page.py")
            else:
                st.info("No results yet. Please run a search first!")

    with col4:
        if st.button("📈 Evaluation", use_container_width=True, type="primary", key="nav_eval"):
            st.switch_page("pages/evaluation_page.py")

    st.markdown("---")

    # System Architecture Overview
    st.markdown("### 🏗️ System Architecture")

    st.markdown("""
        <div class='info-box fade-in'>
            <h4 style='color: #f1f5f9; margin-top: 0; font-weight: 700;'>Hybrid Ranking System</h4>
            <p style='margin-bottom: 0; color: #cbd5e1; font-weight: 600;'>
                Our system combines three powerful ranking methods:
            </p>
            <ul style='margin: 12px 0 0 20px; color: #cbd5e1; font-weight: 500;'>
                <li><strong style='color: #f1f5f9;'>SimRank (40%)</strong> - Graph-based similarity using random walk</li>
                <li><strong style='color: #f1f5f9;'>Horn's Index (30%)</strong> - Entity overlap and conceptual overlap</li>
                <li><strong style='color: #f1f5f9;'>Embeddings (30%)</strong> - Deep semantic understanding with sentence-transformers</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Performance highlights
    st.markdown("### 🎯 Performance Highlights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
            <div class='success-box'>
                <div style='font-size: 2rem; margin-bottom: 8px;'>✅</div>
                <h4 style='color: #a7f3d0; margin: 8px 0; font-weight: 800;'>High Precision</h4>
                <p style='margin: 0; font-size: 0.95rem; color: #a7f3d0; font-weight: 600;'>
                    {stats['best_precision']*100:.1f}% Precision@5 on curated heritage document queries
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class='success-box'>
                <div style='font-size: 2rem; margin-bottom: 8px;'>⚡</div>
                <h4 style='color: #fde68a; margin: 8px 0; font-weight: 800;'>Ultra-Fast</h4>
                <p style='margin: 0; font-size: 0.95rem; color: #fde68a; font-weight: 600;'>
                    {stats['avg_latency']:.2f}ms query latency with efficient caching
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class='success-box'>
                <div style='font-size: 2rem; margin-bottom: 8px;'>🎨</div>
                <h4 style='color: #e0e7ff; margin: 8px 0; font-weight: 800;'>Explainable</h4>
                <p style='margin: 0; font-size: 0.95rem; color: #e0e7ff; font-weight: 600;'>
                    Knowledge graph paths show why documents are recommended
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Example queries
    st.markdown("---")
    st.markdown("### 💡 Example Queries to Try")

    examples_col1, examples_col2 = st.columns(2)

    with examples_col1:
        st.markdown("""
            <div class='result-card'>
                <h4 style='color: #f1f5f9; margin-top: 0; font-weight: 700;'>🕌 Architecture & Monuments</h4>
                <ul style='line-height: 2; color: #cbd5e1; font-weight: 500;'>
                    <li>"Mughal architecture in Delhi"</li>
                    <li>"Ancient Buddhist monasteries"</li>
                    <li>"Dravidian temples in Tamil Nadu"</li>
                    <li>"Rajput forts and palaces"</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with examples_col2:
        st.markdown("""
            <div class='result-card'>
                <h4 style='color: #f1f5f9; margin-top: 0; font-weight: 700;'>🎨 Art & Culture</h4>
                <ul style='line-height: 2; color: #cbd5e1; font-weight: 500;'>
                    <li>"Rock-cut caves in Maharashtra"</li>
                    <li>"Medieval Islamic architecture"</li>
                    <li>"Temple sculptures in Khajuraho"</li>
                    <li>"Colonial era buildings"</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    render()