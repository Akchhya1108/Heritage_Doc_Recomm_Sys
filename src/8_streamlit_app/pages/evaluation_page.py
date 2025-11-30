"""
Evaluation Page - FIXED VERSION

Dynamically loads evaluation results and displays performance metrics.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
from pathlib import Path


@st.cache_data
def load_evaluation_results():
    """Load evaluation results from JSON file."""
    results_path = Path('results/method_comparison.json')
    
    if not results_path.exists():
        st.warning("⚠️ Evaluation results not found. Showing sample data.")
        # Return sample data structure
        return {
            'timestamp': '2025-11-30',
            'methods': {
                'Hybrid (50-50)': {
                    'precision@5': 0.828,
                    'precision@10': 0.79,
                    'recall@10': 0.15,
                    'ndcg@10': 0.687,
                    'MAP': 0.71,
                    'diversity': 0.65,
                    'coverage': 0.878,
                    'temporal_accuracy': 0.56,
                    'query_latency_ms': 0.21
                },
                'SimRank-Only': {
                    'precision@5': 0.824,
                    'precision@10': 0.78,
                    'recall@10': 0.14,
                    'ndcg@10': 0.684,
                    'MAP': 0.70,
                    'diversity': 0.62,
                    'coverage': 0.845,
                    'temporal_accuracy': 0.54,
                    'query_latency_ms': 0.24
                },
                'Embedding-Only': {
                    'precision@5': 0.276,
                    'precision@10': 0.25,
                    'recall@10': 0.08,
                    'ndcg@10': 0.288,
                    'MAP': 0.31,
                    'diversity': 0.71,
                    'coverage': 0.823,
                    'temporal_accuracy': 0.45,
                    'query_latency_ms': 0.28
                }
            }
        }
    
    try:
        with open(results_path, 'r') as f:
            data = json.load(f)
            return data
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return None


def create_metric_comparison_chart(results, metric_name):
    """Create bar chart comparing methods for a specific metric."""
    methods = list(results['methods'].keys())
    values = [results['methods'][m].get(metric_name, 0) for m in methods]
    
    fig = go.Figure(data=[
        go.Bar(
            x=methods,
            y=values,
            marker_color='#2563eb',
            text=[f'{v:.3f}' for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f'{metric_name.replace("_", " ").title()} by Method',
        xaxis_title='Method',
        yaxis_title=metric_name.replace('_', ' ').title(),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter', 'color': '#f1f5f9'},
        xaxis={'tickangle': -45}
    )
    
    return fig


def create_radar_chart(results):
    """Create radar chart for multi-dimensional comparison."""
    methods = list(results['methods'].keys())
    
    # Select key metrics for radar
    metrics = ['precision@10', 'ndcg@10', 'diversity', 'coverage']
    
    fig = go.Figure()
    
    for method in methods:
        values = [results['methods'][method].get(m, 0) for m in metrics]
        # Close the radar by repeating first value
        values_closed = values + [values[0]]
        metrics_closed = metrics + [metrics[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=metrics_closed,
            fill='toself',
            name=method
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter', 'color': '#f1f5f9'}
    )
    
    return fig


def render():
    """Render evaluation page - MAIN ENTRY POINT."""
    st.markdown('<h1 class="main-header">📈 Performance Evaluation</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <strong>📊 Evaluation Metrics:</strong> Comparing different recommendation methods 
        using standard information retrieval metrics.
    </div>
    """, unsafe_allow_html=True)
    
    # Load results
    results = load_evaluation_results()
    
    if not results:
        st.error("❌ Failed to load evaluation results")
        return
    
    methods = results.get('methods', {})
    
    if not methods:
        st.error("❌ No methods found in results")
        return
    
    # Summary statistics
    st.markdown("### 📊 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Find best method for each metric
    best_precision = max(methods.items(), key=lambda x: x[1].get('precision@5', 0))
    best_ndcg = max(methods.items(), key=lambda x: x[1].get('ndcg@10', 0))
    
    with col1:
        st.metric("Best Precision@5", f"{best_precision[1].get('precision@5', 0):.3f}", 
                 best_precision[0])
    
    with col2:
        st.metric("Best NDCG@10", f"{best_ndcg[1].get('ndcg@10', 0):.3f}", 
                 best_ndcg[0])
    
    with col3:
        avg_latency = sum(m.get('query_latency_ms', 0) for m in methods.values()) / len(methods)
        st.metric("Avg Latency", f"{avg_latency:.2f}ms")
    
    with col4:
        st.metric("Methods", len(methods))
    
    st.markdown("---")
    
    # Method comparison
    st.markdown("### 📈 Method Comparison")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        fig1 = create_metric_comparison_chart(results, 'precision@10')
        st.plotly_chart(fig1, use_container_width=True)
    
    with chart_col2:
        fig2 = create_metric_comparison_chart(results, 'ndcg@10')
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # Radar chart
    st.markdown("### 🎯 Multi-Dimensional Comparison")
    radar_fig = create_radar_chart(results)
    st.plotly_chart(radar_fig, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed metrics table
    st.markdown("### 📋 Detailed Metrics")
    
    # Create DataFrame
    metrics_data = []
    for method_name, metrics in methods.items():
        row = {'Method': method_name}
        row.update(metrics)
        metrics_data.append(row)
    
    df = pd.DataFrame(metrics_data)
    
    # Round numeric columns
    numeric_cols = df.select_dtypes(include=['float64']).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    
    st.dataframe(df, use_container_width=True, height=400)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download Results as CSV",
        data=csv,
        file_name="evaluation_results.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    render()