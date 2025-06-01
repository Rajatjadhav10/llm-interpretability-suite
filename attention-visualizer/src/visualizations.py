import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
import streamlit as st
import networkx as nx

class AttentionVisualizer:
    """Create various attention visualizations."""
    
    def __init__(self, figure_width: int = 800, figure_height: int = 600):
        """
        Initialize visualizer with default figure dimensions.
        
        Args:
            figure_width: Default width for figures
            figure_height: Default height for figures
        """
        self.figure_width = figure_width
        self.figure_height = figure_height
        
        # Set up color scales
        self.color_scales = {
            'viridis': 'Viridis',
            'plasma': 'Plasma', 
            'blues': 'Blues',
            'reds': 'Reds',
            'rdylbu': 'RdYlBu',
            'spectral': 'Spectral'
        }

    def _format_tokens_for_display(self, tokens: List[str]) -> List[str]:
        """
        Format tokens for better display in visualizations.
        
        Args:
            tokens: List of raw tokens
            
        Returns:
            List of formatted tokens
        """
        formatted = []
        for token in tokens:
            # Handle subword tokens (BERT-style ##)
            if token.startswith('##'):
                formatted.append(token[2:])
            # Handle GPT-2 style tokens (Ġ prefix)
            elif token.startswith('Ġ'):
                formatted.append(token[1:])
            # Handle special tokens
            elif token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
                formatted.append(f"[{token.strip('[]<>')}]")
            else:
                formatted.append(token)
        
        return formatted

    def create_attention_heatmap(self, 
                               attention_matrix: np.ndarray,
                               tokens: List[str],
                               title: str = "Attention Heatmap",
                               colorscale: str = "Viridis") -> go.Figure:
        """
        Create an attention heatmap using Plotly.
        
        Args:
            attention_matrix: 2D attention weights (seq_len x seq_len)
            tokens: List of token strings
            title: Plot title
            colorscale: Color scale name
            
        Returns:
            Plotly figure object
        """
        # Format tokens for display
        display_tokens = self._format_tokens_for_display(tokens)
        
        # Create figure
        fig = go.Figure()
        
        # Add heatmap trace
        fig.add_trace(go.Heatmap(
            z=attention_matrix,
            x=display_tokens,
            y=display_tokens,
            colorscale=colorscale,
            showscale=True,
            hoverongaps=False,
            hovertemplate=
                "<b>From:</b> %{y}<br>" +
                "<b>To:</b> %{x}<br>" +
                "<b>Attention:</b> %{z:.3f}<br>" +
                "<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis_title="Attending To",
            yaxis_title="Attending From", 
            width=self.figure_width,
            height=self.figure_height,
            xaxis={'side': 'bottom'},
            yaxis={'autorange': 'reversed'}  # Reverse y-axis for better readability
        )
        
        return fig

    def create_multi_head_visualization(self,
                                      attention_weights: np.ndarray,
                                      tokens: List[str],
                                      layer_idx: int = 0,
                                      max_heads: int = 12) -> go.Figure:
        """
        Create a multi-head attention visualization showing multiple heads in one layer.
        
        Args:
            attention_weights: 4D array (layers, heads, seq_len, seq_len)
            tokens: List of token strings
            layer_idx: Which layer to visualize
            max_heads: Maximum number of heads to show
            
        Returns:
            Plotly figure with subplots for each head
        """
        display_tokens = self._format_tokens_for_display(tokens)
        num_heads = min(attention_weights.shape[1], max_heads)
        
        # Calculate subplot layout
        cols = 4 if num_heads > 4 else num_heads
        rows = (num_heads + cols - 1) // cols
        
        # Create subplots
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f"Head {i+1}" for i in range(num_heads)],
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        for head_idx in range(num_heads):
            row = head_idx // cols + 1
            col = head_idx % cols + 1
            
            attention_matrix = attention_weights[layer_idx, head_idx]
            
            fig.add_trace(
                go.Heatmap(
                    z=attention_matrix,
                    x=display_tokens,
                    y=display_tokens,
                    colorscale="Viridis",
                    showscale=(head_idx == 0),  # Only show scale for first plot
                    hovertemplate=
                        f"<b>Head {head_idx+1}</b><br>" +
                        "<b>From:</b> %{y}<br>" +
                        "<b>To:</b> %{x}<br>" +
                        "<b>Attention:</b> %{z:.3f}<br>" +
                        "<extra></extra>"
                ),
                row=row,
                col=col
            )
        
        # Update layout
        fig.update_layout(
            title={
                'text': f"Multi-Head Attention - Layer {layer_idx + 1}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            width=self.figure_width * 1.2,
            height=self.figure_height * (rows / 2),
        )
        
        # Update all xaxis and yaxis to reverse y-axis
        for i in range(1, num_heads + 1):
            fig.update_yaxes(autorange='reversed', row=(i-1)//cols + 1, col=(i-1)%cols + 1)
        
        return fig

    def create_attention_flow_graph(self,
                                   attention_matrix: np.ndarray,
                                   tokens: List[str],
                                   threshold: float = 0.1,
                                   title: str = "Attention Flow Graph") -> go.Figure:
        """
        Create an attention flow graph showing connections between tokens.
        
        Args:
            attention_matrix: 2D attention weights (seq_len x seq_len)
            tokens: List of token strings
            threshold: Minimum attention weight to show as connection
            title: Plot title
            
        Returns:
            Plotly figure with network graph
        """
        display_tokens = self._format_tokens_for_display(tokens)
        n_tokens = len(tokens)
        
        # Filter attention weights above threshold
        strong_connections = np.where(attention_matrix > threshold)
        
        # Create edges and weights
        edge_x = []
        edge_y = []
        edge_weights = []
        edge_info = []
        
        # Position tokens in a circle
        angles = np.linspace(0, 2*np.pi, n_tokens, endpoint=False)
        node_x = np.cos(angles)
        node_y = np.sin(angles)
        
        # Create edges
        for from_idx, to_idx in zip(strong_connections[0], strong_connections[1]):
            if from_idx != to_idx:  # Skip self-attention
                weight = attention_matrix[from_idx, to_idx]
                
                # Add edge coordinates
                edge_x.extend([node_x[from_idx], node_x[to_idx], None])
                edge_y.extend([node_y[from_idx], node_y[to_idx], None])
                edge_weights.append(weight)
                edge_info.append(f"{display_tokens[from_idx]} → {display_tokens[to_idx]}: {weight:.3f}")
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        if edge_x:
            fig.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(width=2, color='rgba(125,125,125,0.5)'),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=20,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            ),
            text=display_tokens,
            textposition="middle center",
            textfont=dict(size=10, color='black'),
            hovertemplate="<b>Token:</b> %{text}<br><extra></extra>",
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': f"{title} (threshold > {threshold})",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=self.figure_width,
            height=self.figure_height,
            plot_bgcolor='white'
        )
        
        return fig

    def create_layer_comparison(self,
                              attention_weights: np.ndarray,
                              tokens: List[str],
                              aggregation: str = "mean") -> go.Figure:
        """
        Create a comparison of attention patterns across layers.
        
        Args:
            attention_weights: 4D array (layers, heads, seq_len, seq_len)
            tokens: List of token strings
            aggregation: How to aggregate across heads ("mean", "max")
            
        Returns:
            Plotly figure comparing layers
        """
        display_tokens = self._format_tokens_for_display(tokens)
        num_layers = attention_weights.shape[0]
        
        # Aggregate across heads for each layer
        if aggregation == "mean":
            layer_attention = np.mean(attention_weights, axis=1)
        elif aggregation == "max":
            layer_attention = np.max(attention_weights, axis=1)
        else:
            layer_attention = np.mean(attention_weights, axis=1)
        
        # Create subplots
        cols = 4 if num_layers > 4 else num_layers
        rows = (num_layers + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f"Layer {i+1}" for i in range(num_layers)],
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        for layer_idx in range(num_layers):
            row = layer_idx // cols + 1
            col = layer_idx % cols + 1
            
            fig.add_trace(
                go.Heatmap(
                    z=layer_attention[layer_idx],
                    x=display_tokens,
                    y=display_tokens,
                    colorscale="Viridis",
                    showscale=(layer_idx == 0),
                    hovertemplate=
                        f"<b>Layer {layer_idx+1}</b><br>" +
                        "<b>From:</b> %{y}<br>" +
                        "<b>To:</b> %{x}<br>" +
                        "<b>Attention:</b> %{z:.3f}<br>" +
                        "<extra></extra>"
                ),
                row=row,
                col=col
            )
        
        # Update layout
        fig.update_layout(
            title={
                'text': f"Layer Comparison ({aggregation.title()} across heads)",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            width=self.figure_width * 1.5,
            height=self.figure_height * (rows / 2),
        )
        
        # Update all yaxis to reverse
        for i in range(1, num_layers + 1):
            fig.update_yaxes(autorange='reversed', row=(i-1)//cols + 1, col=(i-1)%cols + 1)
        
        return fig

    def create_token_importance_plot(self,
                                   tokens: List[str],
                                   importance_scores: np.ndarray,
                                   title: str = "Token Importance") -> go.Figure:
        """
        Create a bar plot showing token importance scores.
        
        Args:
            tokens: List of token strings
            importance_scores: 1D array of importance scores
            title: Plot title
            
        Returns:
            Plotly bar chart
        """
        display_tokens = self._format_tokens_for_display(tokens)
        
        # Create bar plot
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=display_tokens,
            y=importance_scores,
            marker_color='lightblue',
            marker_line_color='darkblue',
            marker_line_width=1,
            hovertemplate="<b>Token:</b> %{x}<br><b>Importance:</b> %{y:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis_title="Tokens",
            yaxis_title="Importance Score",
            width=self.figure_width,
            height=self.figure_height // 2,
            xaxis_tickangle=-45
        )
        
        return fig

    def create_attention_distribution_plot(self,
                                         attention_weights: np.ndarray,
                                         title: str = "Attention Distribution") -> go.Figure:
        """
        Create a histogram showing the distribution of attention weights.
        
        Args:
            attention_weights: Multi-dimensional attention array
            title: Plot title
            
        Returns:
            Plotly histogram
        """
        # Flatten attention weights
        flat_attention = attention_weights.flatten()
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=flat_attention,
            nbinsx=50,
            marker_color='lightcoral',
            marker_line_color='darkred',
            marker_line_width=1,
            hovertemplate="<b>Attention Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>"
        ))
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis_title="Attention Weight",
            yaxis_title="Frequency",
            width=self.figure_width,
            height=self.figure_height // 2
        )
        
        return fig

# Utility functions for advanced visualizations
def calculate_attention_entropy(attention_matrix: np.ndarray) -> np.ndarray:
    """Calculate entropy for each token's attention distribution."""
    # Add small epsilon to avoid log(0)
    epsilon = 1e-12
    attention_with_eps = attention_matrix + epsilon
    
    # Calculate entropy for each row (each token's attention distribution)
    entropy = -np.sum(attention_with_eps * np.log(attention_with_eps), axis=1)
    return entropy

def find_attention_patterns(attention_matrix: np.ndarray, 
                           tokens: List[str],
                           pattern_type: str = "high_attention") -> List[Tuple[str, str, float]]:
    """Find specific attention patterns in the matrix."""
    patterns = []
    
    if pattern_type == "high_attention":
        # Find pairs with highest attention
        threshold = np.percentile(attention_matrix, 90)
        high_attention = np.where(attention_matrix > threshold)
        
        for from_idx, to_idx in zip(high_attention[0], high_attention[1]):
            if from_idx != to_idx:  # Skip self-attention
                patterns.append((
                    tokens[from_idx],
                    tokens[to_idx], 
                    attention_matrix[from_idx, to_idx]
                ))
    
    # Sort by attention weight
    patterns.sort(key=lambda x: x[2], reverse=True)
    return patterns[:10]  # Return top 10