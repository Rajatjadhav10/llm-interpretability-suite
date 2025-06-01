import streamlit as st

# Configure Streamlit page - must be first Streamlit command
st.set_page_config(
    page_title=" LLM Attention Visualizer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

import numpy as np
import plotly.graph_objects as go
from typing import Dict, Optional
import traceback

# Import our modules
from src.config import config
from src.model_utils import AttentionExtractor, get_available_models, load_model_safe
from src.text_processing import TextProcessor, preprocess_text_for_model
from src.visualizations import AttentionVisualizer

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.info-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #4ECDC4;
    margin: 1rem 0;
}

.metric-card {
    background-color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
}

.stAlert > div {
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_cached(model_name: str) -> Optional[AttentionExtractor]:
    """Load model with caching to avoid reloading."""
    return load_model_safe(model_name)

@st.cache_data
def process_text_cached(model_name: str, text: str, aggregation_method: str) -> Dict:
    """Process text and extract attention with caching."""
    extractor = load_model_cached(model_name)
    if extractor is None:
        raise Exception(f"Failed to load model: {model_name}")
    
    if aggregation_method == "none":
        return extractor.get_attention_weights(text)
    else:
        return extractor.get_aggregated_attention(text, method=aggregation_method)

def main():
    """Main application function."""
    
    # Header
    st.markdown('<p class="main-header"> LLM Attention Visualizer</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <b> What does this do?</b><br>
    This app visualizes how transformer models pay attention to different words in your text. 
    Enter any sentence and see which words the model focuses on!
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = config.get_model_options()
        selected_model_display = st.selectbox(
            "Choose Model",
            options=list(model_options.keys()),
            help="Different models will show different attention patterns"
        )
        selected_model = model_options[selected_model_display]
        
        # Aggregation method
        agg_options = config.get_aggregation_options()
        selected_agg_display = st.selectbox(
            " Aggregation Method",
            options=list(agg_options.keys()) + ["None (Show Individual Heads)"],
            help="How to combine attention across layers and heads"
        )
        
        if selected_agg_display == "None (Show Individual Heads)":
            aggregation_method = "none"
        else:
            aggregation_method = agg_options[selected_agg_display]
        
        # Visualization options
        st.subheader("Visualization")
        
        viz_type = st.selectbox(
            "Visualization Type",
            ["Attention Heatmap", "Multi-Head View", "Attention Flow", "Layer Comparison"],
            help="Different ways to visualize attention patterns"
        )
        
        colorscale = st.selectbox(
            "Color Scheme",
            config.get_colorscale_options(),
            index=0
        )
        
        # Advanced options
        with st.expander(" Advanced Options"):
            if aggregation_method == "none":
                layer_idx = st.slider("Layer", 0, 11, 0, help="Which layer to visualize")
                head_idx = st.slider("Head", 0, 11, 0, help="Which attention head to visualize")
            else:
                layer_idx = head_idx = 0
            
            threshold = st.slider(
                "Flow Graph Threshold", 
                0.0, 0.5, 0.1, 0.05,
                help="Minimum attention weight to show in flow graph"
            )
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(" Input Text")
        
        # Example texts
        if st.button(" Use Random Example"):
            example_text = np.random.choice(config.app_config.example_texts)
            st.session_state.text_input = example_text
        
        # Text input
        user_text = st.text_area(
            "Enter your text:",
            value=st.session_state.get('text_input', config.app_config.example_texts[0]),
            height=150,
            max_chars=config.app_config.max_text_length,
            help=f"Maximum {config.app_config.max_text_length} characters"
        )
        
        # Validate input
        is_valid, validation_message = config.validate_text_input(user_text)
        
        if not is_valid:
            st.error(validation_message)
            return
        elif validation_message:
            st.warning(validation_message)
        
        # Process button
        process_button = st.button(" Analyze Attention", type="primary", use_container_width=True)
    
    with col2:
        st.subheader(" Attention Visualization")
        
        if process_button or user_text:
            try:
                with st.spinner(f"Loading {selected_model_display} model..."):
                    # Process text and get attention
                    preprocessed_text = preprocess_text_for_model(user_text, selected_model)
                    
                    attention_data = process_text_cached(
                        selected_model, 
                        preprocessed_text, 
                        aggregation_method
                    )
                
                # Display model info
                col_info1, col_info2, col_info3 = st.columns(3)
                
                with col_info1:
                    st.markdown("""
                    <div class="metric-card">
                        <h4> Tokens</h4>
                        <h2>{}</h2>
                    </div>
                    """.format(attention_data['num_tokens']), unsafe_allow_html=True)
                
                with col_info2:
                    st.markdown("""
                    <div class="metric-card">
                        <h4> Layers</h4>
                        <h2>{}</h2>
                    </div>
                    """.format(attention_data['num_layers']), unsafe_allow_html=True)
                
                with col_info3:
                    st.markdown("""
                    <div class="metric-card">
                        <h4> Heads</h4>
                        <h2>{}</h2>
                    </div>
                    """.format(attention_data['num_heads']), unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Generate visualization based on type
                viz_type_map = {
                    "Attention Heatmap": "heatmap",
                    "Multi-Head View": "multi_head", 
                    "Attention Flow": "flow_graph",
                    "Layer Comparison": "layer_comparison"
                }
                
                viz_key = viz_type_map[viz_type]
                
                with st.spinner("Generating visualization..."):
                    if viz_key == "heatmap":
                        visualizer = AttentionVisualizer()
                        
                        if aggregation_method != "none":
                            fig = visualizer.create_attention_heatmap(
                                attention_data['aggregated_attention'],
                                attention_data['tokens'],
                                title=f"{selected_model_display} - {selected_agg_display}",
                                colorscale=colorscale
                            )
                        else:
                            fig = visualizer.create_attention_heatmap(
                                attention_data['attention_weights'][layer_idx, head_idx],
                                attention_data['tokens'],
                                title=f"{selected_model_display} - Layer {layer_idx+1}, Head {head_idx+1}",
                                colorscale=colorscale
                            )
                    
                    elif viz_key == "multi_head":
                        visualizer = AttentionVisualizer()
                        fig = visualizer.create_multi_head_visualization(
                            attention_data['attention_weights'],
                            attention_data['tokens'],
                            layer_idx=layer_idx
                        )
                    
                    elif viz_key == "flow_graph":
                        visualizer = AttentionVisualizer()
                        if aggregation_method != "none":
                            matrix = attention_data['aggregated_attention']
                        else:
                            matrix = attention_data['attention_weights'][layer_idx, head_idx]
                        
                        fig = visualizer.create_attention_flow_graph(
                            matrix, attention_data['tokens'], threshold=threshold
                        )
                    
                    elif viz_key == "layer_comparison":
                        visualizer = AttentionVisualizer()
                        fig = visualizer.create_layer_comparison(
                            attention_data['attention_weights'],
                            attention_data['tokens']
                        )
                
                # Display the visualization
                st.plotly_chart(fig, use_container_width=True)
                
                # Token information
                with st.expander(" Token Details"):
                    tokens_df_data = {
                        "Position": list(range(len(attention_data['tokens']))),
                        "Token": attention_data['tokens'],
                        "Token ID": attention_data['input_ids'].tolist()
                    }
                    
                    if aggregation_method != "none":
                        # Calculate token importance
                        from src.text_processing import TextProcessor
                        processor = TextProcessor(None)  # We don't need tokenizer for this
                        importance_scores = processor.get_token_importance_scores(
                            attention_data['aggregated_attention']
                        )
                        tokens_df_data["Importance"] = [f"{score:.3f}" for score in importance_scores]
                    
                    st.dataframe(tokens_df_data, use_container_width=True)
                
                # Download option
                st.download_button(
                    " Download Attention Data",
                    data=str(attention_data['attention_weights'].tolist()),
                    file_name=f"attention_weights_{selected_model}_{hash(user_text)}.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"❌ Error processing text: {str(e)}")
                st.error("Please try a different model or text.")
                
                # Show detailed error in expander for debugging
                with st.expander(" Debug Information"):
                    st.code(traceback.format_exc())
        
        else:
            # Show placeholder
            st.info(" Enter text and click 'Analyze Attention' to see visualization")
            
            # Show example visualization with dummy data
            st.subheader(" Example Output")
            
            # Create dummy heatmap
            dummy_tokens = ["[CLS]", "The", "cat", "sat", "on", "mat", "[SEP]"]
            dummy_attention = np.random.rand(len(dummy_tokens), len(dummy_tokens))
            dummy_attention = dummy_attention / dummy_attention.sum(axis=1, keepdims=True)
            
            visualizer = AttentionVisualizer(figure_width=600, figure_height=400)
            dummy_fig = visualizer.create_attention_heatmap(
                dummy_attention, 
                dummy_tokens, 
                title="Example: Attention Pattern",
                colorscale="Viridis"
            )
            
            st.plotly_chart(dummy_fig, use_container_width=True)
            st.caption("*This is just an example with random data*")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p> Built with Streamlit • Powered by HuggingFace Transformers</p>
        <p>Part of the LLM Interpretability & Dataset Engineering Suite</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()