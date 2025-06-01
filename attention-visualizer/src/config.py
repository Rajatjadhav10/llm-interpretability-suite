import os
from dataclasses import dataclass
from typing import List, Dict, Any
import streamlit as st

@dataclass
class ModelConfig:
    """Configuration for model settings."""
    default_model: str = "bert-base-uncased"
    available_models: List[str] = None
    max_sequence_length: int = 512
    cache_dir: str = "./models/"
    
    def __post_init__(self):
        if self.available_models is None:
            self.available_models = [
                "bert-base-uncased",
                "bert-large-uncased",
                "distilbert-base-uncased", 
                "gpt2",
                "gpt2-medium"
            ]

@dataclass 
class VisualizationConfig:
    """Configuration for visualization settings."""
    default_colorscale: str = "Viridis"
    available_colorscales: List[str] = None
    figure_width: int = 800
    figure_height: int = 600
    heatmap_font_size: int = 10
    show_token_labels: bool = True
    aggregation_methods: List[str] = None
    
    def __post_init__(self):
        if self.available_colorscales is None:
            self.available_colorscales = [
                "Viridis", "Plasma", "Blues", "Reds", "YlOrRd", 
                "RdYlBu", "Spectral", "Cividis"
            ]
        
        if self.aggregation_methods is None:
            self.aggregation_methods = [
                "mean", "max", "last_layer"
            ]

@dataclass
class AppConfig:
    """Main application configuration."""
    app_title: str = "🔍 LLM Attention Visualizer"
    app_description: str = """
    Visualize attention patterns in transformer models. Enter text to see how different 
    tokens attend to each other across layers and heads.
    """
    
    # UI Settings
    sidebar_width: int = 300
    main_width: int = 700
    max_text_length: int = 500
    
    # Default example texts
    example_texts: List[str] = None
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    
    def __post_init__(self):
        if self.example_texts is None:
            self.example_texts = [
                "The cat sat on the mat.",
                "Attention is all you need for understanding transformers.",
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning models learn patterns from data.",
                "Natural language processing enables computers to understand human language."
            ]

class ConfigManager:
    """Manages all configuration objects and Streamlit settings."""
    
    def __init__(self):
        self.model_config = ModelConfig()
        self.viz_config = VisualizationConfig() 
        self.app_config = AppConfig()
        
        # Set up Streamlit page config
        self._setup_streamlit_config()
    
    def _setup_streamlit_config(self):
        """Configure Streamlit page settings."""
        try:
            st.set_page_config(
                page_title=self.app_config.app_title,
                page_icon="🔍",
                layout="wide",
                initial_sidebar_state="expanded"
            )
        except Exception:
            # Page config already set
            pass
    
    def get_model_options(self) -> Dict[str, str]:
        """Get model options for UI dropdown."""
        return {
            "BERT Base": "bert-base-uncased",
            "BERT Large": "bert-large-uncased", 
            "DistilBERT": "distilbert-base-uncased",
            "GPT-2": "gpt2",
            "GPT-2 Medium": "gpt2-medium"
        }
    
    def get_colorscale_options(self) -> List[str]:
        """Get available colorscale options."""
        return self.viz_config.available_colorscales
    
    def get_aggregation_options(self) -> Dict[str, str]:
        """Get aggregation method options for UI."""
        return {
            "Average All": "mean",
            "Maximum": "max", 
            "Last Layer": "last_layer"
        }
    
    def validate_text_input(self, text: str) -> tuple[bool, str]:
        """
        Validate user text input.
        
        Args:
            text: User input text
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not text or not text.strip():
            return False, "Please enter some text to analyze."
        
        if len(text) > self.app_config.max_text_length:
            return False, f"Text too long. Maximum {self.app_config.max_text_length} characters allowed."
        
        # Check for potentially problematic characters
        if any(ord(c) > 127 for c in text):
            return True, "Text contains non-ASCII characters. Results may vary."
        
        return True, ""
    
    def get_cache_key(self, model_name: str, text: str) -> str:
        """Generate cache key for attention weights."""
        return f"{model_name}_{hash(text)}"

# Global configuration instance
config = ConfigManager()

# Streamlit caching configuration
@st.cache_data(ttl=config.app_config.cache_ttl)
def cached_attention_extraction(model_name: str, text: str):
    """Cached version of attention extraction for Streamlit."""
    from .model_utils import AttentionExtractor
    
    extractor = AttentionExtractor(model_name)
    return extractor.get_attention_weights(text)

# Environment-based overrides
def load_env_config():
    """Load configuration overrides from environment variables."""
    # Model settings
    if os.getenv("DEFAULT_MODEL"):
        config.model_config.default_model = os.getenv("DEFAULT_MODEL")
    
    if os.getenv("MODEL_CACHE_DIR"):
        config.model_config.cache_dir = os.getenv("MODEL_CACHE_DIR")
    
    # App settings  
    if os.getenv("MAX_TEXT_LENGTH"):
        try:
            config.app_config.max_text_length = int(os.getenv("MAX_TEXT_LENGTH"))
        except ValueError:
            pass
    
    if os.getenv("DISABLE_CACHING"):
        config.app_config.enable_caching = False

# Load environment config on import
load_env_config()

# Export main objects
__all__ = [
    "config", 
    "ModelConfig", 
    "VisualizationConfig", 
    "AppConfig", 
    "ConfigManager",
    "cached_attention_extraction"
]