import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    BertTokenizer, 
    BertModel,
    GPT2Tokenizer, 
    GPT2Model,
    DistilBertTokenizer, 
    DistilBertModel
)
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionExtractor:
    """Extract and process attention weights from transformer models."""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        Initialize the attention extractor with a specific model.
        
        Args:
            model_name: HuggingFace model name or path
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._load_model()
    
    def _load_model(self):
        """Load the specified model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer and model based on model type
            if "bert" in self.model_name.lower() and "distil" not in self.model_name.lower():
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                self.model = BertModel.from_pretrained(
                    self.model_name, 
                    output_attentions=True,
                    cache_dir="./models/"
                )
            elif "distilbert" in self.model_name.lower():
                self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
                self.model = DistilBertModel.from_pretrained(
                    self.model_name, 
                    output_attentions=True,
                    cache_dir="./models/"
                )
            elif "gpt2" in self.model_name.lower():
                self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
                # GPT-2 doesn't have a pad token by default
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = GPT2Model.from_pretrained(
                    self.model_name, 
                    output_attentions=True,
                    cache_dir="./models/"
                )
            else:
                # Fallback to Auto classes
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(
                    self.model_name, 
                    output_attentions=True,
                    cache_dir="./models/"
                )
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")
            raise
    
    def get_attention_weights(self, text: str) -> Dict:
        """
        Extract attention weights for given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing tokens, attention weights, and metadata
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model outputs with attention
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract attention weights
            attention_weights = outputs.attentions  # Tuple of attention matrices
            
            # Convert tokens back to strings
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # Process attention weights
            processed_attention = self._process_attention_weights(attention_weights)
            
            return {
                "tokens": tokens,
                "attention_weights": processed_attention,
                "input_ids": inputs["input_ids"][0].cpu().numpy(),
                "model_name": self.model_name,
                "num_layers": len(attention_weights),
                "num_heads": attention_weights[0].shape[1],
                "num_tokens": len(tokens)
            }
            
        except Exception as e:
            logger.error(f"Error extracting attention weights: {str(e)}")
            raise
    
    def _process_attention_weights(self, attention_weights: Tuple) -> np.ndarray:
        """
        Process raw attention weights into numpy arrays.
        
        Args:
            attention_weights: Tuple of attention tensors from model
            
        Returns:
            Numpy array of shape (num_layers, num_heads, seq_len, seq_len)
        """
        # Convert to numpy and remove batch dimension
        processed = []
        for layer_attention in attention_weights:
            # Shape: (batch_size, num_heads, seq_len, seq_len) -> (num_heads, seq_len, seq_len)
            layer_np = layer_attention[0].cpu().numpy()
            processed.append(layer_np)
        
        # Stack all layers: (num_layers, num_heads, seq_len, seq_len)
        return np.stack(processed)
    
    def get_aggregated_attention(self, text: str, method: str = "mean") -> Dict:
        """
        Get aggregated attention weights across heads/layers.
        
        Args:
            text: Input text
            method: Aggregation method ("mean", "max", "last_layer")
            
        Returns:
            Dictionary with aggregated attention weights
        """
        attention_data = self.get_attention_weights(text)
        attention_weights = attention_data["attention_weights"]
        
        if method == "mean":
            # Average across all heads and layers
            aggregated = np.mean(attention_weights, axis=(0, 1))
        elif method == "max":
            # Max across all heads and layers
            aggregated = np.max(attention_weights, axis=(0, 1))
        elif method == "last_layer":
            # Average across heads in the last layer
            aggregated = np.mean(attention_weights[-1], axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        attention_data["aggregated_attention"] = aggregated
        attention_data["aggregation_method"] = method
        
        return attention_data

def get_available_models() -> List[str]:
    """Return list of supported model names."""
    return [
        "bert-base-uncased",
        "bert-large-uncased", 
        "distilbert-base-uncased",
        "gpt2",
        "gpt2-medium"
    ]

def load_model_safe(model_name: str) -> Optional[AttentionExtractor]:
    """
    Safely load a model with error handling.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        AttentionExtractor instance or None if loading fails
    """
    try:
        return AttentionExtractor(model_name)
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        return None

# Example usage and testing
if __name__ == "__main__":
    # Test the attention extractor
    print("Testing AttentionExtractor...")
    
    # Test with BERT
    extractor = AttentionExtractor("bert-base-uncased")
    
    test_text = "The cat sat on the mat."
    print(f"\nAnalyzing: '{test_text}'")
    
    # Get attention weights
    attention_data = extractor.get_attention_weights(test_text)
    print(f"Tokens: {attention_data['tokens']}")
    print(f"Attention shape: {attention_data['attention_weights'].shape}")
    print(f"Model: {attention_data['model_name']}")
    print(f"Layers: {attention_data['num_layers']}, Heads: {attention_data['num_heads']}")
    
    # Get aggregated attention
    agg_data = extractor.get_aggregated_attention(test_text, method="last_layer")
    print(f"Aggregated attention shape: {agg_data['aggregated_attention'].shape}")