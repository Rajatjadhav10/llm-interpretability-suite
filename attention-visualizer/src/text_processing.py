import re
from typing import List, Dict, Tuple, Optional
import numpy as np
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    """Handle text preprocessing and tokenization for attention visualization."""
    
    def __init__(self, tokenizer):
        """
        Initialize with a tokenizer.
        
        Args:
            tokenizer: HuggingFace tokenizer instance
        """
        self.tokenizer = tokenizer
    
    def clean_text(self, text: str) -> str:
        """
        Clean input text for better tokenization.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s.,!?;:\-\'\"()]', '', text)
        
        return text
    
    def tokenize_with_mapping(self, text: str) -> Dict:
        """
        Tokenize text and create mapping between tokens and original words.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Dictionary with tokenization info and mappings
        """
        # Clean text first
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        encoding = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
            padding=False,
            truncation=True,
            max_length=512
        )
        
        # Get tokens as strings
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        
        # Create word-to-token mapping if offset mapping is available
        word_token_mapping = []
        if 'offset_mapping' in encoding:
            word_token_mapping = self._create_word_token_mapping(
                cleaned_text, tokens, encoding['offset_mapping'][0]
            )
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'input_ids': encoding['input_ids'][0].numpy(),
            'attention_mask': encoding['attention_mask'][0].numpy(),
            'word_token_mapping': word_token_mapping,
            'num_tokens': len(tokens)
        }
    
    def _create_word_token_mapping(self, text: str, tokens: List[str], 
                                   offsets: List[Tuple[int, int]]) -> List[Dict]:
        """
        Create mapping between words and tokens.
        
        Args:
            text: Original text
            tokens: List of tokens
            offsets: Token offset positions
            
        Returns:
            List of mapping dictionaries
        """
        mappings = []
        words = text.split()
        current_word_idx = 0
        current_char_pos = 0
        
        for token_idx, (token, (start, end)) in enumerate(zip(tokens, offsets)):
            # Skip special tokens
            if token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
                mappings.append({
                    'token_idx': token_idx,
                    'token': token,
                    'word_idx': -1,
                    'word': None,
                    'is_special': True
                })
                continue
            
            # Find which word this token belongs to
            word_idx = self._find_word_for_token(start, end, words, text)
            
            mappings.append({
                'token_idx': token_idx,
                'token': token,
                'word_idx': word_idx,
                'word': words[word_idx] if word_idx < len(words) else None,
                'is_special': False,
                'char_start': start,
                'char_end': end
            })
        
        return mappings
    
    def _find_word_for_token(self, char_start: int, char_end: int, 
                            words: List[str], text: str) -> int:
        """Find which word index a token belongs to."""
        if char_start == 0 and char_end == 0:
            return -1
        
        current_pos = 0
        for word_idx, word in enumerate(words):
            word_start = text.find(word, current_pos)
            word_end = word_start + len(word)
            
            if char_start >= word_start and char_end <= word_end + 1:
                return word_idx
            
            current_pos = word_end
        
        return len(words) - 1  # Default to last word
    
    def format_tokens_for_display(self, tokens: List[str]) -> List[str]:
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
    
    def get_token_importance_scores(self, attention_weights: np.ndarray, 
                                   method: str = "sum") -> np.ndarray:
        """
        Calculate importance scores for tokens based on attention.
        
        Args:
            attention_weights: 2D attention matrix (seq_len x seq_len)
            method: Scoring method ("sum", "mean", "max", "entropy")
            
        Returns:
            1D array of importance scores per token
        """
        if method == "sum":
            # Sum of attention received by each token
            scores = np.sum(attention_weights, axis=0)
        elif method == "mean":
            # Average attention received by each token
            scores = np.mean(attention_weights, axis=0)
        elif method == "max":
            # Maximum attention received by each token
            scores = np.max(attention_weights, axis=0)
        elif method == "entropy":
            # Entropy of attention distribution from each token
            scores = -np.sum(attention_weights * np.log(attention_weights + 1e-12), axis=1)
        else:
            raise ValueError(f"Unknown scoring method: {method}")
        
        return scores
    
    def highlight_important_tokens(self, tokens: List[str], 
                                  scores: np.ndarray, 
                                  top_k: int = 5) -> List[Tuple[str, float, bool]]:
        """
        Identify and highlight the most important tokens.
        
        Args:
            tokens: List of tokens
            scores: Importance scores per token
            top_k: Number of top tokens to highlight
            
        Returns:
            List of (token, score, is_highlighted) tuples
        """
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:]
        
        result = []
        for idx, (token, score) in enumerate(zip(tokens, scores)):
            is_highlighted = idx in top_indices
            result.append((token, float(score), is_highlighted))
        
        return result

def preprocess_text_for_model(text: str, model_name: str) -> str:
    """
    Preprocess text based on model requirements.
    
    Args:
        text: Input text
        model_name: Name of the model
        
    Returns:
        Preprocessed text
    """
    # Basic cleaning
    text = text.strip()
    
    # Model-specific preprocessing
    if "gpt" in model_name.lower():
        # GPT models handle text as-is mostly
        pass
    elif "bert" in model_name.lower():
        # BERT models benefit from sentence-like structure
        if not text.endswith(('.', '!', '?')):
            text += '.'
    
    return text

def get_token_colors(scores: np.ndarray, colorscale: str = "viridis") -> List[str]:
    """
    Generate colors for tokens based on importance scores.
    
    Args:
        scores: Importance scores
        colorscale: Color scale name
        
    Returns:
        List of hex color codes
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    # Normalize scores to 0-1
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    
    # Get colormap
    cmap = plt.get_cmap(colorscale.lower())
    
    # Generate colors
    colors = []
    for score in norm_scores:
        rgba = cmap(score)
        hex_color = mcolors.to_hex(rgba)
        colors.append(hex_color)
    
    return colors

# Utility functions
def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def split_long_text(text: str, max_tokens: int = 450) -> List[str]:
    """Split long text into smaller chunks for processing."""
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        # Rough estimate: 1 word ≈ 1.3 tokens
        if len(current_chunk) * 1.3 > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        current_chunk.append(word)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # Test text processing
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    processor = TextProcessor(tokenizer)
    
    test_text = "The quick brown fox jumps over the lazy dog."
    result = processor.tokenize_with_mapping(test_text)
    
    print(f"Original: {result['original_text']}")
    print(f"Tokens: {result['tokens']}")
    print(f"Formatted: {processor.format_tokens_for_display(result['tokens'])}")