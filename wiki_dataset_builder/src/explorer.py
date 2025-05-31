from datasets import load_dataset
from loguru import logger
from typing import Iterator, Dict, Any
import json
from .config import WikiConfig
class WikipediaExplorer:
    """Explore and understand the Wikipedia dataset structure."""
    
    def __init__(self, config: 'WikiConfig'):
        self.config = config
        self.dataset = None
    
    def load_dataset(self):
        """Load the Wikipedia dataset in streaming mode."""
        try:
            logger.info(f"Loading Wikipedia dataset: {self.config.get('dataset.name')}")
            
            self.dataset = load_dataset(
                self.config.get('dataset.name'),
                self.config.get('dataset.version'),
                streaming=self.config.get('dataset.streaming', True),
                trust_remote_code=self.config.get('dataset.trust_remote_code', True)
            )
            
            logger.info("Dataset loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return False
    
    def inspect_sample(self, num_samples: int = 3) -> None:
        """Inspect the first few samples to understand data structure."""
        if not self.dataset:
            logger.error("Dataset not loaded. Call load_dataset() first.")
            return
        
        logger.info(f"Inspecting first {num_samples} samples...")
        
        # Get the train split (Wikipedia only has train split)
        train_data = self.dataset['train']
        
        for i, sample in enumerate(train_data):
            if i >= num_samples:
                break
                
            logger.info(f"\n--- Sample {i+1} ---")
            logger.info(f"Title: {sample.get('title', 'N/A')}")
            logger.info(f"Text length: {len(sample.get('text', ''))}")
            logger.info(f"ID: {sample.get('id', 'N/A')}")
            logger.info(f"URL: {sample.get('url', 'N/A')}")
            
            # Show first 200 characters of text
            text = sample.get('text', '')
            preview = text[:200] + "..." if len(text) > 200 else text
            logger.info(f"Text preview: {preview}")
            
            # Show available keys
            logger.info(f"Available keys: {list(sample.keys())}")
    
    def get_dataset_stats(self, sample_size: int = 100) -> Dict[str, Any]:
        """Get basic statistics about the dataset."""
        if not self.dataset:
            logger.error("Dataset not loaded. Call load_dataset() first.")
            return {}
        
        logger.info(f"Calculating dataset statistics from {sample_size} samples...")
        
        stats = {
            'total_samples_checked': 0,
            'avg_text_length': 0,
            'min_text_length': float('inf'),
            'max_text_length': 0,
            'common_keys': set(),
            'sample_titles': []
        }
        
        text_lengths = []
        train_data = self.dataset['train']
        
        for i, sample in enumerate(train_data):
            if i >= sample_size:
                break
                
            stats['total_samples_checked'] += 1
            
            # Text length stats
            text_len = len(sample.get('text', ''))
            text_lengths.append(text_len)
            stats['min_text_length'] = min(stats['min_text_length'], text_len)
            stats['max_text_length'] = max(stats['max_text_length'], text_len)
            
            # Common keys
            if i == 0:
                stats['common_keys'] = set(sample.keys())
            else:
                stats['common_keys'] = stats['common_keys'].intersection(sample.keys())
            
            # Sample titles
            if i < 10:
                stats['sample_titles'].append(sample.get('title', 'N/A'))
        
        # Calculate average
        if text_lengths:
            stats['avg_text_length'] = sum(text_lengths) / len(text_lengths)
        
        # Convert set to list for JSON serialization
        stats['common_keys'] = list(stats['common_keys'])
        
        logger.info("Dataset Statistics:")
        logger.info(f"  Samples checked: {stats['total_samples_checked']}")
        logger.info(f"  Average text length: {stats['avg_text_length']:.0f}")
        logger.info(f"  Min text length: {stats['min_text_length']}")
        logger.info(f"  Max text length: {stats['max_text_length']}")
        logger.info(f"  Common keys: {stats['common_keys']}")
        
        return stats
