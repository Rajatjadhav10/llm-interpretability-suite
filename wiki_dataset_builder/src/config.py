import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class WikiConfig:
    """Configuration management for Wikipedia dataset builder."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_default_config()
        
        if config_path and Path(config_path).exists():
            self.config.update(self._load_config_file(config_path))
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration settings."""
        return {
            'dataset': {
                'name': 'wikipedia',
                'version': '20220301.en',
                'streaming': True,
                'trust_remote_code': True
            },
            'processing': {
                'max_articles': 1000,  # For testing, increase for production
                'min_article_length': 100,
                'max_article_length': 50000,
                'chunk_size': 512,  # For RAG chunking
                'enable_chunking': False
            },
            'cleaning': {
                'remove_citations': True,
                'remove_infoboxes': True,
                'remove_references': True,
                'remove_external_links': True,
                'clean_markup': True
            },
            'output': {
                'format': 'jsonl',  # Options: jsonl, parquet, hf_dataset
                'output_dir': './output',
                'filename_prefix': 'wikipedia_clean'
            },
            'logging': {
                'level': 'INFO',
                'log_file': './logs/builder.log'
            }
        }
    
    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation (e.g., 'dataset.name')."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value