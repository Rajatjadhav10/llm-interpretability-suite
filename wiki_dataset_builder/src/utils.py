import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger
import hashlib
import time

def ensure_directory(path: str) -> Path:
    """Ensure directory exists, create if it doesn't."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def save_json(data: Dict[str, Any], filepath: str, indent: int = 2) -> bool:
    """Save data as JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.debug(f"Saved JSON: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON {filepath}: {e}")
        return False

def load_json(filepath: str) -> Optional[Dict[str, Any]]:
    """Load JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON {filepath}: {e}")
        return None

def get_file_hash(filepath: str) -> str:
    """Get MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Failed to hash file {filepath}: {e}")
        return ""

def format_bytes(bytes_value: int) -> str:
    """Format bytes into human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        }
    except ImportError:
        return {'rss_mb': 0, 'vms_mb': 0}

def create_sample_config() -> Dict[str, Any]:
    """Create a sample configuration for testing."""
    return {
        'dataset': {
            'name': 'wikipedia',
            'version': '20220301.en',
            'streaming': True,
            'trust_remote_code': True
        },
        'processing': {
            'max_articles': 100,  # Small for testing
            'min_article_length': 50,
            'max_article_length': 5000,
            'chunk_size': 256,
            'chunk_overlap': 32,
            'enable_chunking': True
        },
        'cleaning': {
            'remove_citations': True,
            'remove_infoboxes': True,
            'remove_references': True,
            'remove_external_links': True,
            'clean_markup': True
        },
        'output': {
            'format': 'jsonl',
            'output_dir': './test_output',
            'filename_prefix': 'wiki_test'
        },
        'logging': {
            'level': 'DEBUG',
            'log_file': './logs/test.log'
        }
    }

class Timer:
    """Simple timer context manager."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"⏱️  Starting: {self.description}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"✅ Completed: {self.description} in {format_duration(duration)}")
    
    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

def validate_article_data(article: Dict[str, Any]) -> List[str]:
    """Validate article data structure and return list of issues."""
    issues = []
    
    # Check required fields
    required_fields = ['id', 'title', 'text']
    for field in required_fields:
        if field not in article:
            issues.append(f"Missing required field: {field}")
        elif not article[field]:
            issues.append(f"Empty required field: {field}")
    
    # Check data types
    if 'id' in article and not isinstance(article['id'], (str, int)):
        issues.append("Field 'id' should be string or integer")
    
    if 'title' in article and not isinstance(article['title'], str):
        issues.append("Field 'title' should be string")
    
    if 'text' in article and not isinstance(article['text'], str):
        issues.append("Field 'text' should be string")
    
    # Check text length
    if 'text' in article and len(article['text']) < 10:
        issues.append("Article text too short (< 10 characters)")
    
    return issues

def clean_filename(filename: str) -> str:
    """Clean filename to be filesystem-safe."""
    # Remove or replace problematic characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    
    return filename.strip()

def estimate_processing_time(num_articles: int, articles_per_second: float = 5.0) -> str:
    """Estimate processing time based on number of articles."""
    total_seconds = num_articles / articles_per_second
    return format_duration(total_seconds)