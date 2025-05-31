from datasets import load_dataset
from typing import Iterator, Dict, Any, List
from pathlib import Path
import json
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

from .config import WikiConfig
from .cleaner import WikipediaTextCleaner
from .chunker import WikipediaChunker

class WikipediaDatasetBuilder:
    """Build cleaned Wikipedia datasets for LLM training and RAG."""
    
    def __init__(self, config: WikiConfig):
        self.config = config
        self.cleaner = WikipediaTextCleaner(config)
        self.chunker = WikipediaChunker(config)
        
        # Create output directory
        output_dir = Path(self.config.get('output.output_dir', './output'))
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
    
    def build_dataset(self) -> Dict[str, Any]:
        """Build the complete dataset."""
        
        logger.info("🚀 Starting Wikipedia dataset building...")
        
        # Load dataset
        dataset = self._load_dataset()
        if not dataset:
            return {'success': False, 'error': 'Failed to load dataset'}
        
        # Process articles
        processed_articles = self._process_articles(dataset)
        
        # Save results
        save_results = self._save_dataset(processed_articles)
        
        # Generate statistics
        stats = self._generate_statistics(processed_articles)
        
        logger.info("✅ Dataset building complete!")
        
        return {
            'success': True,
            'stats': stats,
            'save_results': save_results,
            'output_dir': str(self.output_dir)
        }
    
    def _load_dataset(self):
        """Load Wikipedia dataset."""
        try:
            logger.info(f"Loading Wikipedia dataset: {self.config.get('dataset.name')}")
            
            dataset = load_dataset(
                self.config.get('dataset.name'),
                self.config.get('dataset.version'),
                streaming=self.config.get('dataset.streaming', True),
                trust_remote_code=self.config.get('dataset.trust_remote_code', True)
            )
            
            logger.info("Dataset loaded successfully!")
            return dataset['train']  # Wikipedia only has train split
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return None
    
    def _process_articles(self, dataset) -> List[Dict[str, Any]]:
        """Process and clean articles."""
        
        max_articles = self.config.get('processing.max_articles', 1000)
        processed_articles = []
        
        logger.info(f"Processing up to {max_articles} articles...")
        
        with tqdm(total=max_articles, desc="Processing articles") as pbar:
            for i, article in enumerate(dataset):
                if i >= max_articles:
                    break
                
                # Validate article
                if not self.cleaner.is_valid_article(article):
                    logger.debug(f"Skipping invalid article: {article.get('title', 'Unknown')}")
                    pbar.update(1)
                    continue
                
                # Clean article
                try:
                    cleaned_article = self.cleaner.clean_article(article)
                    
                    # Chunk if enabled
                    chunks = self.chunker.chunk_article(cleaned_article)
                    processed_articles.extend(chunks)
                    
                    logger.debug(f"Processed: {article.get('title', 'Unknown')} -> {len(chunks)} chunks")
                    
                except Exception as e:
                    logger.error(f"Error processing article '{article.get('title', 'Unknown')}': {e}")
                
                pbar.update(1)
        
        logger.info(f"Processed {len(processed_articles)} article chunks")
        return processed_articles
    
    def _save_dataset(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Save processed dataset in specified format."""
        
        output_format = self.config.get('output.format', 'jsonl')
        filename_prefix = self.config.get('output.filename_prefix', 'wikipedia_clean')
        
        results = {}
        
        try:
            if output_format == 'jsonl':
                results['jsonl'] = self._save_jsonl(articles, filename_prefix)
            
            elif output_format == 'parquet':
                results['parquet'] = self._save_parquet(articles, filename_prefix)
            
            elif output_format == 'both':
                results['jsonl'] = self._save_jsonl(articles, filename_prefix)
                results['parquet'] = self._save_parquet(articles, filename_prefix)
            
            # Always save a sample
            results['sample'] = self._save_sample(articles[:10], filename_prefix)
            
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            results['error'] = str(e)
        
        return results
    
    def _save_jsonl(self, articles: List[Dict[str, Any]], prefix: str) -> str:
        """Save articles as JSONL format."""
        
        filepath = self.output_dir / f"{prefix}.jsonl"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for article in articles:
                json.dump(article, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Saved JSONL: {filepath} ({len(articles)} articles)")
        return str(filepath)
    
    def _save_parquet(self, articles: List[Dict[str, Any]], prefix: str) -> str:
        """Save articles as Parquet format."""
        
        filepath = self.output_dir / f"{prefix}.parquet"
        
        # Convert to PyArrow table
        table = pa.Table.from_pylist(articles)
        
        # Save as Parquet
        pq.write_table(table, filepath)
        
        logger.info(f"Saved Parquet: {filepath} ({len(articles)} articles)")
        return str(filepath)
    
    def _save_sample(self, articles: List[Dict[str, Any]], prefix: str) -> str:
        """Save sample articles for inspection."""
        
        filepath = self.output_dir / f"{prefix}_sample.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved sample: {filepath} ({len(articles)} articles)")
        return str(filepath)
    
    def _generate_statistics(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive statistics."""
        
        if not articles:
            return {'error': 'No articles processed'}
        
        # Basic stats
        total_articles = len(articles)
        total_length = sum(len(article.get('text', '')) for article in articles)
        avg_length = total_length / total_articles if total_articles > 0 else 0
        
        # Length distribution
        lengths = [len(article.get('text', '')) for article in articles]
        lengths.sort()
        
        stats = {
            'total_articles': total_articles,
            'total_characters': total_length,
            'avg_length': round(avg_length, 2),
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'median_length': lengths[len(lengths)//2] if lengths else 0,
            'chunking_enabled': self.config.get('processing.enable_chunking', False),
            'chunk_size': self.config.get('processing.chunk_size', 512),
            'cleaning_config': {
                'remove_citations': self.config.get('cleaning.remove_citations', True),
                'remove_infoboxes': self.config.get('cleaning.remove_infoboxes', True),
                'remove_references': self.config.get('cleaning.remove_references', True),
                'clean_markup': self.config.get('cleaning.clean_markup', True)
            }
        }
        
        # Save stats
        stats_file = self.output_dir / 'build_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics saved: {stats_file}")
        return stats