from typing import List, Dict, Any, Optional
from loguru import logger
import re
from .config import WikiConfig

class WikipediaChunker:
    """Chunk Wikipedia articles for RAG systems."""
    
    def __init__(self, config: 'WikiConfig'):
        self.config = config
        self.chunk_size = config.get('processing.chunk_size', 512)
        self.overlap_size = config.get('processing.chunk_overlap', 50)
    
    def chunk_article(self, article: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk an article into smaller pieces for RAG."""
        
        if not self.config.get('processing.enable_chunking', False):
            return [article]  # Return as single chunk
        
        text = article.get('text', '')
        if not text:
            return []
        
        # Try different chunking strategies
        chunks = self._chunk_by_sentences(text)
        
        if not chunks:
            chunks = self._chunk_by_characters(text)
        
        # Create chunk objects
        chunked_articles = []
        for i, chunk_text in enumerate(chunks):
            chunk = {
                'id': f"{article.get('id', 'unknown')}_{i}",
                'title': article.get('title', ''),
                'text': chunk_text,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'url': article.get('url', ''),
                'original_length': len(article.get('text', '')),
                'chunk_length': len(chunk_text)
            }
            chunked_articles.append(chunk)
        
        logger.debug(f"Chunked '{article.get('title', 'Unknown')}' into {len(chunks)} pieces")
        return chunked_articles
    
    def _chunk_by_sentences(self, text: str) -> List[str]:
        """Chunk text by sentences, respecting chunk size limits."""
        
        # Simple sentence splitting (can be improved with spaCy/NLTK)
        sentences = re.split(r'[.!?]+\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Handle sentences longer than chunk size
                if len(sentence) > self.chunk_size:
                    # Split long sentence by character limit
                    chunks.extend(self._chunk_by_characters(sentence))
                    current_chunk = ""
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_by_characters(self, text: str) -> List[str]:
        """Chunk text by character count with overlap."""
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If this isn't the last chunk, try to break at word boundary
            if end < len(text):
                # Look for the last space before the end position
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.overlap_size
            if start <= 0 or start >= end:  # Prevent infinite loops
                start = end
        
        return chunks
    
    def get_chunking_stats(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about chunking process."""
        
        if not articles:
            return {
                'original_articles': 0,
                'total_chunks': 0,
                'avg_chunks_per_article': 0,
                'chunk_size': self.chunk_size,
                'overlap_size': self.overlap_size
            }
        
        total_original = len(articles)
        total_chunks = sum(len(self.chunk_article(article)) for article in articles)
        
        return {
            'original_articles': total_original,
            'total_chunks': total_chunks,
            'avg_chunks_per_article': total_chunks / total_original if total_original > 0 else 0,
            'chunk_size': self.chunk_size,
            'overlap_size': self.overlap_size
        }