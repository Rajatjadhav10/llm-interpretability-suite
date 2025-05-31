"""Text cleaning utilities for Wikipedia articles."""

import re
from typing import Dict, Any, Optional
from loguru import logger
import html
from .config import WikiConfig
class WikipediaTextCleaner:
    """Clean and preprocess Wikipedia article text."""
    
    def __init__(self, config: 'WikiConfig'):
        self.config = config
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for text cleaning."""
        
        # Citation patterns like [1], [2], [citation needed]
        self.citation_pattern = re.compile(r'\[\d+\]|\[citation needed\]|\[clarification needed\]')
        
        # Reference sections
        self.reference_pattern = re.compile(r'==\s*References\s*==.*?(?=\n==|\Z)', re.DOTALL | re.IGNORECASE)
        
        # External links sections
        self.external_links_pattern = re.compile(r'==\s*External links\s*==.*?(?=\n==|\Z)', re.DOTALL | re.IGNORECASE)
        
        # See also sections  
        self.see_also_pattern = re.compile(r'==\s*See also\s*==.*?(?=\n==|\Z)', re.DOTALL | re.IGNORECASE)
        
        # Infobox patterns
        self.infobox_pattern = re.compile(r'\{\{Infobox.*?\}\}', re.DOTALL | re.IGNORECASE)
        
        # Template patterns like {{template}}
        self.template_pattern = re.compile(r'\{\{[^}]*\}\}')
        
        # File/Image patterns
        self.file_pattern = re.compile(r'\[\[File:.*?\]\]|\[\[Image:.*?\]\]', re.DOTALL | re.IGNORECASE)
        
        # Category patterns
        self.category_pattern = re.compile(r'\[\[Category:.*?\]\]', re.IGNORECASE)
        
        # Wiki markup patterns
        self.bold_pattern = re.compile(r"'''(.*?)'''")
        self.italic_pattern = re.compile(r"''(.*?)''")
        self.link_pattern = re.compile(r'\[\[([^|\]]*)\|?([^\]]*)\]\]')
        
        # HTML entities and tags
        self.html_tag_pattern = re.compile(r'<[^>]+>')
        
        # Multiple whitespace
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Section headers
        self.header_pattern = re.compile(r'^=+\s*.*?\s*=+$', re.MULTILINE)
    
    def clean_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Clean a Wikipedia article based on configuration settings."""
        
        cleaned_article = article.copy()
        text = article.get('text', '')
        
        if not text:
            logger.warning(f"Empty text for article: {article.get('title', 'Unknown')}")
            return cleaned_article
        
        # Apply cleaning steps based on configuration
        if self.config.get('cleaning.remove_citations', True):
            text = self._remove_citations(text)
        
        if self.config.get('cleaning.remove_infoboxes', True):
            text = self._remove_infoboxes(text)
        
        if self.config.get('cleaning.remove_references', True):
            text = self._remove_references(text)
        
        if self.config.get('cleaning.remove_external_links', True):
            text = self._remove_external_links(text)
        
        if self.config.get('cleaning.clean_markup', True):
            text = self._clean_markup(text)
        
        # Final cleanup
        text = self._final_cleanup(text)
        
        # Update the article
        cleaned_article['text'] = text
        cleaned_article['original_length'] = len(article.get('text', ''))
        cleaned_article['cleaned_length'] = len(text)
        cleaned_article['cleaning_stats'] = self._get_cleaning_stats(article.get('text', ''), text)
        
        return cleaned_article
    
    def _remove_citations(self, text: str) -> str:
        """Remove citation markers like [1], [2], [citation needed]."""
        return self.citation_pattern.sub('', text)
    
    def _remove_infoboxes(self, text: str) -> str:
        """Remove infobox templates."""
        return self.infobox_pattern.sub('', text)
    
    def _remove_references(self, text: str) -> str:
        """Remove references sections."""
        text = self.reference_pattern.sub('', text)
        text = self.see_also_pattern.sub('', text)
        return text
    
    def _remove_external_links(self, text: str) -> str:
        """Remove external links sections."""
        return self.external_links_pattern.sub('', text)
    
    def _clean_markup(self, text: str) -> str:
        """Clean various Wikipedia markup."""
        
        # Remove templates
        text = self.template_pattern.sub('', text)
        
        # Remove files and images
        text = self.file_pattern.sub('', text)
        
        # Remove categories
        text = self.category_pattern.sub('', text)
        
        # Clean wiki links - keep the display text
        text = self.link_pattern.sub(lambda m: m.group(2) if m.group(2) else m.group(1), text)
        
        # Remove bold and italic markup
        text = self.bold_pattern.sub(r'\1', text)
        text = self.italic_pattern.sub(r'\1', text)
        
        # Remove section headers
        text = self.header_pattern.sub('', text)
        
        # Remove HTML tags
        text = self.html_tag_pattern.sub('', text)
        
        # Decode HTML entities
        text = html.unescape(text)
        
        return text
    
    def _final_cleanup(self, text: str) -> str:
        """Final text cleanup and normalization."""
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        return text
    
    def _get_cleaning_stats(self, original: str, cleaned: str) -> Dict[str, Any]:
        """Generate cleaning statistics."""
        return {
            'original_length': len(original),
            'cleaned_length': len(cleaned),
            'reduction_ratio': 1 - (len(cleaned) / len(original)) if original else 0,
            'citations_removed': len(self.citation_pattern.findall(original)),
            'templates_removed': len(self.template_pattern.findall(original)),
            'files_removed': len(self.file_pattern.findall(original))
        }
    
    def is_valid_article(self, article: Dict[str, Any]) -> bool:
        """Check if article meets quality criteria."""
        
        text = article.get('text', '')
        text_length = len(text)
        
        min_length = self.config.get('processing.min_article_length', 100)
        max_length = self.config.get('processing.max_article_length', 50000)
        
        # Check length requirements
        if text_length < min_length or text_length > max_length:
            return False
        
        # Check for redirect pages
        if text.strip().lower().startswith('#redirect'):
            return False
        
        # Check for disambiguation pages
        if 'disambiguation' in article.get('title', '').lower():
            return False
        
        return True
