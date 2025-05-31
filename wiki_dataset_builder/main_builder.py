import argparse
import sys
from pathlib import Path
from loguru import logger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import WikiConfig
from src.builder import WikipediaDatasetBuilder
from src.explorer import WikipediaExplorer

def setup_logging(config: WikiConfig):
    """Setup logging configuration."""
    log_level = config.get('logging.level', 'INFO')
    log_file = config.get('logging.log_file', './logs/builder.log')
    
    # Create logs directory
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure loguru
    logger.remove()  # Remove default handler
    logger.add(sys.stdout, level=log_level, colorize=True)
    logger.add(log_file, level=log_level, rotation="10 MB")

def explore_dataset(config: WikiConfig, num_samples: int = 3):
    """Explore the Wikipedia dataset."""
    logger.info("🔍 Exploring Wikipedia dataset...")
    
    explorer = WikipediaExplorer(config)
    
    if not explorer.load_dataset():
        logger.error("Failed to load dataset for exploration")
        return False
    
    # Inspect samples
    explorer.inspect_sample(num_samples)
    
    # Get statistics
    stats = explorer.get_dataset_stats(sample_size=100)
    
    # Save exploration results
    import json
    output_dir = Path(config.get('output.output_dir', './output'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'exploration_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"✅ Exploration complete! Stats saved to {output_dir / 'exploration_stats.json'}")
    return True

def build_dataset(config: WikiConfig):
    """Build the cleaned Wikipedia dataset."""
    logger.info("🚀 Starting Wikipedia dataset building...")
    
    builder = WikipediaDatasetBuilder(config)
    results = builder.build_dataset()
    
    if results['success']:
        logger.info("✅ Dataset building completed successfully!")
        logger.info(f"📊 Statistics:")
        stats = results['stats']
        logger.info(f"  • Total articles: {stats.get('total_articles', 0)}")
        logger.info(f"  • Total characters: {stats.get('total_characters', 0):,}")
        logger.info(f"  • Average length: {stats.get('avg_length', 0):.0f}")
        logger.info(f"  • Output directory: {results['output_dir']}")
        
        if stats.get('chunking_enabled'):
            logger.info(f"  • Chunking: Enabled (size: {stats.get('chunk_size', 0)})")
        else:
            logger.info(f"  • Chunking: Disabled")
            
        return True
    else:
        logger.error(f"❌ Dataset building failed: {results.get('error', 'Unknown error')}")
        return False

def validate_config(config_path: str) -> bool:
    """Validate configuration file."""
    if not Path(config_path).exists():
        logger.error(f"Configuration file not found: {config_path}")
        return False
    
    try:
        config = WikiConfig(config_path)
        required_keys = [
            'dataset.name',
            'processing.max_articles',
            'output.format',
            'output.output_dir'
        ]
        
        for key in required_keys:
            if config.get(key) is None:
                logger.error(f"Missing required configuration: {key}")
                return False
        
        logger.info("✅ Configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Wikipedia Dataset Builder - Build clean datasets for LLM training and RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explore dataset structure
  python main_builder.py explore
  
  # Build dataset with default config
  python main_builder.py build
  
  # Build with custom config
  python main_builder.py build --config configs/rag.yaml
  
  # Validate configuration
  python main_builder.py validate --config configs/rag.yaml
        """
    )
    
    parser.add_argument(
        'command',
        choices=['explore', 'build', 'validate'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/default.yaml',
        help='Configuration file path (default: configs/default.yaml)'
    )
    
    parser.add_argument(
        '--samples', '-s',
        type=int,
        default=3,
        help='Number of samples to inspect during exploration (default: 3)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = WikiConfig(args.config if Path(args.config).exists() else None)
        if args.verbose:
            config.config['logging']['level'] = 'DEBUG'
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Setup logging
    setup_logging(config)
    
    # Execute command
    success = True
    
    if args.command == 'validate':
        success = validate_config(args.config)
    
    elif args.command == 'explore':
        success = explore_dataset(config, args.samples)
    
    elif args.command == 'build':
        # Validate config first
        if not validate_config(args.config):
            sys.exit(1)
        success = build_dataset(config)
    
    if not success:
        sys.exit(1)
    
    logger.info("🎉 All operations completed successfully!")

if __name__ == "__main__":
    main()