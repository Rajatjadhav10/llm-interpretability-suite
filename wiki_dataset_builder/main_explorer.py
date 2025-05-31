from src.config import WikiConfig
from src.explorer import WikipediaExplorer
from loguru import logger
import sys
from pathlib import Path

def setup_logging(config: WikiConfig):
    """Setup logging configuration."""
    log_level = config.get('logging.level', 'INFO')
    log_file = config.get('logging.log_file', './logs/builder.log')
    
    # Create logs directory if it doesn't exist
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Remove default logger and add our configuration
    logger.remove()
    logger.add(sys.stderr, level=log_level, format="{time} | {level} | {message}")
    logger.add(log_file, level=log_level, format="{time} | {level} | {function} | {message}")

def main():
    """Main exploration function for Day 1."""
    print("🚀 Wikipedia Dataset Builder - Day 1: Foundation & Data Loading")
    print("=" * 60)
    
    # Load configuration
    config = WikiConfig()
    setup_logging(config)
    
    logger.info("Starting Wikipedia dataset exploration...")
    
    # Create explorer
    explorer = WikipediaExplorer(config)
    
    # Load dataset
    if not explorer.load_dataset():
        logger.error("Failed to load dataset. Exiting.")
        return
    
    # Inspect samples
    explorer.inspect_sample(num_samples=3)
    
    # Get statistics
    stats = explorer.get_dataset_stats(sample_size=50)
    
    # Save stats to file
    output_dir = Path(config.get('output.output_dir', './output'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats_file = output_dir / 'dataset_stats.json'
    with open(stats_file, 'w') as f:
        import json
        json.dump(stats, f, indent=2)
    
    logger.info(f"Dataset statistics saved to: {stats_file}")
    logger.info("Day 1 exploration complete! 🎉")
    
    print(f"\n✅ Day 1 Complete!")
    print(f"📊 Dataset statistics saved to: {stats_file}")
    print(f"📝 Logs saved to: {config.get('logging.log_file')}")
    print(f"\n🔥 Next: Run this script to see your Wikipedia data in action!")

if __name__ == "__main__":
    main()