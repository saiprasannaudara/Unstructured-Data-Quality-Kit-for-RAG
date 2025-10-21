"""
Master RAG Data Quality Pipeline
Runs all steps end-to-end with monitoring and reporting
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime
import time
import sys

# Import all pipeline components
# Note: Adjust imports based on your actual file structure
try:
    sys.path.append('src')
    from ingestion import DocumentIngestionValidator
    from text_cleaning import TextCleaner
    from quality_assessment import ContentQualityAssessor
    from semantic_dedup import SemanticDeduplicator
    from document_chunking import DocumentChunker
    from metadata_enrichment import MetadataEnricher
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Make sure all pipeline scripts are in the 'src' directory")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/master_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set console output encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


class MasterPipeline:
    """
    Master pipeline orchestrator for RAG data quality.
    Runs all steps with monitoring and error handling.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the master pipeline.
        
        Args:
            config: Configuration dictionary for all steps
        """
        self.config = config or self._default_config()
        self.results = {
            'pipeline_start': datetime.now().isoformat(),
            'steps': {},
            'overall_status': 'pending'
        }
        
        # Create all necessary directories
        self._setup_directories()
    
    def _default_config(self) -> Dict:
        """Return default configuration for all pipeline steps."""
        return {
            'ingestion': {
                'raw_data_dir': 'data/raw',
                'processed_dir': 'data/processed',
                'failed_dir': 'data/failed'
            },
            'cleaning': {
                'processed_dir': 'data/processed',
                'cleaned_dir': 'data/cleaned',
                'keep_urls': False,
                'keep_emails': False
            },
            'quality_assessment': {
                'cleaned_dir': 'data/cleaned',
                'quality_dir': 'data/quality_assessed',
                'quality_threshold': 0.5
            },
            'deduplication': {
                'input_dir': 'data/quality_assessed/high_quality',
                'output_dir': 'data/deduplicated',
                'exact_match': True,
                'fuzzy_match': True,
                'semantic_match': True,
                'semantic_threshold': 0.95,
                'fuzzy_threshold': 0.85
            },
            'chunking': {
                'input_dir': 'data/deduplicated',
                'output_dir': 'data/chunked',
                'chunk_size': 512,
                'chunk_overlap': 50,
                'chunking_strategy': 'recursive',
                'min_chunk_size': 100,
                'max_chunk_size': 1000
            },
            'enrichment': {
                'input_dir': 'data/chunked',
                'output_dir': 'data/enriched',
                'enable_keywords': True,
                'enable_entities': True,
                'enable_summaries': True,
                'enable_questions': True,
                'enable_topics': True,
                'max_keywords': 5
            }
        }
    
    def _setup_directories(self):
        """Create all required directories."""
        dirs = [
            'data/raw', 'data/processed', 'data/cleaned', 
            'data/quality_assessed/high_quality', 'data/quality_assessed/low_quality',
            'data/deduplicated', 'data/chunked', 'data/enriched',
            'data/failed', 'logs'
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def run_step(self, step_name: str, step_function, step_config: Dict) -> Dict:
        """
        Run a single pipeline step with error handling and timing.
        
        Args:
            step_name: Name of the step
            step_function: Function to execute
            step_config: Configuration for the step
            
        Returns:
            Dictionary with step results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"STEP: {step_name.upper()}")
        logger.info(f"{'='*70}")
        
        start_time = time.time()
        
        try:
            result = step_function(**step_config)
            elapsed_time = time.time() - start_time
            
            self.results['steps'][step_name] = {
                'status': 'success',
                'elapsed_time': round(elapsed_time, 2),
                'result': result
            }
            
            logger.info(f"[SUCCESS] {step_name} completed in {elapsed_time:.2f}s")
            return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"[ERROR] {step_name} failed: {str(e)}")
            
            self.results['steps'][step_name] = {
                'status': 'failed',
                'elapsed_time': round(elapsed_time, 2),
                'error': str(e)
            }
            
            raise
    
    def run_full_pipeline(self, skip_steps: List[str] = None) -> Dict:
        """
        Run the complete pipeline from ingestion to enrichment.
        
        Args:
            skip_steps: List of step names to skip
            
        Returns:
            Dictionary with all results
        """
        skip_steps = skip_steps or []
        
        logger.info("\n" + "="*70)
        logger.info("STARTING MASTER RAG DATA QUALITY PIPELINE")
        logger.info("="*70)
        logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        try:
            # Step 1: Document Ingestion
            if 'ingestion' not in skip_steps:
                ingestion_system = DocumentIngestionValidator(**self.config['ingestion'])
                self.run_step('ingestion', ingestion_system.process_all_files, {})
            
            # Step 2: Text Cleaning
            if 'cleaning' not in skip_steps:
                cleaner = TextCleaner(
                    processed_dir=self.config['cleaning']['processed_dir'],
                    cleaned_dir=self.config['cleaning']['cleaned_dir']
                )
                self.run_step('cleaning', cleaner.process_all_files, {
                    'keep_urls': self.config['cleaning']['keep_urls'],
                    'keep_emails': self.config['cleaning']['keep_emails']
                })
            
            # Step 3: Quality Assessment
            if 'quality_assessment' not in skip_steps:
                assessor = ContentQualityAssessor(**self.config['quality_assessment'])
                self.run_step('quality_assessment', assessor.assess_all_files, {})
            
            # Step 4: Semantic Deduplication
            if 'deduplication' not in skip_steps:
                deduplicator = SemanticDeduplicator(**self.config['deduplication'])
                self.run_step('deduplication', deduplicator.run_deduplication, {})
            
            # Step 5: Document Chunking
            if 'chunking' not in skip_steps:
                chunker = DocumentChunker(**self.config['chunking'])
                self.run_step('chunking', chunker.process_all_files, {
                    'output_format': 'jsonl'
                })
            
            # Step 6: Metadata Enrichment
            if 'enrichment' not in skip_steps:
                enricher = MetadataEnricher(**self.config['enrichment'])
                self.run_step('enrichment', enricher.process_all_files, {})
            
            # Mark as successful
            self.results['overall_status'] = 'success'
            self.results['pipeline_end'] = datetime.now().isoformat()
            
            # Calculate total time
            start = datetime.fromisoformat(self.results['pipeline_start'])
            end = datetime.fromisoformat(self.results['pipeline_end'])
            self.results['total_time'] = (end - start).total_seconds()
            
            # Generate final report
            self._generate_final_report()
            
            logger.info("\n" + "="*70)
            logger.info("[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*70)
            
            return self.results
            
        except Exception as e:
            self.results['overall_status'] = 'failed'
            self.results['pipeline_end'] = datetime.now().isoformat()
            self.results['error'] = str(e)
            
            logger.error("\n" + "="*70)
            logger.error(f"[ERROR] PIPELINE FAILED: {str(e)}")
            logger.error("="*70)
            
            return self.results
    
    def _generate_final_report(self):
        """Generate comprehensive final report."""
        logger.info("\n" + "="*70)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*70)
        
        # Overall stats
        logger.info(f"\nTotal Pipeline Time: {self.results['total_time']:.2f}s")
        logger.info(f"Status: {self.results['overall_status']}")
        
        # Step-by-step breakdown
        logger.info("\nStep-by-Step Results:")
        for step_name, step_data in self.results['steps'].items():
            status_icon = "[SUCCESS]" if step_data['status'] == 'success' else "[FAILED]"
            logger.info(f"  {status_icon} {step_name}: {step_data['elapsed_time']}s")
        
        # Extract key metrics
        logger.info("\nKey Metrics:")
        
        try:
            # Ingestion
            if 'ingestion' in self.results['steps']:
                logger.info("  Ingestion:")
                logger.info(f"    Files processed: {self.results['steps']['ingestion']['result']}")
            
            # Quality Assessment
            if 'quality_assessment' in self.results['steps']:
                qa_result = self.results['steps']['quality_assessment']['result']
                logger.info("  Quality Assessment:")
                logger.info(f"    Files assessed: {qa_result}")
            
            # Deduplication
            if 'deduplication' in self.results['steps']:
                dedup_result = self.results['steps']['deduplication']['result']
                if 'statistics' in dedup_result:
                    stats = dedup_result['statistics']
                    logger.info("  Deduplication:")
                    logger.info(f"    Total files: {stats.get('total_files', 'N/A')}")
                    logger.info(f"    Unique files: {stats.get('unique_files', 'N/A')}")
                    logger.info(f"    Duplicates removed: {stats.get('total_duplicates_removed', 'N/A')}")
            
            # Chunking
            if 'chunking' in self.results['steps']:
                chunk_result = self.results['steps']['chunking']['result']
                if 'statistics' in chunk_result:
                    stats = chunk_result['statistics']
                    logger.info("  Chunking:")
                    logger.info(f"    Total chunks: {stats.get('total_chunks', 'N/A')}")
                    logger.info(f"    Avg chunks per doc: {stats.get('avg_chunks_per_doc', 'N/A'):.1f}")
            
            # Enrichment
            if 'enrichment' in self.results['steps']:
                enrich_result = self.results['steps']['enrichment']['result']
                if 'statistics' in enrich_result:
                    stats = enrich_result['statistics']
                    logger.info("  Enrichment:")
                    logger.info(f"    Chunks enriched: {stats.get('enriched_chunks', 'N/A')}")
        
        except Exception as e:
            logger.warning(f"Could not extract all metrics: {str(e)}")
        
        logger.info("="*70)
        
        # Save detailed report
        report_path = 'logs/pipeline_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"\nDetailed report saved to: {report_path}")
    
    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status."""
        return {
            'status': self.results['overall_status'],
            'completed_steps': [
                name for name, data in self.results['steps'].items() 
                if data['status'] == 'success'
            ],
            'failed_steps': [
                name for name, data in self.results['steps'].items() 
                if data['status'] == 'failed'
            ],
            'total_time': self.results.get('total_time', 0)
        }


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run RAG Data Quality Pipeline')
    parser.add_argument('--config', help='Path to configuration JSON file')
    parser.add_argument('--skip', nargs='+', 
                       choices=['ingestion', 'cleaning', 'quality_assessment', 
                               'deduplication', 'chunking', 'enrichment'],
                       help='Steps to skip')
    parser.add_argument('--chunk-size', type=int, help='Override chunk size')
    parser.add_argument('--quality-threshold', type=float, help='Override quality threshold')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = None
    
    # Initialize pipeline
    pipeline = MasterPipeline(config=config)
    
    # Apply command-line overrides
    if args.chunk_size:
        pipeline.config['chunking']['chunk_size'] = args.chunk_size
    if args.quality_threshold:
        pipeline.config['quality_assessment']['quality_threshold'] = args.quality_threshold
    
    # Run pipeline
    results = pipeline.run_full_pipeline(skip_steps=args.skip or [])
    
    # Print final status
    status = pipeline.get_pipeline_status()
    print("\n" + "="*70)
    print("FINAL STATUS")
    print("="*70)
    print(f"Overall Status: {status['status']}")
    print(f"Completed Steps: {', '.join(status['completed_steps'])}")
    if status['failed_steps']:
        print(f"Failed Steps: {', '.join(status['failed_steps'])}")
    print(f"Total Time: {status['total_time']:.2f}s")
    print("="*70)
    
    # Exit with appropriate code
    sys.exit(0 if status['status'] == 'success' else 1)


if __name__ == "__main__":
    main()