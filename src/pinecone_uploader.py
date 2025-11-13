"""
Pinecone Uploader for RAG System (NEW API v3.x)
"""

import logging
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime
from tqdm import tqdm
import sys

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


class PineconeUploader:
    """Upload enriched chunks to Pinecone."""
    
    def __init__(self,
                 api_key: str,
                 index_name: str = "rag-chunks",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 cloud: str = "aws",
                 region: str = "us-east-1"):
        
        self.index_name = index_name
        self.cloud = cloud
        self.region = region
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        logger.info("[SUCCESS] Embedding model loaded")
        
        # Initialize Pinecone
        logger.info("Initializing Pinecone...")
        self.pc = Pinecone(api_key=api_key)
        
        # Setup index
        self._setup_index()
        
        # Statistics
        self.stats = {
            'total_chunks': 0,
            'uploaded': 0,
            'failed': 0
        }
    
    def _setup_index(self):
        """Create or connect to index."""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            logger.info(f"Creating new index: {self.index_name}")
            
            # Get embedding dimension
            sample = self.model.encode(["test"])
            dimension = sample.shape[1]
            
            # Create index
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self.cloud,
                    region=self.region
                )
            )
            
            # Wait for index to be ready
            import time
            logger.info("Waiting for index to be ready...")
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
        
        # Connect to index
        self.index = self.pc.Index(self.index_name)
        logger.info(f"[SUCCESS] Connected to index: {self.index_name}")
    
    def load_chunks(self, enriched_dir: str = "data/enriched") -> List[Dict]:
        """Load enriched chunks."""
        logger.info(f"Loading chunks from {enriched_dir}")
        
        chunks = []
        enriched_path = Path(enriched_dir)
        jsonl_files = list(enriched_path.glob("*_enriched.jsonl"))
        
        if not jsonl_files:
            logger.error(f"No enriched files found in {enriched_dir}")
            return []
        
        for file in tqdm(jsonl_files, desc="Loading files"):
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        chunks.append(json.loads(line))
        
        self.stats['total_chunks'] = len(chunks)
        logger.info(f"[SUCCESS] Loaded {len(chunks)} chunks from {len(jsonl_files)} files")
        return chunks
    
    def upload(self, chunks: List[Dict], batch_size: int = 100):
        """Upload chunks to Pinecone."""
        logger.info("Starting upload to Pinecone...")
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Uploading"):
            batch = chunks[i:i + batch_size]
            
            try:
                vectors = []
                
                for chunk in batch:
                    # Generate embedding
                    embedding = self.model.encode([chunk['text']])[0]
                    
                    # Prepare metadata
                    metadata = {
                        'text': chunk['text'][:1000],
                        'source_file': str(chunk.get('source_file', '')),
                        'chunk_index': int(chunk.get('chunk_index', 0)),
                        'token_count': int(chunk.get('token_count', 0)),
                        'quality_score': float(chunk.get('quality_score', 0)),
                        'keywords': ','.join(chunk.get('keywords', [])[:5]),
                        'primary_topic': str(chunk.get('primary_topic', '')),
                        'entity_count': int(chunk.get('entity_count', 0))
                    }
                    
                    # Add to batch
                    vectors.append({
                        'id': chunk['chunk_id'],
                        'values': embedding.tolist(),
                        'metadata': metadata
                    })
                
                # Upload batch
                self.index.upsert(vectors=vectors)
                self.stats['uploaded'] += len(batch)
                
            except Exception as e:
                logger.error(f"Error in batch {i}: {str(e)}")
                self.stats['failed'] += len(batch)
        
        # Print summary
        self._print_summary()
        self._save_report()
    
    def _print_summary(self):
        """Print upload summary."""
        logger.info("\n" + "="*70)
        logger.info("UPLOAD SUMMARY")
        logger.info("="*70)
        logger.info(f"Total chunks: {self.stats['total_chunks']}")
        logger.info(f"Uploaded: {self.stats['uploaded']}")
        logger.info(f"Failed: {self.stats['failed']}")
        
        if self.stats['total_chunks'] > 0:
            success_rate = (self.stats['uploaded'] / self.stats['total_chunks']) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
        
        logger.info("="*70 + "\n")
    
    def _save_report(self):
        """Save upload report."""
        report = {
            'database': 'pinecone',
            'index_name': self.index_name,
            'statistics': self.stats,
            'timestamp': datetime.now().isoformat()
        }
        
        Path('logs').mkdir(exist_ok=True)
        
        with open('logs/pinecone_upload.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("Report saved to: logs/pinecone_upload.json")


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Upload to Pinecone')
    parser.add_argument('--api-key', required=True, help='Pinecone API key')
    parser.add_argument('--index-name', default='rag-chunks', help='Index name')
    parser.add_argument('--enriched-dir', default='data/enriched', help='Enriched dir')
    parser.add_argument('--cloud', default='aws', choices=['aws', 'gcp', 'azure'])
    parser.add_argument('--region', default='us-east-1', help='Cloud region')
    parser.add_argument('--batch-size', type=int, default=100)
    
    args = parser.parse_args()
    
    try:
        # Initialize uploader
        uploader = PineconeUploader(
            api_key=args.api_key,
            index_name=args.index_name,
            cloud=args.cloud,
            region=args.region
        )
        
        # Load chunks
        chunks = uploader.load_chunks(args.enriched_dir)
        
        if not chunks:
            print("\n[ERROR] No chunks found!")
            sys.exit(1)
        
        # Upload
        uploader.upload(chunks, batch_size=args.batch_size)
        
        print("\n[SUCCESS] Upload complete!")
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()