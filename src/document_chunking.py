import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import re

# Text processing
import spacy
import tiktoken
from tqdm import tqdm
import numpy as np

# LangChain for chunking
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SpacyTextSplitter
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/chunking.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set console output encoding for Windows compatibility
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


class DocumentChunker:
    """
    Advanced document chunking system for RAG applications.
    Supports multiple chunking strategies with quality validation.
    """
    
    def __init__(self,
                 input_dir: str = "data/deduplicated",
                 output_dir: str = "data/chunked",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 chunking_strategy: str = "recursive",
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 1000,
                 encoding_name: str = "cl100k_base"):
        """
        Initialize the document chunker.
        
        Args:
            input_dir: Directory with deduplicated documents
            output_dir: Directory to save chunked documents
            chunk_size: Target chunk size in tokens
            chunk_overlap: Number of tokens to overlap between chunks
            chunking_strategy: Strategy to use ("fixed", "recursive", "semantic")
            min_chunk_size: Minimum acceptable chunk size
            max_chunk_size: Maximum acceptable chunk size
            encoding_name: Tokenizer encoding (for OpenAI models)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Chunking parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Initialize tokenizer
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.warning(f"Could not load {encoding_name}, using cl100k_base")
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Load spaCy model (lazy loading)
        self.nlp = None
        
        # Initialize text splitter based on strategy
        self.text_splitter = self._initialize_splitter()
        
        # Statistics tracking
        self.stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'avg_chunks_per_doc': 0,
            'avg_chunk_size': [],
            'chunks_too_small': 0,
            'chunks_too_large': 0,
            'chunk_size_distribution': []
        }
    
    def _initialize_splitter(self):
        """Initialize the appropriate text splitter based on strategy."""
        
        if self.chunking_strategy == "fixed":
            logger.info("Using Fixed-Size Character Splitter")
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size * 4,  # Approximate: 1 token ≈ 4 chars
                chunk_overlap=self.chunk_overlap * 4,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        
        elif self.chunking_strategy == "recursive":
            logger.info("Using Recursive Character Splitter")
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size * 4,
                chunk_overlap=self.chunk_overlap * 4,
                length_function=len,
                separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
        
        elif self.chunking_strategy == "semantic":
            logger.info("Using Semantic (Sentence-Based) Splitter")
            # Load spaCy for sentence splitting
            self.nlp = self._load_spacy()
            return SpacyTextSplitter(
                chunk_size=self.chunk_size * 4,
                chunk_overlap=self.chunk_overlap * 4,
                pipeline="en_core_web_sm"
            )
        
        elif self.chunking_strategy == "token":
            logger.info("Using Token-Based Splitter")
            return TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                encoding_name="cl100k_base"
            )
        
        else:
            logger.warning(f"Unknown strategy '{self.chunking_strategy}', using recursive")
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size * 4,
                chunk_overlap=self.chunk_overlap * 4
            )
    
    def _load_spacy(self):
        """Load spaCy model for sentence splitting."""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            return nlp
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {str(e)}")
            logger.info("Run: python -m spacy download en_core_web_sm")
            return None
    
    # ===================== TOKEN COUNTING =====================
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text."""
        try:
            tokens = self.encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Error counting tokens: {str(e)}")
            # Fallback: estimate 1 token ≈ 4 characters
            return len(text) // 4
    
    # ===================== CHUNKING METHODS =====================
    
    def chunk_document(self, text: str, filename: str) -> List[Dict]:
        """
        Split document into chunks using the configured strategy.
        
        Args:
            text: Document text content
            filename: Original filename for metadata
            
        Returns:
            List of chunk dictionaries with metadata
        """
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create chunk objects with metadata
            chunk_objects = []
            
            for i, chunk_text in enumerate(chunks):
                # Count tokens
                token_count = self.count_tokens(chunk_text)
                
                # Validate chunk size
                is_valid = self.min_chunk_size <= token_count <= self.max_chunk_size
                
                if token_count < self.min_chunk_size:
                    self.stats['chunks_too_small'] += 1
                elif token_count > self.max_chunk_size:
                    self.stats['chunks_too_large'] += 1
                
                # Create chunk object
                chunk_obj = {
                    'chunk_id': f"{filename.replace('.txt', '')}_{i}",
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'text': chunk_text,
                    'token_count': token_count,
                    'char_count': len(chunk_text),
                    'source_file': filename,
                    'is_valid': is_valid,
                    'chunking_strategy': self.chunking_strategy
                }
                
                # Add quality metrics
                chunk_obj.update(self._assess_chunk_quality(chunk_text))
                
                chunk_objects.append(chunk_obj)
                self.stats['avg_chunk_size'].append(token_count)
            
            return chunk_objects
            
        except Exception as e:
            logger.error(f"Error chunking document {filename}: {str(e)}")
            return []
    
    # ===================== CHUNK QUALITY ASSESSMENT =====================
    
    def _assess_chunk_quality(self, chunk_text: str) -> Dict:
        """
        Assess the quality of a single chunk.
        
        Returns:
            Dictionary with quality metrics
        """
        # Check if chunk ends mid-sentence
        ends_complete = chunk_text.rstrip()[-1] in '.!?'
        
        # Check if chunk starts mid-sentence (simple heuristic)
        starts_complete = chunk_text.lstrip()[0].isupper() if chunk_text.strip() else False
        
        # Count sentences
        sentence_count = len([s for s in re.split(r'[.!?]+', chunk_text) if s.strip()])
        
        # Calculate word count
        word_count = len(chunk_text.split())
        
        # Check for incomplete words (hyphenation at boundaries)
        has_hyphen_start = chunk_text.lstrip().startswith('-')
        has_hyphen_end = chunk_text.rstrip().endswith('-')
        
        return {
            'ends_complete': ends_complete,
            'starts_complete': starts_complete,
            'sentence_count': sentence_count,
            'word_count': word_count,
            'has_boundary_issues': has_hyphen_start or has_hyphen_end,
            'quality_score': self._calculate_chunk_quality_score(
                ends_complete, starts_complete, has_hyphen_start or has_hyphen_end
            )
        }
    
    def _calculate_chunk_quality_score(self, ends_complete: bool, 
                                      starts_complete: bool, 
                                      has_boundary_issues: bool) -> float:
        """Calculate a quality score for the chunk (0-1)."""
        score = 1.0
        
        if not ends_complete:
            score -= 0.3
        if not starts_complete:
            score -= 0.2
        if has_boundary_issues:
            score -= 0.2
        
        return max(0.0, score)
    
    # ===================== METADATA ENRICHMENT =====================
    
    def enrich_chunk_metadata(self, chunk: Dict, doc_metadata: Dict = None) -> Dict:
        """
        Add additional metadata to chunks for better retrieval.
        
        Args:
            chunk: Chunk dictionary
            doc_metadata: Optional document-level metadata
            
        Returns:
            Enriched chunk dictionary
        """
        # Add document metadata if available
        if doc_metadata:
            chunk['document_metadata'] = doc_metadata
        
        # Add position information
        chunk['position_ratio'] = chunk['chunk_index'] / max(1, chunk['total_chunks'])
        
        # Categorize position
        if chunk['position_ratio'] < 0.25:
            chunk['position_category'] = 'beginning'
        elif chunk['position_ratio'] < 0.75:
            chunk['position_category'] = 'middle'
        else:
            chunk['position_category'] = 'end'
        
        # Add timestamp
        chunk['created_at'] = datetime.now().isoformat()
        
        return chunk
    
    # ===================== FILE PROCESSING =====================
    
    def process_file(self, filepath: Path) -> List[Dict]:
        """
        Process a single file: load, chunk, and enrich.
        
        Args:
            filepath: Path to the document file
            
        Returns:
            List of chunk dictionaries
        """
        logger.info(f"Processing file: {filepath.name}")
        
        try:
            # Read file
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Create document metadata
            doc_metadata = {
                'source_file': filepath.name,
                'file_size': filepath.stat().st_size,
                'char_count': len(text),
                'estimated_tokens': self.count_tokens(text)
            }
            
            # Chunk the document
            chunks = self.chunk_document(text, filepath.name)
            
            # Enrich chunks with metadata
            enriched_chunks = []
            for chunk in chunks:
                enriched_chunk = self.enrich_chunk_metadata(chunk, doc_metadata)
                enriched_chunks.append(enriched_chunk)
            
            # Update statistics
            self.stats['total_documents'] += 1
            self.stats['total_chunks'] += len(chunks)
            
            logger.info(f"[SUCCESS] Created {len(chunks)} chunks from {filepath.name}")
            
            return enriched_chunks
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to process {filepath.name}: {str(e)}")
            return []
    
    def save_chunks(self, chunks: List[Dict], output_format: str = "jsonl"):
        """
        Save chunks to file.
        
        Args:
            chunks: List of chunk dictionaries
            output_format: Format to save ("jsonl", "json", or "txt")
        """
        if not chunks:
            return
        
        # Get source filename
        source_file = chunks[0]['source_file']
        base_name = source_file.replace('_deduplicated.txt', '').replace('.txt', '')
        
        if output_format == "jsonl":
            # Save as JSONL (one chunk per line)
            output_path = self.output_dir / f"{base_name}_chunks.jsonl"
            with open(output_path, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        
        elif output_format == "json":
            # Save as JSON array
            output_path = self.output_dir / f"{base_name}_chunks.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        elif output_format == "txt":
            # Save as plain text (for review)
            output_path = self.output_dir / f"{base_name}_chunks.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    f.write(f"=== CHUNK {chunk['chunk_index'] + 1}/{chunk['total_chunks']} ===\n")
                    f.write(f"Tokens: {chunk['token_count']} | Quality: {chunk['quality_score']:.2f}\n")
                    f.write(f"{'-'*60}\n")
                    f.write(chunk['text'])
                    f.write(f"\n{'='*60}\n\n")
        
        logger.debug(f"Saved {len(chunks)} chunks to {output_path}")
    
    def process_all_files(self, output_format: str = "jsonl") -> Dict:
        """
        Process all documents in the input directory.
        
        Args:
            output_format: Format to save chunks ("jsonl", "json", or "txt")
            
        Returns:
            Dictionary with processing results and statistics
        """
        logger.info(f"Starting document chunking from {self.input_dir}")
        logger.info(f"Strategy: {self.chunking_strategy} | Chunk size: {self.chunk_size} tokens | Overlap: {self.chunk_overlap} tokens")
        
        # Get all text files
        all_files = list(self.input_dir.glob('*.txt'))
        logger.info(f"Found {len(all_files)} files to process")
        
        if not all_files:
            logger.warning("No files found to process")
            return {'status': 'no_files'}
        
        # Process each file
        all_chunks = []
        
        for filepath in tqdm(all_files, desc="Chunking documents"):
            chunks = self.process_file(filepath)
            
            if chunks:
                # Save chunks for this document
                self.save_chunks(chunks, output_format)
                all_chunks.extend(chunks)
        
        # Calculate statistics
        if self.stats['avg_chunk_size']:
            self.stats['avg_chunks_per_doc'] = self.stats['total_chunks'] / max(1, self.stats['total_documents'])
            self.stats['mean_chunk_size'] = int(np.mean(self.stats['avg_chunk_size']))
            self.stats['median_chunk_size'] = int(np.median(self.stats['avg_chunk_size']))
            self.stats['std_chunk_size'] = int(np.std(self.stats['avg_chunk_size']))
        
        # Generate reports
        self.generate_chunking_report(all_chunks)
        self.print_summary()
        
        return {
            'status': 'success',
            'statistics': self.stats,
            'total_chunks': len(all_chunks)
        }
    
    # ===================== REPORTING =====================
    
    def generate_chunking_report(self, all_chunks: List[Dict]):
        """Generate comprehensive chunking report."""
        
        # Create summary statistics
        report_data = []
        
        # Group chunks by document
        from collections import defaultdict
        docs = defaultdict(list)
        for chunk in all_chunks:
            docs[chunk['source_file']].append(chunk)
        
        for source_file, chunks in docs.items():
            avg_quality = np.mean([c['quality_score'] for c in chunks])
            avg_tokens = np.mean([c['token_count'] for c in chunks])
            
            report_data.append({
                'source_file': source_file,
                'num_chunks': len(chunks),
                'avg_tokens_per_chunk': int(avg_tokens),
                'avg_quality_score': round(avg_quality, 3),
                'chunks_too_small': sum(1 for c in chunks if c['token_count'] < self.min_chunk_size),
                'chunks_too_large': sum(1 for c in chunks if c['token_count'] > self.max_chunk_size)
            })
        
        # Save as CSV
        import pandas as pd
        df = pd.DataFrame(report_data)
        df = df.sort_values('num_chunks', ascending=False)
        
        report_path = 'logs/chunking_report.csv'
        df.to_csv(report_path, index=False)
        logger.info(f"Chunking report saved to: {report_path}")
        
        # Save detailed chunk metadata
        metadata_path = self.output_dir / 'chunks_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'chunking_config': {
                    'strategy': self.chunking_strategy,
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap,
                    'min_chunk_size': self.min_chunk_size,
                    'max_chunk_size': self.max_chunk_size
                },
                'statistics': self.stats,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
    
    def print_summary(self):
        """Print chunking statistics."""
        logger.info("\n" + "="*60)
        logger.info("CHUNKING SUMMARY")
        logger.info("="*60)
        logger.info(f"Strategy: {self.chunking_strategy}")
        logger.info(f"Total documents processed: {self.stats['total_documents']}")
        logger.info(f"Total chunks created: {self.stats['total_chunks']}")
        logger.info(f"Average chunks per document: {self.stats['avg_chunks_per_doc']:.1f}")
        
        if 'mean_chunk_size' in self.stats:
            logger.info(f"\nChunk Size Statistics:")
            logger.info(f"  Mean: {self.stats['mean_chunk_size']} tokens")
            logger.info(f"  Median: {self.stats['median_chunk_size']} tokens")
            logger.info(f"  Std Dev: {self.stats['std_chunk_size']} tokens")
        
        logger.info(f"\nQuality Issues:")
        logger.info(f"  Chunks too small (<{self.min_chunk_size}): {self.stats['chunks_too_small']}")
        logger.info(f"  Chunks too large (>{self.max_chunk_size}): {self.stats['chunks_too_large']}")
        
        logger.info("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Initialize chunker with your preferred strategy
    chunker = DocumentChunker(
        input_dir="data/deduplicated",
        output_dir="data/chunked",
        chunk_size=512,              # Target: 512 tokens per chunk
        chunk_overlap=50,            # Overlap: 50 tokens
        chunking_strategy="recursive",  # Options: "fixed", "recursive", "semantic", "token"
        min_chunk_size=100,          # Minimum acceptable size
        max_chunk_size=1000          # Maximum acceptable size
    )
    
    # Process all documents
    results = chunker.process_all_files(output_format="jsonl")  # Options: "jsonl", "json", "txt"
    
    # Save results
    with open('logs/chunking_results.json', 'w') as f:
        json.dump({
            'status': results['status'],
            'statistics': results['statistics']
        }, f, indent=2)
    
    print("\nChunking complete!")
    print(f"Chunked files: data/chunked/")
    print(f"Chunking report: logs/chunking_report.csv")
    print(f"Metadata: data/chunked/chunks_metadata.json")