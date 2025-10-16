import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set
import json
from datetime import datetime
from collections import defaultdict
import re

# Data processing
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys, os
if os.name == "nt":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Similarity and deduplication
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash, MinHashLSH
from sentence_transformers import SentenceTransformer

from pathlib import Path
Path("logs").mkdir(parents=True, exist_ok=True)
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/deduplication.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SemanticDeduplicator:
    """
    Comprehensive deduplication system for RAG documents.
    Uses exact matching, fuzzy matching, and semantic similarity.
    """
    
    def __init__(self, 
                 input_dir: str = "data/quality_assessed/high_quality",
                 output_dir: str = "data/deduplicated",
                 exact_match: bool = True,
                 fuzzy_match: bool = True,
                 semantic_match: bool = True,
                 semantic_threshold: float = 0.95,
                 fuzzy_threshold: float = 0.85,
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the deduplication system.
        
        Args:
            input_dir: Directory with quality-assessed documents
            output_dir: Directory to save deduplicated files
            exact_match: Enable exact duplicate detection
            fuzzy_match: Enable fuzzy duplicate detection
            semantic_match: Enable semantic duplicate detection
            semantic_threshold: Similarity threshold for semantic duplicates (0-1)
            fuzzy_threshold: Similarity threshold for fuzzy duplicates (0-1)
            model_name: Sentence transformer model for embeddings
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Deduplication settings
        self.exact_match = exact_match
        self.fuzzy_match = fuzzy_match
        self.semantic_match = semantic_match
        self.semantic_threshold = semantic_threshold
        self.fuzzy_threshold = fuzzy_threshold
        
        # Load embedding model (lazy loading - only if needed)
        self.model = None
        self.model_name = model_name
        
        # Statistics tracking
        self.stats = {
            'total_files': 0,
            'unique_files': 0,
            'exact_duplicates': 0,
            'fuzzy_duplicates': 0,
            'semantic_duplicates': 0,
            'total_duplicates_removed': 0
        }
        
        # Storage for deduplication results
        self.file_hashes = {}  # filename -> hash
        self.hash_to_files = defaultdict(list)  # hash -> [filenames]
        self.duplicate_groups = []  # List of duplicate file groups
    
    def load_embedding_model(self):
        """Lazy load the sentence transformer model."""
        if self.model is None and self.semantic_match:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("✓ Model loaded successfully")
    
    def calculate_file_hash(self, text: str) -> str:
        """Calculate MD5 hash of text content."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def normalize_text_for_comparison(self, text: str) -> str:
        """
        Normalize text for fair comparison.
        Removes minor differences that shouldn't count as "different".
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation variations
        text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()
    
    # ===================== EXACT MATCHING =====================
    
    def find_exact_duplicates(self, files_data: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Find exact duplicates using hash comparison.
        
        Args:
            files_data: Dictionary of {filename: text_content}
            
        Returns:
            Dictionary of {hash: [list of duplicate filenames]}
        """
        logger.info("Finding exact duplicates...")
        
        hash_groups = defaultdict(list)
        
        for filename, text in tqdm(files_data.items(), desc="Hashing files"):
            # Normalize text before hashing
            normalized_text = self.normalize_text_for_comparison(text)
            file_hash = self.calculate_file_hash(normalized_text)
            
            self.file_hashes[filename] = file_hash
            hash_groups[file_hash].append(filename)
        
        # Filter to only groups with duplicates
        duplicate_groups = {h: files for h, files in hash_groups.items() if len(files) > 1}
        
        num_duplicates = sum(len(files) - 1 for files in duplicate_groups.values())
        self.stats['exact_duplicates'] = num_duplicates
        
        logger.info(f"✓ Found {len(duplicate_groups)} exact duplicate groups "
                   f"({num_duplicates} duplicate files)")
        
        return duplicate_groups
    
    # ===================== FUZZY MATCHING =====================
    
    def create_minhash(self, text: str, num_perm: int = 128) -> MinHash:
        """
        Create MinHash signature for fuzzy matching.
        
        Args:
            text: Text content
            num_perm: Number of permutations (higher = more accurate but slower)
            
        Returns:
            MinHash object
        """
        minhash = MinHash(num_perm=num_perm)
        
        # Create shingles (character n-grams)
        # Using 3-character shingles
        for i in range(len(text) - 2):
            shingle = text[i:i+3]
            minhash.update(shingle.encode('utf-8'))
        
        return minhash
    
    def find_fuzzy_duplicates(self, files_data: Dict[str, str], 
                             exclude_files: Set[str] = None) -> List[List[str]]:
        """
        Find fuzzy duplicates using MinHash LSH.
        
        Args:
            files_data: Dictionary of {filename: text_content}
            exclude_files: Files already marked as exact duplicates
            
        Returns:
            List of duplicate file groups
        """
        logger.info("Finding fuzzy duplicates with MinHash LSH...")
        
        if exclude_files is None:
            exclude_files = set()
        
        # Filter out already identified duplicates
        remaining_files = {f: text for f, text in files_data.items() 
                          if f not in exclude_files}
        
        if len(remaining_files) < 2:
            logger.info("Not enough files for fuzzy matching")
            return []
        
        # Create LSH index
        lsh = MinHashLSH(threshold=self.fuzzy_threshold, num_perm=128)
        
        # Create MinHash signatures
        minhashes = {}
        for filename, text in tqdm(remaining_files.items(), desc="Creating MinHashes"):
            normalized_text = self.normalize_text_for_comparison(text)
            minhash = self.create_minhash(normalized_text)
            minhashes[filename] = minhash
            lsh.insert(filename, minhash)
        
        # Find similar documents
        duplicate_groups = []
        processed = set()
        
        for filename, minhash in minhashes.items():
            if filename in processed:
                continue
            
            # Query for similar documents
            similar = lsh.query(minhash)
            
            if len(similar) > 1:
                duplicate_groups.append(list(similar))
                processed.update(similar)
        
        num_duplicates = sum(len(group) - 1 for group in duplicate_groups)
        self.stats['fuzzy_duplicates'] = num_duplicates
        
        logger.info(f"✓ Found {len(duplicate_groups)} fuzzy duplicate groups "
                   f"({num_duplicates} duplicate files)")
        
        return duplicate_groups
    
    # ===================== SEMANTIC MATCHING =====================
    
    def find_semantic_duplicates(self, files_data: Dict[str, str],
                                 exclude_files: Set[str] = None,
                                 batch_size: int = 32) -> List[List[str]]:
        """
        Find semantic duplicates using embedding similarity.
        
        Args:
            files_data: Dictionary of {filename: text_content}
            exclude_files: Files already marked as duplicates
            batch_size: Batch size for embedding generation
            
        Returns:
            List of duplicate file groups
        """
        logger.info("Finding semantic duplicates with embeddings...")
        
        # Load model
        self.load_embedding_model()
        
        if exclude_files is None:
            exclude_files = set()
        
        # Filter out already identified duplicates
        remaining_files = {f: text for f, text in files_data.items() 
                          if f not in exclude_files}
        
        if len(remaining_files) < 2:
            logger.info("Not enough files for semantic matching")
            return []
        
        # Prepare data
        filenames = list(remaining_files.keys())
        texts = [remaining_files[f] for f in filenames]
        
        # Truncate texts if too long (model limit is usually 512 tokens)
        # Take first 1000 characters as representative sample
        texts_sample = [text[:1000] for text in texts]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.model.encode(texts_sample, 
                                      batch_size=batch_size,
                                      show_progress_bar=True,
                                      convert_to_numpy=True)
        
        # Calculate pairwise similarities
        logger.info("Calculating pairwise similarities...")
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find duplicate groups
        duplicate_groups = []
        processed = set()
        
        for i in range(len(filenames)):
            if i in processed:
                continue
            
            # Find documents with similarity above threshold
            similar_indices = np.where(similarity_matrix[i] >= self.semantic_threshold)[0]
            
            if len(similar_indices) > 1:
                group = [filenames[idx] for idx in similar_indices]
                duplicate_groups.append(group)
                processed.update(similar_indices)
        
        num_duplicates = sum(len(group) - 1 for group in duplicate_groups)
        self.stats['semantic_duplicates'] = num_duplicates
        
        logger.info(f"✓ Found {len(duplicate_groups)} semantic duplicate groups "
                   f"({num_duplicates} duplicate files)")
        
        return duplicate_groups
    
    # ===================== DUPLICATE RESOLUTION =====================
    
    def load_quality_scores(self) -> Dict[str, float]:
        """
        Load quality scores from previous assessment step.
        
        Returns:
            Dictionary of {filename: quality_score}
        """
        quality_scores = {}
        
        # Try to load from quality assessment results
        results_file = Path('logs/quality_assessment_results.json')
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
                
            for result in results:
                if result['status'] == 'success':
                    # Extract base filename (remove _assessed suffix)
                    filename = result['source_file'].replace('_cleaned', '').replace('.txt', '_assessed.txt')
                    score = result['composite_score']['composite_quality_score']
                    quality_scores[filename] = score
        
        return quality_scores
    
    def select_best_version(self, duplicate_group: List[str], 
                           quality_scores: Dict[str, float]) -> str:
        """
        Select the best version from a duplicate group.
        
        Criteria (in order):
        1. Highest quality score
        2. Longest content (if no quality scores)
        3. Alphabetically first (tie-breaker)
        
        Args:
            duplicate_group: List of duplicate filenames
            quality_scores: Dictionary of quality scores
            
        Returns:
            Filename of best version
        """
        if len(duplicate_group) == 1:
            return duplicate_group[0]
        
        # Check if we have quality scores
        scored_files = [(f, quality_scores.get(f, 0)) for f in duplicate_group]
        
        # Sort by score (descending), then by filename (ascending)
        scored_files.sort(key=lambda x: (-x[1], x[0]))
        
        return scored_files[0][0]
    
    def merge_duplicate_groups(self, exact_groups: Dict[str, List[str]],
                               fuzzy_groups: List[List[str]],
                               semantic_groups: List[List[str]]) -> List[Tuple[str, List[str]]]:
        """
        Merge all duplicate groups and resolve to keep best version.
        
        Returns:
            List of tuples: (file_to_keep, files_to_remove)
        """
        logger.info("Merging duplicate groups and selecting best versions...")
        
        # Load quality scores
        quality_scores = self.load_quality_scores()
        
        # Combine all duplicate groups
        all_groups = []
        
        # Add exact duplicate groups
        for files in exact_groups.values():
            all_groups.append(files)
        
        # Add fuzzy groups
        all_groups.extend(fuzzy_groups)
        
        # Add semantic groups
        all_groups.extend(semantic_groups)
        
        # Resolve overlapping groups (a file might be in multiple groups)
        # Use union-find to merge overlapping groups
        file_to_group = {}
        for group_id, group in enumerate(all_groups):
            for filename in group:
                if filename not in file_to_group:
                    file_to_group[filename] = group_id
                else:
                    # Merge groups
                    old_group_id = file_to_group[filename]
                    for f in all_groups[old_group_id]:
                        file_to_group[f] = group_id
                    all_groups[group_id].extend(all_groups[old_group_id])
                    all_groups[old_group_id] = []
        
        # Remove empty groups and deduplicate group members
        merged_groups = []
        for group in all_groups:
            if group:
                unique_group = list(set(group))
                if len(unique_group) > 1:
                    merged_groups.append(unique_group)
        
        # Select best version from each group
        resolution = []
        for group in merged_groups:
            best_file = self.select_best_version(group, quality_scores)
            files_to_remove = [f for f in group if f != best_file]
            resolution.append((best_file, files_to_remove))
        
        logger.info(f"✓ Resolved {len(merged_groups)} duplicate groups")
        
        return resolution
    
    # ===================== FILE PROCESSING =====================
    
    def load_files(self) -> Dict[str, str]:
        """
        Load all text files from input directory.
        
        Returns:
            Dictionary of {filename: text_content}
        """
        logger.info(f"Loading files from {self.input_dir}")
        
        files_data = {}
        file_paths = list(self.input_dir.glob('*.txt'))
        
        for filepath in tqdm(file_paths, desc="Loading files"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                files_data[filepath.name] = text
            except Exception as e:
                logger.error(f"Error loading {filepath.name}: {str(e)}")
        
        self.stats['total_files'] = len(files_data)
        logger.info(f"✓ Loaded {len(files_data)} files")
        
        return files_data
    
    def save_deduplicated_files(self, files_data: Dict[str, str],
                               files_to_keep: Set[str]):
        """
        Save deduplicated files to output directory.
        
        Args:
            files_data: Original files data
            files_to_keep: Set of filenames to keep
        """
        logger.info("Saving deduplicated files...")
        
        for filename in tqdm(files_to_keep, desc="Saving files"):
            if filename in files_data:
                output_path = self.output_dir / filename.replace('_assessed', '_deduplicated')
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(files_data[filename])
        
        logger.info(f"✓ Saved {len(files_to_keep)} unique files to {self.output_dir}")
    
    def generate_deduplication_report(self, resolution: List[Tuple[str, List[str]]]):
        """
        Generate detailed deduplication report.
        
        Args:
            resolution: List of (kept_file, removed_files) tuples
        """
        logger.info("Generating deduplication report...")
        
        report_data = []
        
        for kept_file, removed_files in resolution:
            for removed_file in removed_files:
                report_data.append({
                    'removed_file': removed_file,
                    'kept_file': kept_file,
                    'reason': 'duplicate'
                })
        
        # Save as CSV
        if report_data:
            df = pd.DataFrame(report_data)
            report_path = 'logs/deduplication_report.csv'
            df.to_csv(report_path, index=False)
            logger.info(f"✓ Deduplication report saved to: {report_path}")
        
        # Save detailed duplicate groups
        duplicate_groups_data = {
            'duplicate_groups': [
                {
                    'kept': kept,
                    'removed': removed
                }
                for kept, removed in resolution
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        with open('logs/duplicate_groups.json', 'w', encoding='utf-8') as f:
            json.dump(duplicate_groups_data, f, indent=2)
    
    def run_deduplication(self) -> Dict:
        """
        Run the complete deduplication pipeline.
        
        Returns:
            Dictionary with deduplication results and statistics
        """
        logger.info("\n" + "="*60)
        logger.info("STARTING DEDUPLICATION PIPELINE")
        logger.info("="*60)
        
        # Load all files
        files_data = self.load_files()
        
        if len(files_data) < 2:
            logger.warning("Not enough files for deduplication")
            return {'status': 'skipped', 'reason': 'insufficient_files'}
        
        # Track files to exclude from subsequent checks
        excluded_files = set()
        
        # Step 1: Exact duplicate detection
        exact_groups = {}
        if self.exact_match:
            exact_groups = self.find_exact_duplicates(files_data)
            # Add all but one from each group to excluded
            for files in exact_groups.values():
                excluded_files.update(files[1:])  # Exclude all but first
        
        # Step 2: Fuzzy duplicate detection
        fuzzy_groups = []
        if self.fuzzy_match:
            fuzzy_groups = self.find_fuzzy_duplicates(files_data, excluded_files)
            # Add found duplicates to excluded
            for group in fuzzy_groups:
                excluded_files.update(group[1:])
        
        # Step 3: Semantic duplicate detection
        semantic_groups = []
        if self.semantic_match:
            semantic_groups = self.find_semantic_duplicates(files_data, excluded_files)
        
        # Step 4: Merge and resolve duplicates
        resolution = self.merge_duplicate_groups(exact_groups, fuzzy_groups, semantic_groups)
        
        # Calculate final statistics
        files_to_keep = set(files_data.keys())
        for _, removed_files in resolution:
            files_to_keep -= set(removed_files)
        
        self.stats['unique_files'] = len(files_to_keep)
        self.stats['total_duplicates_removed'] = self.stats['total_files'] - self.stats['unique_files']
        
        # Step 5: Save deduplicated files
        self.save_deduplicated_files(files_data, files_to_keep)
        
        # Step 6: Generate reports
        self.generate_deduplication_report(resolution)
        
        # Print summary
        self.print_summary()
        
        return {
            'status': 'success',
            'statistics': self.stats,
            'duplicate_groups': len(resolution),
            'resolution': resolution
        }
    
    def print_summary(self):
        """Print deduplication statistics."""
        logger.info("\n" + "="*60)
        logger.info("DEDUPLICATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total files processed: {self.stats['total_files']}")
        logger.info(f"Unique files kept: {self.stats['unique_files']}")
        logger.info(f"Total duplicates removed: {self.stats['total_duplicates_removed']}")
        logger.info(f"  - Exact duplicates: {self.stats['exact_duplicates']}")
        logger.info(f"  - Fuzzy duplicates: {self.stats['fuzzy_duplicates']}")
        logger.info(f"  - Semantic duplicates: {self.stats['semantic_duplicates']}")
        
        if self.stats['total_files'] > 0:
            dedup_rate = (self.stats['total_duplicates_removed'] / self.stats['total_files']) * 100
            logger.info(f"Deduplication rate: {dedup_rate:.1f}%")
        
        logger.info("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Initialize deduplicator
    deduplicator = SemanticDeduplicator(
        input_dir="data/quality_assessed/high_quality",
        output_dir="data/deduplicated",
        exact_match=True,           # Enable exact matching
        fuzzy_match=True,           # Enable fuzzy matching
        semantic_match=True,        # Enable semantic matching
        semantic_threshold=0.95,    # 95% similarity for semantic duplicates
        fuzzy_threshold=0.85        # 85% similarity for fuzzy duplicates
    )
    
    # Run deduplication
    results = deduplicator.run_deduplication()
    
    # Save results
    with open('logs/deduplication_results.json', 'w') as f:
        # Convert sets to lists for JSON serialization
        serializable_results = {
            'status': results['status'],
            'statistics': results['statistics'],
            'duplicate_groups': results['duplicate_groups']
        }
        json.dump(serializable_results, f, indent=2)
    
    print("\nDeduplication complete!")
    print(f"Deduplicated files: data/deduplicated/")
    print(f"Deduplication report: logs/deduplication_report.csv")
    print(f"Duplicate groups: logs/duplicate_groups.json")