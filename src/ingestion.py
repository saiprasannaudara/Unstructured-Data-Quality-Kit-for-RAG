import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hashlib
from datetime import datetime

# Document processing libraries
import PyPDF2
from docx import Document
import chardet

import sys

# Reconfigure console streams to UTF-8 (Python 3.7+)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ingestion.log', encoding="utf-8", errors="replace"),
        logging.StreamHandler(sys.stdout)
    ],
    force=True,
)
logger = logging.getLogger(__name__)


class DocumentIngestionValidator:
    """
    Handles document ingestion and validation for RAG systems.
    Supports multiple file formats and performs quality checks.
    """
    
    def __init__(self, raw_data_dir: str = "data/raw", 
                 processed_dir: str = "data/processed",
                 failed_dir: str = "data/failed"):
        """
        Initialize the ingestion system.
        
        Args:
            raw_data_dir: Directory containing raw documents
            processed_dir: Directory to save extracted text
            failed_dir: Directory to move failed documents
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        self.failed_dir = Path(failed_dir)
        
        # Create directories if they don't exist
        for directory in [self.raw_data_dir, self.processed_dir, self.failed_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Supported file formats
        self.supported_formats = {'.txt', '.pdf', '.docx', '.md', '.html'}
        
        # Statistics tracking
        self.stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'empty_files': 0,
            'corrupted': 0,
            'unsupported_format': 0
        }
    
    def get_file_hash(self, filepath: Path) -> str:
        """Generate MD5 hash for file deduplication."""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def detect_encoding(self, filepath: Path) -> str:
        """Detect file encoding using chardet."""
        with open(filepath, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8'
    
    def extract_text_from_txt(self, filepath: Path) -> Tuple[str, Dict]:
        """
        Extract text from TXT files with encoding detection.
        
        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            # Detect encoding
            encoding = self.detect_encoding(filepath)
            logger.info(f"Detected encoding for {filepath.name}: {encoding}")
            
            # Read file
            with open(filepath, 'r', encoding=encoding, errors='ignore') as f:
                text = f.read()
            
            metadata = {
                'filename': filepath.name,
                'format': 'txt',
                'encoding': encoding,
                'size_bytes': filepath.stat().st_size,
                'char_count': len(text),
                'file_hash': self.get_file_hash(filepath)
            }
            
            return text, metadata
            
        except Exception as e:
            logger.error(f"Error extracting text from {filepath.name}: {str(e)}")
            raise
    
    def extract_text_from_pdf(self, filepath: Path) -> Tuple[str, Dict]:
        """
        Extract text from PDF files.
        
        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            text = ""
            page_count = 0
            
            with open(filepath, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                page_count = len(pdf_reader.pages)
                
                # Extract text from each page
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        text += page_text + "\n\n"
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num} from {filepath.name}: {str(e)}")
            
            metadata = {
                'filename': filepath.name,
                'format': 'pdf',
                'page_count': page_count,
                'size_bytes': filepath.stat().st_size,
                'char_count': len(text),
                'file_hash': self.get_file_hash(filepath)
            }
            
            return text, metadata
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {filepath.name}: {str(e)}")
            raise
    
    def extract_text_from_docx(self, filepath: Path) -> Tuple[str, Dict]:
        """
        Extract text from DOCX files.
        
        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            doc = Document(filepath)
            
            # Extract text from paragraphs
            paragraphs = [paragraph.text for paragraph in doc.paragraphs]
            text = "\n\n".join(paragraphs)
            
            metadata = {
                'filename': filepath.name,
                'format': 'docx',
                'paragraph_count': len(paragraphs),
                'size_bytes': filepath.stat().st_size,
                'char_count': len(text),
                'file_hash': self.get_file_hash(filepath)
            }
            
            return text, metadata
            
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {filepath.name}: {str(e)}")
            raise
    
    def validate_extracted_text(self, text: str, metadata: Dict) -> Tuple[bool, List[str]]:
        """
        Validate the extracted text for quality issues.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check 1: Empty or nearly empty
        if len(text.strip()) == 0:
            issues.append("Document is empty")
            return False, issues
        
        if len(text.strip()) < 50:
            issues.append(f"Document too short (only {len(text.strip())} characters)")
        
        # Check 2: Too many special characters (might indicate corruption)
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_char_ratio > 0.3:
            issues.append(f"High special character ratio ({special_char_ratio:.2%})")
        
        # Check 3: Check for garbled text (excessive non-ASCII)
        non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / len(text)
        if non_ascii_ratio > 0.5:
            issues.append(f"High non-ASCII character ratio ({non_ascii_ratio:.2%})")
        
        # Check 4: Repetitive content (might indicate OCR errors)
        words = text.split()
        if len(words) > 10:
            unique_words = len(set(words))
            repetition_ratio = unique_words / len(words)
            if repetition_ratio < 0.3:
                issues.append(f"Low word diversity ({repetition_ratio:.2%})")
        
        # Determine if valid (no critical issues)
        is_valid = len([i for i in issues if "empty" in i.lower()]) == 0
        
        return is_valid, issues
    
    def process_single_file(self, filepath: Path) -> Optional[Dict]:
        """
        Process a single file: extract text and validate.
        
        Returns:
            Dictionary with processing results or None if failed
        """
        logger.info(f"Processing file: {filepath.name}")
        
        # Check file extension
        file_ext = filepath.suffix.lower()
        if file_ext not in self.supported_formats:
            logger.warning(f"Unsupported format: {file_ext}")
            self.stats['unsupported_format'] += 1
            return None
        
        try:
            # Extract text based on format
            if file_ext == '.txt' or file_ext == '.md':
                text, metadata = self.extract_text_from_txt(filepath)
            elif file_ext == '.pdf':
                text, metadata = self.extract_text_from_pdf(filepath)
            elif file_ext == '.docx':
                text, metadata = self.extract_text_from_docx(filepath)
            else:
                logger.warning(f"Handler not implemented for {file_ext}")
                return None
            
            # Validate extracted text
            is_valid, issues = self.validate_extracted_text(text, metadata)
            
            metadata['validation_status'] = 'valid' if is_valid else 'invalid'
            metadata['validation_issues'] = issues
            metadata['processed_timestamp'] = datetime.now().isoformat()
            
            if is_valid:
                # Save processed text
                output_filename = filepath.stem + '_processed.txt'
                output_path = self.processed_dir / output_filename
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                metadata['processed_file_path'] = str(output_path)
                self.stats['successful'] += 1
                logger.info(f"✓ Successfully processed: {filepath.name}")
                
            else:
                # Move to failed directory
                failed_path = self.failed_dir / filepath.name
                filepath.rename(failed_path)
                metadata['failed_reason'] = '; '.join(issues)
                
                if "empty" in str(issues).lower():
                    self.stats['empty_files'] += 1
                else:
                    self.stats['corrupted'] += 1
                
                self.stats['failed'] += 1
                logger.warning(f"✗ Validation failed for {filepath.name}: {issues}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"✗ Failed to process {filepath.name}: {str(e)}")
            self.stats['failed'] += 1
            return {
                'filename': filepath.name,
                'validation_status': 'error',
                'error_message': str(e),
                'processed_timestamp': datetime.now().isoformat()
            }
    
    def process_all_files(self) -> List[Dict]:
        """
        Process all files in the raw data directory.
        
        Returns:
            List of metadata dictionaries for all processed files
        """
        logger.info(f"Starting batch processing from {self.raw_data_dir}")
        
        # Get all files
        all_files = [f for f in self.raw_data_dir.iterdir() if f.is_file()]
        self.stats['total_files'] = len(all_files)
        
        logger.info(f"Found {len(all_files)} files to process")
        
        # Process each file
        results = []
        for filepath in all_files:
            result = self.process_single_file(filepath)
            if result:
                results.append(result)
        
        # Print summary
        self.print_summary()
        
        return results
    
    def print_summary(self):
        """Print processing statistics."""
        logger.info("\n" + "="*60)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total files: {self.stats['total_files']}")
        logger.info(f"Successfully processed: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"  - Empty files: {self.stats['empty_files']}")
        logger.info(f"  - Corrupted: {self.stats['corrupted']}")
        logger.info(f"  - Unsupported format: {self.stats['unsupported_format']}")
        
        if self.stats['total_files'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_files']) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Initialize the ingestion system
    ingestion_system = DocumentIngestionValidator(
        raw_data_dir="data/raw",
        processed_dir="data/processed",
        failed_dir="data/failed"
    )
    
    # Process all documents
    results = ingestion_system.process_all_files()
    
    # Optionally, save results to a CSV for further analysis
    import json
    with open('logs/processing_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print("\nProcessing complete! Check logs/ingestion.log for details.")