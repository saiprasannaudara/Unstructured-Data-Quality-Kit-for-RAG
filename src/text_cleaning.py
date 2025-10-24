import re
import unicodedata
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Text processing libraries
import ftfy  # Fix text encoding
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException

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
        logging.FileHandler('logs/text_cleaning.log', encoding="utf-8", errors="replace"),
        logging.StreamHandler(sys.stdout)
    ],
    force=True,
)
logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Advanced text cleaning and normalization for RAG systems.
    Handles encoding issues, noise removal, and standardization.
    """
    
    def __init__(self, processed_dir: str = "data/processed",
                 cleaned_dir: str = "data/cleaned"):
        """
        Initialize the text cleaning system.
        
        Args:
            processed_dir: Directory with extracted text files
            cleaned_dir: Directory to save cleaned text
        """
        self.processed_dir = Path(processed_dir)
        self.cleaned_dir = Path(cleaned_dir)
        
        # Create output directory
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'total_files': 0,
            'cleaned': 0,
            'failed': 0,
            'total_chars_removed': 0,
            'avg_cleanup_ratio': []
        }
        
        # Common patterns to clean
        self.setup_cleaning_patterns()
    
    def setup_cleaning_patterns(self):
        """Define regex patterns for common cleaning tasks."""
        
        # Page numbers and headers/footers patterns
        self.page_number_pattern = re.compile(
            r'^\s*(?:page\s*)?\d+\s*(?:of\s*\d+)?\s*$',
            re.IGNORECASE | re.MULTILINE
        )
        
        # Multiple whitespace
        self.multiple_spaces = re.compile(r' {2,}')
        self.multiple_newlines = re.compile(r'\n{3,}')
        
        # Special characters that are often artifacts
        self.weird_chars = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
        
        # Common PDF/OCR artifacts
        self.ocr_artifacts = [
            re.compile(r'\f'),  # Form feed
            re.compile(r'\\[nrt]'),  # Literal escape sequences
            re.compile(r'[▪•◦▫](?=\s)'),  # Bullet points (we'll normalize these)
        ]
        
        # URL pattern (we might want to keep or remove these)
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # Email pattern
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # Repeated punctuation
        self.repeated_punct = re.compile(r'([.!?]){2,}')
        
        # HTML entities that might have slipped through
        self.html_entity_pattern = re.compile(r'&[a-z]+;|&#\d+;')
    
    def fix_encoding_issues(self, text: str) -> str:
        """
        Fix common encoding problems using ftfy.
        
        Examples:
            â€™ → '
            â€œ → "
            Ã© → é
        """
        try:
            # ftfy fixes most common encoding issues
            text = ftfy.fix_text(text)
            return text
        except Exception as e:
            logger.warning(f"Error fixing encoding: {str(e)}")
            return text
    
    def normalize_unicode(self, text: str) -> str:
        """
        Normalize unicode characters to standard forms.
        Uses NFKC normalization (compatibility decomposition followed by composition).
        """
        try:
            # NFKC: Compatibility decomposition + canonical composition
            # Converts similar-looking characters to standard forms
            text = unicodedata.normalize('NFKC', text)
            return text
        except Exception as e:
            logger.warning(f"Error normalizing unicode: {str(e)}")
            return text
    
    def remove_html_tags(self, text: str) -> str:
        """Remove any HTML/XML tags that might be present."""
        try:
            soup = BeautifulSoup(text, 'lxml')
            text = soup.get_text()
            return text
        except Exception as e:
            logger.warning(f"Error removing HTML: {str(e)}")
            return text
    
    def clean_whitespace(self, text: str) -> str:
        """
        Clean excessive whitespace while preserving paragraph structure.
        """
        # Remove weird invisible characters
        text = self.weird_chars.sub('', text)
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove trailing/leading spaces from each line
        lines = [line.strip() for line in text.split('\n')]
        
        # Remove completely empty lines but keep single newlines for paragraphs
        text = '\n'.join(lines)
        
        # Reduce multiple spaces to single space
        text = self.multiple_spaces.sub(' ', text)
        
        # Reduce multiple newlines to maximum 2 (paragraph breaks)
        text = self.multiple_newlines.sub('\n\n', text)
        
        return text.strip()
    
    def remove_page_artifacts(self, text: str) -> str:
        """
        Remove page numbers, headers, and footers.
        """
        # Remove page numbers (lines that are just numbers or "Page X")
        text = self.page_number_pattern.sub('', text)
        
        # Remove form feeds
        text = text.replace('\f', '\n')
        
        return text
    
    def normalize_bullets_and_lists(self, text: str) -> str:
        """
        Normalize different bullet point styles to a consistent format.
        """
        # Common bullet characters: •, ◦, ▪, ▫, *, -, ⁃
        bullet_chars = r'[•◦▪▫⁃]'
        
        # Replace various bullet styles with a standard dash
        text = re.sub(f'^\\s*{bullet_chars}\\s+', '- ', text, flags=re.MULTILINE)
        
        # Normalize numbered lists (ensure format: "1. item")
        text = re.sub(r'^\\s*(\\d+)[.):]\\s+', r'\\1. ', text, flags=re.MULTILINE)
        
        return text
    
    def handle_hyphenation(self, text: str) -> str:
        """
        Fix words broken across lines with hyphens.
        Example: "exam-\nple" → "example"
        """
        # Pattern: word ending with hyphen followed by newline and word
        text = re.sub(r'(\\w+)-\\s*\\n\\s*(\\w+)', r'\\1\\2', text)
        return text
    
    def clean_punctuation(self, text: str) -> str:
        """
        Clean up punctuation issues.
        """
        # Fix repeated punctuation (e.g., "!!!" → "!")
        text = self.repeated_punct.sub(r'\\1', text)
        
        # Fix spaces before punctuation
        text = re.sub(r'\\s+([.,!?;:])', r'\\1', text)
        
        # Ensure space after punctuation (if followed by letter)
        text = re.sub(r'([.,!?;:])([A-Za-z])', r'\\1 \\2', text)
        
        return text
    
    def remove_urls_and_emails(self, text: str, keep_urls: bool = False, 
                               keep_emails: bool = False) -> Tuple[str, List[str], List[str]]:
        """
        Optionally remove or extract URLs and emails.
        
        Returns:
            Tuple of (cleaned_text, extracted_urls, extracted_emails)
        """
        extracted_urls = []
        extracted_emails = []
        
        if not keep_urls:
            # Extract URLs before removing
            extracted_urls = self.url_pattern.findall(text)
            # Replace URLs with placeholder or remove
            text = self.url_pattern.sub('[URL]', text)
        
        if not keep_emails:
            # Extract emails before removing
            extracted_emails = self.email_pattern.findall(text)
            # Replace emails with placeholder or remove
            text = self.email_pattern.sub('[EMAIL]', text)
        
        return text, extracted_urls, extracted_emails
    
    def remove_duplicate_lines(self, text: str) -> str:
        """
        Remove consecutive duplicate lines (often from headers/footers).
        """
        lines = text.split('\n')
        cleaned_lines = []
        previous_line = None
        
        for line in lines:
            # Keep line if it's different from previous or if it's empty (paragraph break)
            if line != previous_line or line.strip() == '':
                cleaned_lines.append(line)
            previous_line = line
        
        return '\n'.join(cleaned_lines)
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the text.
        """
        try:
            # Take a sample if text is very long
            sample = text[:1000] if len(text) > 1000 else text
            language = detect(sample)
            return language
        except LangDetectException:
            logger.warning("Could not detect language")
            return "unknown"
    
    def calculate_cleaning_metrics(self, original: str, cleaned: str) -> Dict:
        """
        Calculate metrics about the cleaning process.
        """
        original_len = len(original)
        cleaned_len = len(cleaned)
        chars_removed = original_len - cleaned_len
        cleanup_ratio = (chars_removed / original_len * 100) if original_len > 0 else 0
        
        # Count specific changes
        original_lines = original.count('\n')
        cleaned_lines = cleaned.count('\n')
        
        return {
            'original_length': original_len,
            'cleaned_length': cleaned_len,
            'chars_removed': chars_removed,
            'cleanup_ratio_percent': round(cleanup_ratio, 2),
            'original_lines': original_lines,
            'cleaned_lines': cleaned_lines,
            'compression_ratio': round(cleaned_len / original_len, 3) if original_len > 0 else 0
        }
    
    def clean_text(self, text: str, keep_urls: bool = False, 
                   keep_emails: bool = False) -> Tuple[str, Dict]:
        """
        Main cleaning pipeline - applies all cleaning steps.
        
        Args:
            text: Raw text to clean
            keep_urls: Whether to keep URLs in text
            keep_emails: Whether to keep emails in text
            
        Returns:
            Tuple of (cleaned_text, metadata)
        """
        original_text = text
        metadata = {}
        
        logger.info("Starting text cleaning pipeline...")
        
        # Step 1: Fix encoding issues
        text = self.fix_encoding_issues(text)
        logger.debug("✓ Fixed encoding issues")
        
        # Step 2: Remove HTML tags
        text = self.remove_html_tags(text)
        logger.debug("✓ Removed HTML tags")
        
        # Step 3: Normalize unicode
        text = self.normalize_unicode(text)
        logger.debug("✓ Normalized unicode")
        
        # Step 4: Handle URLs and emails
        text, urls, emails = self.remove_urls_and_emails(text, keep_urls, keep_emails)
        metadata['extracted_urls'] = urls
        metadata['extracted_emails'] = emails
        logger.debug(f"✓ Processed URLs ({len(urls)}) and emails ({len(emails)})")
        
        # Step 5: Remove page artifacts
        text = self.remove_page_artifacts(text)
        logger.debug("✓ Removed page artifacts")
        
        # Step 6: Fix hyphenation
        text = self.handle_hyphenation(text)
        logger.debug("✓ Fixed hyphenation")
        
        # Step 7: Normalize bullets and lists
        text = self.normalize_bullets_and_lists(text)
        logger.debug("✓ Normalized bullets and lists")
        
        # Step 8: Clean punctuation
        text = self.clean_punctuation(text)
        logger.debug("✓ Cleaned punctuation")
        
        # Step 9: Remove duplicate lines
        text = self.remove_duplicate_lines(text)
        logger.debug("✓ Removed duplicate lines")
        
        # Step 10: Clean whitespace (should be last)
        text = self.clean_whitespace(text)
        logger.debug("✓ Cleaned whitespace")
        
        # Detect language
        metadata['detected_language'] = self.detect_language(text)
        
        # Calculate metrics
        metrics = self.calculate_cleaning_metrics(original_text, text)
        metadata.update(metrics)
        
        logger.info(f"Cleaning complete. Removed {metrics['chars_removed']} characters "
                   f"({metrics['cleanup_ratio_percent']}%)")
        
        return text, metadata
    
    def process_file(self, filepath: Path, keep_urls: bool = False, 
                     keep_emails: bool = False) -> Dict:
        """
        Process a single text file through the cleaning pipeline.
        
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing file: {filepath.name}")
        
        try:
            # Read the file
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Clean the text
            cleaned_text, metadata = self.clean_text(text, keep_urls, keep_emails)
            
            # Save cleaned text
            output_filename = filepath.stem.replace('_processed', '') + '_cleaned.txt'
            output_path = self.cleaned_dir / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            # Update metadata
            metadata['source_file'] = filepath.name
            metadata['output_file'] = output_filename
            metadata['output_path'] = str(output_path)
            metadata['processed_timestamp'] = datetime.now().isoformat()
            metadata['status'] = 'success'
            
            # Update stats
            self.stats['cleaned'] += 1
            self.stats['total_chars_removed'] += metadata['chars_removed']
            self.stats['avg_cleanup_ratio'].append(metadata['cleanup_ratio_percent'])
            
            logger.info(f"✓ Successfully cleaned: {filepath.name}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"✗ Failed to clean {filepath.name}: {str(e)}")
            self.stats['failed'] += 1
            return {
                'source_file': filepath.name,
                'status': 'failed',
                'error_message': str(e),
                'processed_timestamp': datetime.now().isoformat()
            }
    
    def process_all_files(self, keep_urls: bool = False, 
                         keep_emails: bool = False) -> List[Dict]:
        """
        Process all files in the processed directory.
        
        Returns:
            List of metadata dictionaries for all cleaned files
        """
        logger.info(f"Starting batch cleaning from {self.processed_dir}")
        
        # Get all text files
        all_files = list(self.processed_dir.glob('*.txt'))
        self.stats['total_files'] = len(all_files)
        
        logger.info(f"Found {len(all_files)} files to clean")
        
        # Process each file
        results = []
        for filepath in all_files:
            result = self.process_file(filepath, keep_urls, keep_emails)
            results.append(result)
        
        # Print summary
        self.print_summary()
        
        return results
    
    def print_summary(self):
        """Print cleaning statistics."""
        logger.info("\n" + "="*60)
        logger.info("CLEANING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total files: {self.stats['total_files']}")
        logger.info(f"Successfully cleaned: {self.stats['cleaned']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Total characters removed: {self.stats['total_chars_removed']:,}")
        
        if self.stats['avg_cleanup_ratio']:
            avg_ratio = sum(self.stats['avg_cleanup_ratio']) / len(self.stats['avg_cleanup_ratio'])
            logger.info(f"Average cleanup ratio: {avg_ratio:.2f}%")
        
        if self.stats['total_files'] > 0:
            success_rate = (self.stats['cleaned'] / self.stats['total_files']) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Initialize the text cleaner
    cleaner = TextCleaner(
        processed_dir="data/processed",
        cleaned_dir="data/cleaned"
    )
    
    # Process all files
    # Set keep_urls=True if you want to preserve URLs
    # Set keep_emails=True if you want to preserve email addresses
    results = cleaner.process_all_files(
        keep_urls=False,  # Change to True to keep URLs
        keep_emails=False  # Change to True to keep emails
    )
    
    # Save results to JSON
    with open('logs/cleaning_results.json', 'w', encoding = 'utf-8') as f:
        json.dump(results, f, indent=2)
    
    print("\nCleaning complete! Check logs/text_cleaning.log for details.")
    print(f"Cleaned files are in: data/cleaned/")
    
    # Print a sample comparison
    if results and results[0]['status'] == 'success':
        print("\n" + "="*60)
        print("SAMPLE OUTPUT (First 500 characters)")
        print("="*60)
        sample_file = Path(results[0]['output_path'])
        if sample_file.exists():
            with open(sample_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(content[:500])
                if len(content) > 500:
                    print("\n... (truncated)")
        print("="*60)