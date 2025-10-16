import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import re
from collections import Counter

# NLP and extraction
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
import spacy
from tqdm import tqdm
import numpy as np

# Keyword extraction
from keybert import KeyBERT
import yake

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/metadata_enrichment.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set console output encoding for Windows compatibility
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


class MetadataEnricher:
    """
    Comprehensive metadata enrichment system for RAG chunks.
    Adds keywords, entities, topics, summaries, and more.
    """
    
    def __init__(self,
                 input_dir: str = "data/chunked",
                 output_dir: str = "data/enriched",
                 enable_keywords: bool = True,
                 enable_entities: bool = True,
                 enable_summaries: bool = True,
                 enable_questions: bool = True,
                 enable_topics: bool = True,
                 max_keywords: int = 5):
        """
        Initialize the metadata enricher.
        
        Args:
            input_dir: Directory with chunked documents
            output_dir: Directory to save enriched chunks
            enable_keywords: Extract keywords
            enable_entities: Extract named entities
            enable_summaries: Generate summaries
            enable_questions: Generate potential questions
            enable_topics: Detect topics
            max_keywords: Maximum number of keywords per chunk
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature flags
        self.enable_keywords = enable_keywords
        self.enable_entities = enable_entities
        self.enable_summaries = enable_summaries
        self.enable_questions = enable_questions
        self.enable_topics = enable_topics
        self.max_keywords = max_keywords
        
        # Initialize NLP models (lazy loading)
        self.nlp = None
        self.keyword_model = None
        self.yake_extractor = None
        self.stop_words = None
        
        # Statistics tracking
        self.stats = {
            'total_chunks': 0,
            'enriched_chunks': 0,
            'failed_chunks': 0,
            'avg_keywords_per_chunk': [],
            'avg_entities_per_chunk': [],
            'chunks_with_entities': 0,
            'chunks_with_keywords': 0
        }
    
    def _load_models(self):
        """Lazy load NLP models."""
        
        # Load spaCy for NER
        if self.enable_entities and self.nlp is None:
            try:
                import spacy
                logger.info("Loading spaCy model for NER...")
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("[SUCCESS] spaCy model loaded")
            except Exception as e:
                logger.warning(f"Could not load spaCy: {str(e)}")
                self.enable_entities = False
        
        # Load KeyBERT for keyword extraction
        if self.enable_keywords and self.keyword_model is None:
            try:
                logger.info("Loading KeyBERT model...")
                self.keyword_model = KeyBERT()
                logger.info("[SUCCESS] KeyBERT model loaded")
            except Exception as e:
                logger.warning(f"Could not load KeyBERT: {str(e)}")
                self.keyword_model = None
        
        # Initialize YAKE as fallback
        if self.enable_keywords and self.yake_extractor is None:
            try:
                self.yake_extractor = yake.KeywordExtractor(
                    lan="en",
                    n=3,  # Max n-gram size
                    dedupLim=0.9,
                    top=self.max_keywords
                )
            except Exception as e:
                logger.warning(f"Could not initialize YAKE: {str(e)}")
        
        # Load stopwords
        if self.stop_words is None:
            try:
                self.stop_words = set(stopwords.words('english'))
            except LookupError:
                logger.info("Downloading NLTK stopwords...")
                nltk.download('stopwords')
                self.stop_words = set(stopwords.words('english'))
    
    # ===================== KEYWORD EXTRACTION =====================
    
    def extract_keywords_keybert(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract keywords using KeyBERT (transformer-based).
        
        Returns:
            List of (keyword, score) tuples
        """
        try:
            if self.keyword_model is None:
                return []
            
            keywords = self.keyword_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_n=self.max_keywords,
                use_mmr=True,  # Maximize diversity
                diversity=0.5
            )
            return keywords
            
        except Exception as e:
            logger.warning(f"KeyBERT extraction failed: {str(e)}")
            return []
    
    def extract_keywords_yake(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract keywords using YAKE (statistical method).
        
        Returns:
            List of (keyword, score) tuples
        """
        try:
            if self.yake_extractor is None:
                return []
            
            keywords = self.yake_extractor.extract_keywords(text)
            # YAKE scores are lower=better, so invert
            keywords = [(kw, 1.0 - score) for kw, score in keywords]
            return keywords
            
        except Exception as e:
            logger.warning(f"YAKE extraction failed: {str(e)}")
            return []
    
    def extract_keywords_tfidf(self, text: str) -> List[str]:
        """
        Simple TF-IDF based keyword extraction (fallback).
        
        Returns:
            List of keywords
        """
        try:
            # Tokenize and clean
            words = word_tokenize(text.lower())
            words = [w for w in words if w.isalpha() and w not in self.stop_words]
            
            # Count frequencies
            word_freq = Counter(words)
            
            # Get top N
            top_words = [word for word, _ in word_freq.most_common(self.max_keywords)]
            return top_words
            
        except Exception as e:
            logger.warning(f"TF-IDF extraction failed: {str(e)}")
            return []
    
    def extract_keywords(self, text: str) -> Dict:
        """
        Extract keywords using multiple methods and combine.
        
        Returns:
            Dictionary with keywords and scores
        """
        all_keywords = {}
        
        # Try KeyBERT first (best quality)
        keybert_keywords = self.extract_keywords_keybert(text)
        if keybert_keywords:
            for kw, score in keybert_keywords:
                all_keywords[kw] = {'score': float(score), 'method': 'keybert'}
        
        # Try YAKE as alternative
        elif self.yake_extractor:
            yake_keywords = self.extract_keywords_yake(text)
            for kw, score in yake_keywords:
                all_keywords[kw] = {'score': float(score), 'method': 'yake'}
        
        # Fallback to simple TF-IDF
        else:
            tfidf_keywords = self.extract_keywords_tfidf(text)
            for kw in tfidf_keywords:
                all_keywords[kw] = {'score': 0.5, 'method': 'tfidf'}
        
        return {
            'keywords': list(all_keywords.keys()),
            'keyword_details': all_keywords,
            'keyword_count': len(all_keywords)
        }
    
    # ===================== NAMED ENTITY RECOGNITION =====================
    
    def extract_entities(self, text: str) -> Dict:
        """
        Extract named entities using spaCy.
        
        Returns:
            Dictionary with entities by type
        """
        if not self.enable_entities or self.nlp is None:
            return {'entities': [], 'entity_types': {}}
        
        try:
            doc = self.nlp(text[:1000000])  # Limit text length for performance
            
            entities = []
            entity_types = {}
            
            for ent in doc.ents:
                entity_info = {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                }
                entities.append(entity_info)
                
                # Group by type
                if ent.label_ not in entity_types:
                    entity_types[ent.label_] = []
                entity_types[ent.label_].append(ent.text)
            
            # Deduplicate entities by type
            for label in entity_types:
                entity_types[label] = list(set(entity_types[label]))
            
            return {
                'entities': entities,
                'entity_types': entity_types,
                'entity_count': len(entities),
                'unique_entities': len(set(e['text'] for e in entities))
            }
            
        except Exception as e:
            logger.warning(f"Entity extraction failed: {str(e)}")
            return {'entities': [], 'entity_types': {}, 'entity_count': 0}
    
    # ===================== SUMMARY GENERATION =====================
    
    def generate_summary(self, text: str, max_sentences: int = 2) -> str:
        """
        Generate a simple extractive summary.
        
        Args:
            text: Text to summarize
            max_sentences: Maximum sentences in summary
            
        Returns:
            Summary string
        """
        if not self.enable_summaries:
            return ""
        
        try:
            sentences = sent_tokenize(text)
            
            if len(sentences) <= max_sentences:
                return text
            
            # Simple extractive: take first and last sentences
            if len(sentences) >= 2:
                summary = sentences[0] + " " + sentences[-1]
            else:
                summary = sentences[0]
            
            return summary.strip()
            
        except Exception as e:
            logger.warning(f"Summary generation failed: {str(e)}")
            return text[:200] + "..." if len(text) > 200 else text
    
    # ===================== QUESTION GENERATION =====================
    
    def generate_questions(self, text: str, keywords: List[str]) -> List[str]:
        """
        Generate potential questions this chunk could answer.
        Uses simple heuristics based on content.
        
        Returns:
            List of potential questions
        """
        if not self.enable_questions:
            return []
        
        questions = []
        
        # Question templates based on keywords
        question_templates = [
            "What is {}?",
            "How does {} work?",
            "Why is {} important?",
            "What are the benefits of {}?",
            "How to use {}?"
        ]
        
        # Generate questions from top keywords
        for keyword in keywords[:3]:  # Top 3 keywords
            template = question_templates[len(questions) % len(question_templates)]
            questions.append(template.format(keyword))
        
        # Check for common question-indicating patterns in text
        text_lower = text.lower()
        
        if "how to" in text_lower or "steps" in text_lower:
            questions.append("What are the steps involved?")
        
        if "benefit" in text_lower or "advantage" in text_lower:
            questions.append("What are the advantages?")
        
        if "problem" in text_lower or "issue" in text_lower:
            questions.append("What problems does this address?")
        
        if "example" in text_lower:
            questions.append("Can you provide examples?")
        
        return list(set(questions))[:5]  # Return max 5 unique questions
    
    # ===================== TOPIC DETECTION =====================
    
    def detect_topics(self, text: str, keywords: List[str]) -> Dict:
        """
        Detect topics/categories based on keywords and content analysis.
        
        Returns:
            Dictionary with detected topics and confidence scores
        """
        if not self.enable_topics:
            return {'topics': [], 'primary_topic': None}
        
        # Define topic keywords (expandable based on your domain)
        topic_keywords = {
            'technology': ['software', 'hardware', 'computer', 'digital', 'technology', 
                          'algorithm', 'data', 'system', 'programming', 'code'],
            'business': ['business', 'company', 'market', 'customer', 'revenue', 
                        'strategy', 'management', 'sales', 'profit', 'enterprise'],
            'science': ['research', 'study', 'experiment', 'scientific', 'analysis',
                       'theory', 'hypothesis', 'results', 'findings', 'evidence'],
            'health': ['health', 'medical', 'patient', 'treatment', 'disease',
                      'medicine', 'clinical', 'doctor', 'hospital', 'care'],
            'education': ['education', 'learning', 'student', 'teaching', 'school',
                         'university', 'course', 'training', 'knowledge', 'study'],
            'finance': ['finance', 'investment', 'money', 'financial', 'bank',
                       'credit', 'loan', 'asset', 'portfolio', 'trading'],
            'legal': ['legal', 'law', 'court', 'regulation', 'compliance',
                     'contract', 'attorney', 'justice', 'policy', 'rights']
        }
        
        # Calculate topic scores
        text_lower = text.lower()
        all_text_words = set(word_tokenize(text_lower))
        keyword_set = set(kw.lower() for kw in keywords)
        
        topic_scores = {}
        
        for topic, topic_words in topic_keywords.items():
            # Count matches in text
            text_matches = len(all_text_words.intersection(set(topic_words)))
            # Count matches in keywords (weighted higher)
            keyword_matches = len(keyword_set.intersection(set(topic_words)))
            
            score = text_matches + (keyword_matches * 2)
            if score > 0:
                topic_scores[topic] = score
        
        # Sort by score
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        
        detected_topics = [topic for topic, score in sorted_topics if score >= 2]
        primary_topic = sorted_topics[0][0] if sorted_topics else "general"
        
        return {
            'topics': detected_topics,
            'primary_topic': primary_topic,
            'topic_scores': dict(sorted_topics[:3])  # Top 3
        }
    
    # ===================== STRUCTURAL ANALYSIS =====================
    
    def analyze_structure(self, text: str) -> Dict:
        """
        Analyze structural elements in the text.
        
        Returns:
            Dictionary with structural metadata
        """
        structure = {
            'has_headings': False,
            'has_lists': False,
            'has_numbers': False,
            'has_code': False,
            'has_urls': False,
            'paragraph_count': 0,
            'question_marks': 0
        }
        
        # Check for headings (lines that are short and followed by content)
        lines = text.split('\n')
        for i, line in enumerate(lines[:-1]):
            if len(line.strip()) < 100 and len(lines[i+1].strip()) > 50:
                structure['has_headings'] = True
                break
        
        # Check for lists (lines starting with -, *, numbers)
        list_pattern = r'^\s*[-*•]\s+|\d+\.\s+'
        if re.search(list_pattern, text, re.MULTILINE):
            structure['has_lists'] = True
        
        # Check for numbers/data
        if re.search(r'\d+', text):
            structure['has_numbers'] = True
        
        # Check for code (simple heuristic)
        code_indicators = ['def ', 'class ', 'function', '()', '{}', ';', '//']
        if any(indicator in text for indicator in code_indicators):
            structure['has_code'] = True
        
        # Check for URLs
        if re.search(r'https?://', text):
            structure['has_urls'] = True
        
        # Count paragraphs
        structure['paragraph_count'] = len([p for p in text.split('\n\n') if p.strip()])
        
        # Count questions
        structure['question_marks'] = text.count('?')
        
        return structure
    
    # ===================== CHUNK ENRICHMENT =====================
    
    def enrich_chunk(self, chunk: Dict) -> Dict:
        """
        Add all metadata to a single chunk.
        
        Args:
            chunk: Original chunk dictionary
            
        Returns:
            Enriched chunk dictionary
        """
        text = chunk.get('text', '')
        
        # Extract keywords
        if self.enable_keywords:
            keyword_data = self.extract_keywords(text)
            chunk['keywords'] = keyword_data['keywords']
            chunk['keyword_details'] = keyword_data['keyword_details']
            
            if keyword_data['keyword_count'] > 0:
                self.stats['chunks_with_keywords'] += 1
                self.stats['avg_keywords_per_chunk'].append(keyword_data['keyword_count'])
        
        # Extract entities
        if self.enable_entities:
            entity_data = self.extract_entities(text)
            chunk['entities'] = entity_data['entities']
            chunk['entity_types'] = entity_data['entity_types']
            chunk['entity_count'] = entity_data.get('entity_count', 0)
            
            if entity_data.get('entity_count', 0) > 0:
                self.stats['chunks_with_entities'] += 1
                self.stats['avg_entities_per_chunk'].append(entity_data['entity_count'])
        
        # Generate summary
        if self.enable_summaries:
            chunk['summary'] = self.generate_summary(text)
        
        # Generate questions
        if self.enable_questions:
            keywords = chunk.get('keywords', [])
            chunk['potential_questions'] = self.generate_questions(text, keywords)
        
        # Detect topics
        if self.enable_topics:
            keywords = chunk.get('keywords', [])
            topic_data = self.detect_topics(text, keywords)
            chunk['topics'] = topic_data['topics']
            chunk['primary_topic'] = topic_data['primary_topic']
            chunk['topic_scores'] = topic_data.get('topic_scores', {})
        
        # Analyze structure
        structure_data = self.analyze_structure(text)
        chunk['structure'] = structure_data
        
        # Add enrichment metadata
        chunk['enrichment_timestamp'] = datetime.now().isoformat()
        chunk['enrichment_version'] = '1.0'
        
        return chunk
    
    # ===================== FILE PROCESSING =====================
    
    def process_file(self, filepath: Path) -> List[Dict]:
        """
        Load and enrich all chunks from a JSONL file.
        
        Args:
            filepath: Path to JSONL file
            
        Returns:
            List of enriched chunks
        """
        logger.info(f"Processing: {filepath.name}")
        
        enriched_chunks = []
        
        try:
            # Read JSONL file
            with open(filepath, 'r', encoding='utf-8') as f:
                chunks = [json.loads(line) for line in f if line.strip()]
            
            self.stats['total_chunks'] += len(chunks)
            
            # Enrich each chunk
            for chunk in tqdm(chunks, desc=f"Enriching {filepath.name}", leave=False):
                try:
                    enriched_chunk = self.enrich_chunk(chunk)
                    enriched_chunks.append(enriched_chunk)
                    self.stats['enriched_chunks'] += 1
                except Exception as e:
                    logger.warning(f"Failed to enrich chunk {chunk.get('chunk_id', 'unknown')}: {str(e)}")
                    self.stats['failed_chunks'] += 1
                    enriched_chunks.append(chunk)  # Keep original if enrichment fails
            
            logger.info(f"[SUCCESS] Enriched {len(enriched_chunks)} chunks from {filepath.name}")
            
            return enriched_chunks
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to process {filepath.name}: {str(e)}")
            return []
    
    def save_enriched_chunks(self, chunks: List[Dict], original_filename: str):
        """Save enriched chunks to JSONL file."""
        if not chunks:
            return
        
        output_filename = original_filename.replace('_chunks.jsonl', '_enriched.jsonl')
        output_path = self.output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        
        logger.debug(f"Saved to: {output_path}")
    
    def process_all_files(self) -> Dict:
        """
        Process all chunk files in the input directory.
        
        Returns:
            Dictionary with processing results
        """
        logger.info("\n" + "="*60)
        logger.info("STARTING METADATA ENRICHMENT")
        logger.info("="*60)
        
        # Load models
        self._load_models()
        
        # Get all JSONL files
        jsonl_files = list(self.input_dir.glob('*_chunks.jsonl'))
        
        if not jsonl_files:
            logger.warning(f"No chunk files found in {self.input_dir}")
            return {'status': 'no_files'}
        
        logger.info(f"Found {len(jsonl_files)} chunk files to enrich")
        
        # Process each file
        for filepath in jsonl_files:
            enriched_chunks = self.process_file(filepath)
            if enriched_chunks:
                self.save_enriched_chunks(enriched_chunks, filepath.name)
        
        # Print summary
        self.print_summary()
        
        return {
            'status': 'success',
            'statistics': self.stats
        }
    
    def print_summary(self):
        """Print enrichment statistics."""
        logger.info("\n" + "="*60)
        logger.info("ENRICHMENT SUMMARY")
        logger.info("="*60)
        logger.info(f"Total chunks processed: {self.stats['total_chunks']}")
        logger.info(f"Successfully enriched: {self.stats['enriched_chunks']}")
        logger.info(f"Failed: {self.stats['failed_chunks']}")
        
        if self.stats['avg_keywords_per_chunk']:
            avg_kw = np.mean(self.stats['avg_keywords_per_chunk'])
            logger.info(f"\nKeyword Statistics:")
            logger.info(f"  Chunks with keywords: {self.stats['chunks_with_keywords']}")
            logger.info(f"  Average keywords per chunk: {avg_kw:.1f}")
        
        if self.stats['avg_entities_per_chunk']:
            avg_ent = np.mean(self.stats['avg_entities_per_chunk'])
            logger.info(f"\nEntity Statistics:")
            logger.info(f"  Chunks with entities: {self.stats['chunks_with_entities']}")
            logger.info(f"  Average entities per chunk: {avg_ent:.1f}")
        
        logger.info("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Initialize enricher
    enricher = MetadataEnricher(
        input_dir="data/chunked",
        output_dir="data/enriched",
        enable_keywords=True,        # Extract keywords
        enable_entities=True,        # Extract named entities
        enable_summaries=True,       # Generate summaries
        enable_questions=True,       # Generate potential questions
        enable_topics=True,          # Detect topics
        max_keywords=5               # Max keywords per chunk
    )
    
    # Process all files
    results = enricher.process_all_files()
    
    # Save results
    with open('logs/enrichment_results.json', 'w') as f:
        json.dump({
            'status': results['status'],
            'statistics': results.get('statistics', {})
        }, f, indent=2)
    
    print("\nMetadata enrichment complete!")
    print(f"Enriched files: data/enriched/")
    print(f"Results: logs/enrichment_results.json")