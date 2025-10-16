import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime
from collections import Counter
import math

# NLP and quality libraries
import textstat
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd

# Ignore DepricationWarnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import Visualization libraries
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

def convert_to_serializable(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [convert_to_serializable(item) for item in obj]
    return obj

from pathlib import Path
Path("logs").mkdir(parents=True, exist_ok=True)

import sys, os
if os.name == "nt":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/quality_assessment.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ContentQualityAssessor:
    """
    Comprehensive content quality assessment for RAG systems.
    Evaluates readability, coherence, completeness, and information density.
    """
    
    def __init__(self, cleaned_dir: str = "data/cleaned",
                 quality_dir: str = "data/quality_assessed",
                 quality_threshold: float = 0.5):
        """
        Initialize the quality assessment system.
        
        Args:
            cleaned_dir: Directory with cleaned text files
            quality_dir: Directory to save quality-assessed files
            quality_threshold: Minimum quality score (0-1) to pass
        """
        self.cleaned_dir = Path(cleaned_dir)
        self.quality_dir = Path(quality_dir)
        self.quality_threshold = quality_threshold
        
        # Create output directories
        self.quality_dir.mkdir(parents=True, exist_ok=True)
        (self.quality_dir / "high_quality").mkdir(exist_ok=True)
        (self.quality_dir / "low_quality").mkdir(exist_ok=True)
        
        # Load stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            logger.warning("NLTK stopwords not found. Downloading...")
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
        
        # Statistics tracking
        self.stats = {
            'total_files': 0,
            'high_quality': 0,
            'low_quality': 0,
            'avg_quality_score': [],
            'quality_distribution': []
        }
        
        # Quality score weights (customize based on your needs)
        self.weights = {
            'readability': 0.2,
            'coherence': 0.25,
            'information_density': 0.25,
            'completeness': 0.15,
            'linguistic_quality': 0.15
        }
    
    # ===================== READABILITY METRICS =====================
    
    def calculate_readability_score(self, text: str) -> Dict[str, float]:
        """
        Calculate multiple readability metrics and normalize to 0-1 scale.
        
        Metrics:
        - Flesch Reading Ease (0-100, higher = easier)
        - Flesch-Kincaid Grade Level
        - SMOG Index
        - Coleman-Liau Index
        - Automated Readability Index
        """
        try:
            # Flesch Reading Ease (0-100, normalize to 0-1)
            flesch_ease = textstat.flesch_reading_ease(text)
            # Convert to 0-1 where 1 is optimal (60-80 is ideal)
            if flesch_ease < 0:
                flesch_normalized = 0
            elif flesch_ease < 30:
                flesch_normalized = flesch_ease / 30 * 0.5
            elif flesch_ease <= 80:
                flesch_normalized = 0.5 + (flesch_ease - 30) / 50 * 0.5
            else:
                flesch_normalized = 1.0
            
            # Grade level (aim for 8-12 grade level)
            grade_level = textstat.flesch_kincaid_grade(text)
            if grade_level < 6:
                grade_normalized = 0.5
            elif grade_level <= 12:
                grade_normalized = 1.0
            elif grade_level <= 16:
                grade_normalized = 1.0 - (grade_level - 12) / 4 * 0.3
            else:
                grade_normalized = 0.5
            
            # Average sentence length (aim for 15-20 words)
            avg_sentence_length = textstat.avg_sentence_length(text)
            if 10 <= avg_sentence_length <= 25:
                sentence_length_score = 1.0
            elif avg_sentence_length < 10:
                sentence_length_score = avg_sentence_length / 10
            else:
                sentence_length_score = max(0, 1.0 - (avg_sentence_length - 25) / 25)
            
            # Composite readability score
            readability_score = (flesch_normalized * 0.4 + 
                               grade_normalized * 0.3 + 
                               sentence_length_score * 0.3)
            
            return {
                'readability_score': round(readability_score, 3),
                'flesch_reading_ease': round(flesch_ease, 2),
                'flesch_kincaid_grade': round(grade_level, 2),
                'avg_sentence_length': round(avg_sentence_length, 2),
                'flesch_normalized': round(flesch_normalized, 3),
                'grade_normalized': round(grade_normalized, 3)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating readability: {str(e)}")
            return {'readability_score': 0.5, 'error': str(e)}
    
    # ===================== COHERENCE METRICS =====================
    
    def calculate_coherence_score(self, text: str) -> Dict[str, float]:
        """
        Measure text coherence through various linguistic features.
        
        Checks:
        - Sentence connectivity
        - Lexical cohesion (word repetition patterns)
        - Logical flow indicators
        """
        try:
            sentences = sent_tokenize(text)
            
            if len(sentences) < 2:
                return {'coherence_score': 0.5, 'reason': 'too_few_sentences'}
            
            # 1. Sentence transition quality (check for transition words)
            transition_words = {
                'however', 'therefore', 'furthermore', 'moreover', 'additionally',
                'consequently', 'thus', 'hence', 'nevertheless', 'meanwhile',
                'subsequently', 'first', 'second', 'finally', 'also', 'besides',
                'in addition', 'for example', 'for instance', 'in contrast'
            }
            
            transition_count = 0
            for sentence in sentences:
                words = word_tokenize(sentence.lower())
                if any(word in transition_words for word in words):
                    transition_count += 1
            
            transition_ratio = transition_count / len(sentences)
            transition_score = min(1.0, transition_ratio * 3)  # Normalize
            
            # 2. Lexical cohesion (word overlap between adjacent sentences)
            cohesion_scores = []
            for i in range(len(sentences) - 1):
                words1 = set(word_tokenize(sentences[i].lower()))
                words2 = set(word_tokenize(sentences[i + 1].lower()))
                
                # Remove stopwords
                words1 = words1 - self.stop_words
                words2 = words2 - self.stop_words
                
                if len(words1) > 0 and len(words2) > 0:
                    overlap = len(words1 & words2) / len(words1 | words2)
                    cohesion_scores.append(overlap)
            
            avg_cohesion = np.mean(cohesion_scores) if cohesion_scores else 0.1
            # Normalize (optimal overlap is around 0.2-0.4)
            if 0.2 <= avg_cohesion <= 0.4:
                cohesion_score = 1.0
            elif avg_cohesion < 0.2:
                cohesion_score = avg_cohesion / 0.2
            else:
                cohesion_score = max(0.5, 1.0 - (avg_cohesion - 0.4) / 0.6)
            
            # 3. Sentence length variance (too much variance = choppy)
            sentence_lengths = [len(word_tokenize(s)) for s in sentences]
            length_variance = np.std(sentence_lengths) / (np.mean(sentence_lengths) + 1)
            # Lower variance is better (more consistent style)
            variance_score = max(0, 1.0 - length_variance / 2)
            
            # Composite coherence score
            coherence_score = (transition_score * 0.3 + 
                             cohesion_score * 0.5 + 
                             variance_score * 0.2)
            
            return {
                'coherence_score': round(coherence_score, 3),
                'transition_ratio': round(transition_ratio, 3),
                'lexical_cohesion': round(avg_cohesion, 3),
                'length_variance': round(length_variance, 3),
                'transition_score': round(transition_score, 3),
                'cohesion_score': round(cohesion_score, 3)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating coherence: {str(e)}")
            return {'coherence_score': 0.5, 'error': str(e)}
    
    # ===================== INFORMATION DENSITY =====================
    
    def calculate_information_density(self, text: str) -> Dict[str, float]:
        """
        Measure the information content relative to text length.
        
        Metrics:
        - Lexical diversity (unique words / total words)
        - Content word ratio (nouns, verbs, adjectives / all words)
        - Named entity density (approximation)
        """
        try:
            words = word_tokenize(text.lower())
            
            if len(words) < 10:
                return {'information_density': 0.3, 'reason': 'too_short'}
            
            # 1. Lexical diversity (Type-Token Ratio)
            unique_words = set(words)
            ttr = len(unique_words) / len(words)
            # Normalize (typical range 0.4-0.8 for good text)
            ttr_score = min(1.0, ttr / 0.6)
            
            # 2. Content word ratio (words that aren't stopwords)
            content_words = [w for w in words if w not in self.stop_words and w.isalpha()]
            content_ratio = len(content_words) / len(words)
            # Normalize (typical range 0.4-0.7)
            content_score = min(1.0, content_ratio / 0.6)
            
            # 3. Average word length (longer words = more information)
            avg_word_length = np.mean([len(w) for w in words if w.isalpha()])
            # Normalize (typical range 4-7 characters)
            if 4 <= avg_word_length <= 7:
                word_length_score = 1.0
            elif avg_word_length < 4:
                word_length_score = avg_word_length / 4
            else:
                word_length_score = max(0.5, 1.0 - (avg_word_length - 7) / 5)
            
            # 4. Number density (presence of numbers often indicates factual content)
            numbers = [w for w in words if any(c.isdigit() for c in w)]
            number_ratio = len(numbers) / len(words)
            # Normalize (5-15% is typical for informative text)
            if 0.05 <= number_ratio <= 0.15:
                number_score = 1.0
            elif number_ratio < 0.05:
                number_score = number_ratio / 0.05
            else:
                number_score = max(0.5, 1.0 - (number_ratio - 0.15) / 0.15)
            
            # Composite information density
            density_score = (ttr_score * 0.3 + 
                           content_score * 0.3 + 
                           word_length_score * 0.2 + 
                           number_score * 0.2)
            
            return {
                'information_density': round(density_score, 3),
                'lexical_diversity': round(ttr, 3),
                'content_word_ratio': round(content_ratio, 3),
                'avg_word_length': round(avg_word_length, 2),
                'number_ratio': round(number_ratio, 3),
                'ttr_score': round(ttr_score, 3)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating information density: {str(e)}")
            return {'information_density': 0.5, 'error': str(e)}
    
    # ===================== COMPLETENESS METRICS =====================
    
    def calculate_completeness_score(self, text: str) -> Dict[str, float]:
        """
        Detect if text is complete or truncated.
        
        Checks:
        - Sentence completion (does last sentence end properly?)
        - Document structure indicators
        - Abnormal ending patterns
        """
        try:
            sentences = sent_tokenize(text)
            
            if not sentences:
                return {'completeness_score': 0.0, 'reason': 'no_sentences'}
            
            # 1. Last sentence completion
            last_sentence = sentences[-1].strip()
            ends_properly = last_sentence[-1] in '.!?'
            ending_score = 1.0 if ends_properly else 0.3
            
            # 2. Check for truncation indicators
            truncation_indicators = [
                r'\.\.\.$',  # Ends with ...
                r'continued',  # Contains "continued"
                r'see next page',
                r'to be continued',
                r'\[truncated\]',
                r'\(cont\)',
            ]
            
            has_truncation = any(re.search(pattern, text.lower()) 
                                for pattern in truncation_indicators)
            truncation_score = 0.2 if has_truncation else 1.0
            
            # 3. Document structure (presence of conclusion indicators)
            conclusion_indicators = [
                'conclusion', 'summary', 'in conclusion', 'to conclude',
                'finally', 'in summary', 'to sum up', 'overall'
            ]
            
            has_conclusion = any(indicator in text.lower() 
                                for indicator in conclusion_indicators)
            conclusion_score = 1.0 if has_conclusion else 0.7
            
            # 4. Length appropriateness (very short might be truncated)
            word_count = len(word_tokenize(text))
            if word_count < 50:
                length_score = 0.3
            elif word_count < 100:
                length_score = 0.7
            else:
                length_score = 1.0
            
            # Composite completeness score
            completeness = (ending_score * 0.3 + 
                          truncation_score * 0.3 + 
                          conclusion_score * 0.2 + 
                          length_score * 0.2)
            
            return {
                'completeness_score': round(completeness, 3),
                'ends_properly': ends_properly,
                'has_truncation_indicators': has_truncation,
                'has_conclusion': has_conclusion,
                'word_count': word_count,
                'ending_score': round(ending_score, 3)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating completeness: {str(e)}")
            return {'completeness_score': 0.5, 'error': str(e)}
    
    # ===================== LINGUISTIC QUALITY =====================
    
    def calculate_linguistic_quality(self, text: str) -> Dict[str, float]:
        """
        Assess basic linguistic quality.
        
        Checks:
        - Proper capitalization
        - Punctuation consistency
        - Word validity
        """
        try:
            # 1. Capitalization check (sentences should start with capital)
            sentences = sent_tokenize(text)
            properly_capitalized = sum(1 for s in sentences 
                                      if s and s[0].isupper())
            capitalization_ratio = properly_capitalized / len(sentences) if sentences else 0
            capitalization_score = capitalization_ratio
            
            # 2. Punctuation consistency
            words = word_tokenize(text)
            sentences_count = len(sentences)
            words_count = len([w for w in words if w.isalpha()])
            
            if words_count > 0:
                words_per_sentence = words_count / sentences_count if sentences_count > 0 else 0
                # Optimal: 10-25 words per sentence
                if 10 <= words_per_sentence <= 25:
                    punctuation_score = 1.0
                elif words_per_sentence < 10:
                    punctuation_score = words_per_sentence / 10
                else:
                    punctuation_score = max(0.5, 1.0 - (words_per_sentence - 25) / 25)
            else:
                punctuation_score = 0.5
            
            # 3. Check for excessive special characters (indicates poor quality)
            alpha_chars = sum(1 for c in text if c.isalpha())
            total_chars = len(text)
            alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0
            # Should be at least 70% alphabetic
            alpha_score = min(1.0, alpha_ratio / 0.7)
            
            # Composite linguistic quality
            linguistic_quality = (capitalization_score * 0.3 + 
                                punctuation_score * 0.4 + 
                                alpha_score * 0.3)
            
            return {
                'linguistic_quality': round(linguistic_quality, 3),
                'capitalization_ratio': round(capitalization_ratio, 3),
                'words_per_sentence': round(words_per_sentence, 2),
                'alpha_ratio': round(alpha_ratio, 3),
                'capitalization_score': round(capitalization_score, 3)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating linguistic quality: {str(e)}")
            return {'linguistic_quality': 0.5, 'error': str(e)}
    
    # ===================== COMPOSITE QUALITY SCORE =====================
    
    def calculate_composite_quality(self, metrics: Dict) -> Dict[str, float]:
        """
        Calculate final composite quality score from all metrics.
        
        Args:
            metrics: Dictionary containing all individual metric scores
            
        Returns:
            Dictionary with composite score and breakdown
        """
        # Extract individual scores
        readability = metrics.get('readability', {}).get('readability_score', 0.5)
        coherence = metrics.get('coherence', {}).get('coherence_score', 0.5)
        info_density = metrics.get('information_density', {}).get('information_density', 0.5)
        completeness = metrics.get('completeness', {}).get('completeness_score', 0.5)
        linguistic = metrics.get('linguistic_quality', {}).get('linguistic_quality', 0.5)
        
        # Calculate weighted composite score
        composite_score = (
            readability * self.weights['readability'] +
            coherence * self.weights['coherence'] +
            info_density * self.weights['information_density'] +
            completeness * self.weights['completeness'] +
            linguistic * self.weights['linguistic_quality']
        )
        
        # Determine quality category
        if composite_score >= 0.8:
            category = 'excellent'
        elif composite_score >= 0.65:
            category = 'good'
        elif composite_score >= 0.5:
            category = 'acceptable'
        elif composite_score >= 0.35:
            category = 'poor'
        else:
            category = 'very_poor'
        
        # Identify weakest areas
        score_dict = {
            'readability': readability,
            'coherence': coherence,
            'information_density': info_density,
            'completeness': completeness,
            'linguistic_quality': linguistic
        }
        
        weakest_metric = min(score_dict.items(), key=lambda x: x[1])
        
        return {
            'composite_quality_score': round(composite_score, 3),
            'quality_category': category,
            'passes_threshold': composite_score >= self.quality_threshold,
            'weakest_metric': weakest_metric[0],
            'weakest_score': round(weakest_metric[1], 3),
            'score_breakdown': {k: round(v, 3) for k, v in score_dict.items()}
        }
    
    # ===================== FILE PROCESSING =====================
    
    def assess_file(self, filepath: Path) -> Dict:
        """
        Assess the quality of a single file.
        
        Returns:
            Dictionary with all quality metrics and scores
        """
        logger.info(f"Assessing file: {filepath.name}")
        
        try:
            # Read the file
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Calculate all metrics
            readability_metrics = self.calculate_readability_score(text)
            coherence_metrics = self.calculate_coherence_score(text)
            density_metrics = self.calculate_information_density(text)
            completeness_metrics = self.calculate_completeness_score(text)
            linguistic_metrics = self.calculate_linguistic_quality(text)
            
            # Combine all metrics
            all_metrics = {
                'readability': readability_metrics,
                'coherence': coherence_metrics,
                'information_density': density_metrics,
                'completeness': completeness_metrics,
                'linguistic_quality': linguistic_metrics
            }
            
            # Calculate composite score
            composite = self.calculate_composite_quality(all_metrics)
            
            # Determine output directory
            if composite['passes_threshold']:
                output_dir = self.quality_dir / "high_quality"
                self.stats['high_quality'] += 1
            else:
                output_dir = self.quality_dir / "low_quality"
                self.stats['low_quality'] += 1
            
            # Copy file to appropriate directory
            output_filename = filepath.stem.replace('_cleaned', '') + '_assessed.txt'
            output_path = output_dir / output_filename
            
            with open(filepath, 'r', encoding='utf-8') as f_in:
                with open(output_path, 'w', encoding='utf-8') as f_out:
                    f_out.write(f_in.read())
            
            # Update statistics
            self.stats['avg_quality_score'].append(composite['composite_quality_score'])
            self.stats['quality_distribution'].append({
                'filename': filepath.name,
                'score': composite['composite_quality_score'],
                'category': composite['quality_category']
            })
            
            # Compile results
            result = {
                'source_file': filepath.name,
                'output_file': output_filename,
                'output_path': str(output_path),
                'assessment_timestamp': datetime.now().isoformat(),
                'composite_score': composite,
                'detailed_metrics': all_metrics,
                'status': 'success'
            }
            
            logger.info(f"✓ Quality Score: {composite['composite_quality_score']:.3f} "
                       f"({composite['quality_category']}) - {filepath.name}")
            
            return result
            
        except Exception as e:
            logger.error(f"✗ Failed to assess {filepath.name}: {str(e)}")
            return {
                'source_file': filepath.name,
                'status': 'failed',
                'error_message': str(e),
                'assessment_timestamp': datetime.now().isoformat()
            }
    
    def assess_all_files(self) -> List[Dict]:
        """
        Assess all files in the cleaned directory.
        
        Returns:
            List of assessment results
        """
        logger.info(f"Starting quality assessment from {self.cleaned_dir}")
        
        # Get all text files
        all_files = list(self.cleaned_dir.glob('*.txt'))
        self.stats['total_files'] = len(all_files)
        
        logger.info(f"Found {len(all_files)} files to assess")
        
        # Process each file
        results = []
        for filepath in all_files:
            result = self.assess_file(filepath)
            results.append(result)
        
        # Generate summary
        self.print_summary()
        self.generate_report(results)
        
        return results
    
    def print_summary(self):
        """Print assessment statistics."""
        logger.info("\n" + "="*60)
        logger.info("QUALITY ASSESSMENT SUMMARY")
        logger.info("="*60)
        logger.info(f"Total files assessed: {self.stats['total_files']}")
        logger.info(f"High quality (≥{self.quality_threshold}): {self.stats['high_quality']}")
        logger.info(f"Low quality (<{self.quality_threshold}): {self.stats['low_quality']}")
        
        if self.stats['avg_quality_score']:
            avg_score = np.mean(self.stats['avg_quality_score'])
            min_score = np.min(self.stats['avg_quality_score'])
            max_score = np.max(self.stats['avg_quality_score'])
            
            logger.info(f"Average quality score: {avg_score:.3f}")
            logger.info(f"Score range: {min_score:.3f} - {max_score:.3f}")
            
            # Category distribution
            categories = [d['category'] for d in self.stats['quality_distribution']]
            category_counts = Counter(categories)
            logger.info("\nCategory Distribution:")
            for category, count in category_counts.most_common():
                logger.info(f"  {category}: {count}")
        
        logger.info("="*60 + "\n")
    
    def generate_report(self, results: List[Dict]):
        """Generate detailed quality report as CSV."""
        try:
            # Prepare data for DataFrame
            report_data = []
            for result in results:
                if result['status'] == 'success':
                    composite = result['composite_score']
                    breakdown = composite['score_breakdown']
                    
                    report_data.append({
                        'filename': result['source_file'],
                        'composite_score': composite['composite_quality_score'],
                        'category': composite['quality_category'],
                        'passes_threshold': composite['passes_threshold'],
                        'readability': breakdown['readability'],
                        'coherence': breakdown['coherence'],
                        'information_density': breakdown['information_density'],
                        'completeness': breakdown['completeness'],
                        'linguistic_quality': breakdown['linguistic_quality'],
                        'weakest_metric': composite['weakest_metric'],
                        'output_path': result['output_path']
                    })
            
            # Create DataFrame and save
            df = pd.DataFrame(report_data)
            df = df.sort_values('composite_score', ascending=False)
            
            report_path = 'logs/quality_assessment_report.csv'
            df.to_csv(report_path, index=False)
            logger.info(f"Quality report saved to: {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Initialize the quality assessor
    assessor = ContentQualityAssessor(
        cleaned_dir="data/cleaned",
        quality_dir="data/quality_assessed",
        quality_threshold=0.5  # Adjust this threshold based on your needs
    )
    
    # Assess all files
    results = assessor.assess_all_files()
    
    # Save detailed results to JSON
    with open('logs/quality_assessment_results.json', 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    print("\nQuality assessment complete!")
    print(f"High quality files: data/quality_assessed/high_quality/")
    print(f"Low quality files: data/quality_assessed/low_quality/")
    print(f"Detailed report: logs/quality_assessment_report.csv")

def create_quality_dashboard(results: List[Dict]):
    """Create visualizations of quality metrics."""
    Path("logs").mkdir(parents=True, exist_ok=True)

    # Extract scores
    scores = []
    categories = []
    filenames = []
    
    for result in results:
        if result['status'] == 'success':
            scores.append(result['composite_score']['composite_quality_score'])
            categories.append(result['composite_score']['quality_category'])
            filenames.append(result['source_file'])
    if not scores:
        print("No successful results to plot.")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Content Quality Assessment Dashboard', fontsize=16)
    
    # 1. Score distribution histogram
    axes[0, 0].hist(scores, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Quality Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Quality Score Distribution')
    axes[0, 0].axvline(x=0.5, color='r', linestyle='--', label='Threshold')
    axes[0, 0].legend()
    
    # 2. Category counts
    from collections import Counter
    cat_counts = Counter(categories)
    axes[0, 1].bar(cat_counts.keys(), cat_counts.values(), alpha=0.7)
    axes[0, 1].set_xlabel('Quality Category')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Documents by Category')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Score breakdown by dimension
    dimensions = ['readability', 'coherence', 'information_density', 
                  'completeness', 'linguistic_quality']
    dim_scores = {dim: [] for dim in dimensions}
    
    for result in results:
        if result['status'] == 'success':
            breakdown = result['composite_score']['score_breakdown']
            for dim in dimensions:
                dim_scores[dim].append(breakdown[dim])
    
    avg_dim_scores = {dim: np.mean(scores) for dim, scores in dim_scores.items()}
    axes[1, 0].bar(range(len(dimensions)), avg_dim_scores.values(), alpha=0.7)
    axes[1, 0].set_xticks(range(len(dimensions)))
    axes[1, 0].set_xticklabels([d.replace('_', '\n') for d in dimensions], 
                                rotation=45, ha='right')
    axes[1, 0].set_ylabel('Average Score')
    axes[1, 0].set_title('Average Scores by Dimension')
    axes[1, 0].set_ylim(0, 1)
    
    # 4. Top and bottom performers
    sorted_results = sorted(zip(filenames, scores), key=lambda x: x[1])
    top_5 = sorted_results[-5:]
    bottom_5 = sorted_results[:5]
    
    all_display = top_5 + bottom_5
    names = [name[:20] + '...' if len(name) > 20 else name 
             for name, _ in all_display]
    values = [score for _, score in all_display]
    colors = ['green']*len(top_5) + ['red']*len(bottom_5)
    
    axes[1, 1].barh(range(len(names)), values, color=colors, alpha=0.7)
    axes[1, 1].set_yticks(range(len(names)))
    axes[1, 1].set_yticklabels(names, fontsize=8)
    axes[1, 1].set_xlabel('Quality Score')
    axes[1, 1].set_title('Top 5 and Bottom 5 Documents')
    axes[1, 1].axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('logs/quality_dashboard.png', dpi=300, bbox_inches='tight')
    print("Dashboard saved to: logs/quality_dashboard.png")
    plt.close()
    
# Create dashboard
create_quality_dashboard(results)