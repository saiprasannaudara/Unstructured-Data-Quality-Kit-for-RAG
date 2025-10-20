"""
Enriched Metadata Viewer
View and inspect enriched chunk metadata
"""

import json
from pathlib import Path
import argparse
from typing import List, Dict
from collections import Counter


class EnrichmentViewer:
    """View and analyze enriched chunk metadata."""
    
    def __init__(self, enriched_dir: str = "data/enriched"):
        self.enriched_dir = Path(enriched_dir)
        self.chunks = []
        self.load_all_chunks()
    
    def load_all_chunks(self):
        """Load all enriched chunks."""
        jsonl_files = list(self.enriched_dir.glob("*_enriched.jsonl"))
        
        for file in jsonl_files:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.chunks.append(json.loads(line))
        
        print(f"Loaded {len(self.chunks)} enriched chunks from {len(jsonl_files)} files\n")
    
    def show_chunk_detail(self, chunk_id: str = None, index: int = None):
        """Display detailed view of a specific chunk."""
        
        if chunk_id:
            chunk = next((c for c in self.chunks if c['chunk_id'] == chunk_id), None)
        elif index is not None:
            chunk = self.chunks[index] if 0 <= index < len(self.chunks) else None
        else:
            chunk = self.chunks[0]  # Show first chunk by default
        
        if not chunk:
            print(f"Chunk not found!")
            return
        
        print("=" * 70)
        print(f"CHUNK DETAILS: {chunk['chunk_id']}")
        print("=" * 70)
        
        # Basic info
        print(f"\n📄 Basic Information:")
        print(f"  Source File: {chunk.get('source_file', 'N/A')}")
        print(f"  Position: {chunk.get('chunk_index', 0) + 1} of {chunk.get('total_chunks', 'N/A')}")
        print(f"  Tokens: {chunk.get('token_count', 'N/A')}")
        print(f"  Words: {chunk.get('word_count', 'N/A')}")
        print(f"  Quality Score: {chunk.get('quality_score', 'N/A')}")
        
        # Keywords
        if 'keywords' in chunk and chunk['keywords']:
            print(f"\n🔑 Keywords ({len(chunk['keywords'])}):")
            for kw in chunk['keywords'][:10]:  # Show top 10
                score = chunk.get('keyword_details', {}).get(kw, {}).get('score', 'N/A')
                print(f"  • {kw} (score: {score})")
        
        # Named Entities
        if 'entity_types' in chunk and chunk['entity_types']:
            print(f"\n🏷️  Named Entities ({chunk.get('entity_count', 0)}):")
            for entity_type, entities in chunk['entity_types'].items():
                print(f"  {entity_type}: {', '.join(entities[:5])}")  # Show first 5
        
        # Topics
        if 'topics' in chunk and chunk['topics']:
            print(f"\n📚 Topics:")
            print(f"  Primary: {chunk.get('primary_topic', 'N/A')}")
            print(f"  All: {', '.join(chunk['topics'])}")
        
        # Summary
        if 'summary' in chunk and chunk['summary']:
            print(f"\n📝 Summary:")
            print(f"  {chunk['summary'][:200]}...")
        
        # Potential Questions
        if 'potential_questions' in chunk and chunk['potential_questions']:
            print(f"\n❓ Potential Questions:")
            for q in chunk['potential_questions'][:5]:
                print(f"  • {q}")
        
        # Structure
        if 'structure' in chunk:
            struct = chunk['structure']
            print(f"\n🏗️  Structure:")
            print(f"  Has headings: {struct.get('has_headings', False)}")
            print(f"  Has lists: {struct.get('has_lists', False)}")
            print(f"  Has code: {struct.get('has_code', False)}")
            print(f"  Paragraphs: {struct.get('paragraph_count', 0)}")
        
        # Text preview
        print(f"\n📖 Text Preview (first 300 chars):")
        print("-" * 70)
        text = chunk.get('text', '')
        print(text[:300] + ("..." if len(text) > 300 else ""))
        print("-" * 70)
    
    def show_statistics(self):
        """Display overall statistics."""
        print("=" * 70)
        print("ENRICHMENT STATISTICS")
        print("=" * 70)
        
        # Overall counts
        print(f"\n📊 Overall:")
        print(f"  Total Chunks: {len(self.chunks)}")
        
        # Keyword statistics
        chunks_with_keywords = sum(1 for c in self.chunks if c.get('keywords'))
        if chunks_with_keywords > 0:
            total_keywords = sum(len(c.get('keywords', [])) for c in self.chunks)
            avg_keywords = total_keywords / len(self.chunks)
            print(f"\n🔑 Keywords:")
            print(f"  Chunks with keywords: {chunks_with_keywords}")
            print(f"  Average per chunk: {avg_keywords:.1f}")
            
            # Most common keywords
            all_keywords = []
            for c in self.chunks:
                all_keywords.extend(c.get('keywords', []))
            keyword_counts = Counter(all_keywords)
            print(f"\n  Top 10 Keywords:")
            for kw, count in keyword_counts.most_common(10):
                print(f"    • {kw}: {count}")
        
        # Entity statistics
        chunks_with_entities = sum(1 for c in self.chunks if c.get('entity_count', 0) > 0)
        if chunks_with_entities > 0:
            total_entities = sum(c.get('entity_count', 0) for c in self.chunks)
            avg_entities = total_entities / len(self.chunks)
            print(f"\n🏷️  Named Entities:")
            print(f"  Chunks with entities: {chunks_with_entities}")
            print(f"  Average per chunk: {avg_entities:.1f}")
            
            # Entity type distribution
            entity_type_counts = Counter()
            for c in self.chunks:
                for entity_type in c.get('entity_types', {}).keys():
                    entity_type_counts[entity_type] += len(c['entity_types'][entity_type])
            
            print(f"\n  Entity Type Distribution:")
            for etype, count in entity_type_counts.most_common(10):
                print(f"    • {etype}: {count}")
        
        # Topic statistics
        all_topics = []
        for c in self.chunks:
            all_topics.extend(c.get('topics', []))
        
        if all_topics:
            topic_counts = Counter(all_topics)
            print(f"\n📚 Topics:")
            print(f"  Unique topics: {len(topic_counts)}")
            print(f"\n  Top 10 Topics:")
            for topic, count in topic_counts.most_common(10):
                print(f"    • {topic}: {count} chunks")
        
        # Structure statistics
        has_headings = sum(1 for c in self.chunks if c.get('structure', {}).get('has_headings'))
        has_lists = sum(1 for c in self.chunks if c.get('structure', {}).get('has_lists'))
        has_code = sum(1 for c in self.chunks if c.get('structure', {}).get('has_code'))
        has_numbers = sum(1 for c in self.chunks if c.get('structure', {}).get('has_numbers'))
        
        print(f"\n🏗️  Structure:")
        print(f"  Chunks with headings: {has_headings}")
        print(f"  Chunks with lists: {has_lists}")
        print(f"  Chunks with code: {has_code}")
        print(f"  Chunks with numbers: {has_numbers}")
        
        # Quality distribution
        print(f"\n✨ Quality Distribution:")
        quality_bins = {
            'Excellent (0.9-1.0)': sum(1 for c in self.chunks if c.get('quality_score', 0) >= 0.9),
            'Good (0.7-0.9)': sum(1 for c in self.chunks if 0.7 <= c.get('quality_score', 0) < 0.9),
            'Fair (0.5-0.7)': sum(1 for c in self.chunks if 0.5 <= c.get('quality_score', 0) < 0.7),
            'Poor (<0.5)': sum(1 for c in self.chunks if c.get('quality_score', 0) < 0.5)
        }
        
        for label, count in quality_bins.items():
            percentage = (count / len(self.chunks)) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
    
    def search_by_keyword(self, keyword: str, max_results: int = 10):
        """Search chunks by keyword."""
        print(f"\n🔍 Searching for keyword: '{keyword}'")
        print("=" * 70)
        
        matching_chunks = []
        for chunk in self.chunks:
            # Check if keyword is in extracted keywords
            keywords = [kw.lower() for kw in chunk.get('keywords', [])]
            if keyword.lower() in keywords:
                matching_chunks.append(chunk)
            # Also check in text
            elif keyword.lower() in chunk.get('text', '').lower():
                matching_chunks.append(chunk)
        
        print(f"\nFound {len(matching_chunks)} matching chunks\n")
        
        for i, chunk in enumerate(matching_chunks[:max_results]):
            print(f"[{i+1}] {chunk['chunk_id']}")
            print(f"    Source: {chunk.get('source_file', 'N/A')}")
            print(f"    Keywords: {', '.join(chunk.get('keywords', [])[:5])}")
            
            # Show snippet with keyword highlighted
            text = chunk.get('text', '')
            keyword_pos = text.lower().find(keyword.lower())
            if keyword_pos >= 0:
                start = max(0, keyword_pos - 50)
                end = min(len(text), keyword_pos + len(keyword) + 50)
                snippet = text[start:end]
                print(f"    Preview: ...{snippet}...")
            print()
    
    def search_by_entity(self, entity_name: str, max_results: int = 10):
        """Search chunks by entity name."""
        print(f"\n🏷️  Searching for entity: '{entity_name}'")
        print("=" * 70)
        
        matching_chunks = []
        for chunk in self.chunks:
            entity_types = chunk.get('entity_types', {})
            for entity_type, entities in entity_types.items():
                if any(entity_name.lower() in e.lower() for e in entities):
                    matching_chunks.append((chunk, entity_type))
                    break
        
        print(f"\nFound {len(matching_chunks)} matching chunks\n")
        
        for i, (chunk, entity_type) in enumerate(matching_chunks[:max_results]):
            print(f"[{i+1}] {chunk['chunk_id']} - Type: {entity_type}")
            print(f"    Source: {chunk.get('source_file', 'N/A')}")
            entities = chunk.get('entity_types', {}).get(entity_type, [])
            print(f"    Entities: {', '.join(entities[:5])}")
            print()
    
    def search_by_topic(self, topic: str, max_results: int = 10):
        """Search chunks by topic."""
        print(f"\n📚 Searching for topic: '{topic}'")
        print("=" * 70)
        
        matching_chunks = [
            c for c in self.chunks 
            if topic.lower() in [t.lower() for t in c.get('topics', [])]
        ]
        
        print(f"\nFound {len(matching_chunks)} matching chunks\n")
        
        for i, chunk in enumerate(matching_chunks[:max_results]):
            print(f"[{i+1}] {chunk['chunk_id']}")
            print(f"    Source: {chunk.get('source_file', 'N/A')}")
            print(f"    Primary Topic: {chunk.get('primary_topic', 'N/A')}")
            print(f"    All Topics: {', '.join(chunk.get('topics', []))}")
            print(f"    Keywords: {', '.join(chunk.get('keywords', [])[:5])}")
            print()
    
    def list_all_topics(self):
        """List all unique topics found."""
        all_topics = []
        for c in self.chunks:
            all_topics.extend(c.get('topics', []))
        
        topic_counts = Counter(all_topics)
        
        print("\n📚 All Topics Found:")
        print("=" * 70)
        for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {topic}: {count} chunks")
    
    def list_all_entities(self, entity_type: str = None):
        """List all unique entities, optionally filtered by type."""
        entity_counter = Counter()
        entity_type_filter = entity_type.upper() if entity_type else None
        
        for chunk in self.chunks:
            entity_types = chunk.get('entity_types', {})
            for etype, entities in entity_types.items():
                if entity_type_filter is None or etype == entity_type_filter:
                    entity_counter.update(entities)
        
        title = f"All {entity_type_filter} Entities" if entity_type_filter else "All Named Entities"
        print(f"\n🏷️  {title}:")
        print("=" * 70)
        
        for entity, count in entity_counter.most_common(50):
            print(f"  {entity}: {count} occurrences")
    
    def compare_chunks(self, chunk_id1: str, chunk_id2: str):
        """Compare two chunks side by side."""
        chunk1 = next((c for c in self.chunks if c['chunk_id'] == chunk_id1), None)
        chunk2 = next((c for c in self.chunks if c['chunk_id'] == chunk_id2), None)
        
        if not chunk1 or not chunk2:
            print("One or both chunks not found!")
            return
        
        print("=" * 70)
        print(f"COMPARING CHUNKS")
        print("=" * 70)
        
        print(f"\nChunk 1: {chunk_id1}")
        print(f"Chunk 2: {chunk_id2}")
        print()
        
        # Compare basic stats
        print("📊 Basic Stats Comparison:")
        print(f"  Tokens: {chunk1.get('token_count', 'N/A')} vs {chunk2.get('token_count', 'N/A')}")
        print(f"  Quality: {chunk1.get('quality_score', 'N/A')} vs {chunk2.get('quality_score', 'N/A')}")
        
        # Compare keywords
        kw1 = set(chunk1.get('keywords', []))
        kw2 = set(chunk2.get('keywords', []))
        common_kw = kw1 & kw2
        
        print(f"\n🔑 Keywords:")
        print(f"  Chunk 1: {', '.join(list(kw1)[:5])}")
        print(f"  Chunk 2: {', '.join(list(kw2)[:5])}")
        print(f"  Common: {', '.join(common_kw) if common_kw else 'None'}")
        
        # Compare topics
        topics1 = set(chunk1.get('topics', []))
        topics2 = set(chunk2.get('topics', []))
        common_topics = topics1 & topics2
        
        print(f"\n📚 Topics:")
        print(f"  Chunk 1: {', '.join(topics1)}")
        print(f"  Chunk 2: {', '.join(topics2)}")
        print(f"  Common: {', '.join(common_topics) if common_topics else 'None'}")
    
    def export_metadata_summary(self, output_file: str = "logs/metadata_summary.json"):
        """Export a summary of all metadata to JSON."""
        
        # Collect statistics
        summary = {
            'total_chunks': len(self.chunks),
            'statistics': {
                'keywords': {
                    'chunks_with_keywords': sum(1 for c in self.chunks if c.get('keywords')),
                    'total_keywords': sum(len(c.get('keywords', [])) for c in self.chunks),
                    'avg_per_chunk': sum(len(c.get('keywords', [])) for c in self.chunks) / len(self.chunks)
                },
                'entities': {
                    'chunks_with_entities': sum(1 for c in self.chunks if c.get('entity_count', 0) > 0),
                    'total_entities': sum(c.get('entity_count', 0) for c in self.chunks),
                    'avg_per_chunk': sum(c.get('entity_count', 0) for c in self.chunks) / len(self.chunks)
                },
                'topics': {
                    'unique_topics': len(set(t for c in self.chunks for t in c.get('topics', []))),
                    'chunks_with_topics': sum(1 for c in self.chunks if c.get('topics'))
                }
            },
            'top_keywords': dict(Counter([kw for c in self.chunks for kw in c.get('keywords', [])]).most_common(20)),
            'top_topics': dict(Counter([t for c in self.chunks for t in c.get('topics', [])]).most_common(20)),
            'entity_types': dict(Counter([
                et for c in self.chunks 
                for et in c.get('entity_types', {}).keys()
            ])),
            'generated_at': str(Path(output_file).parent.resolve())
        }
        
        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Metadata summary exported to: {output_file}")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='View enriched chunk metadata')
    parser.add_argument('--dir', default='data/enriched', help='Directory with enriched chunks')
    parser.add_argument('--stats', action='store_true', help='Show overall statistics')
    parser.add_argument('--chunk', help='Show details for specific chunk ID')
    parser.add_argument('--index', type=int, help='Show chunk by index')
    parser.add_argument('--search-keyword', help='Search chunks by keyword')
    parser.add_argument('--search-entity', help='Search chunks by entity')
    parser.add_argument('--search-topic', help='Search chunks by topic')
    parser.add_argument('--list-topics', action='store_true', help='List all topics')
    parser.add_argument('--list-entities', help='List entities (optionally filter by type: PERSON, ORG, GPE, etc.)')
    parser.add_argument('--compare', nargs=2, metavar=('CHUNK1', 'CHUNK2'), help='Compare two chunks')
    parser.add_argument('--export', help='Export metadata summary to file')
    parser.add_argument('--max-results', type=int, default=10, help='Maximum search results to show')
    
    args = parser.parse_args()
    
    # Initialize viewer
    viewer = EnrichmentViewer(enriched_dir=args.dir)
    
    if not viewer.chunks:
        print("No enriched chunks found!")
        return
    
    # Execute requested action
    if args.stats:
        viewer.show_statistics()
    
    elif args.chunk:
        viewer.show_chunk_detail(chunk_id=args.chunk)
    
    elif args.index is not None:
        viewer.show_chunk_detail(index=args.index)
    
    elif args.search_keyword:
        viewer.search_by_keyword(args.search_keyword, max_results=args.max_results)
    
    elif args.search_entity:
        viewer.search_by_entity(args.search_entity, max_results=args.max_results)
    
    elif args.search_topic:
        viewer.search_by_topic(args.search_topic, max_results=args.max_results)
    
    elif args.list_topics:
        viewer.list_all_topics()
    
    elif args.list_entities is not None:
        entity_type = args.list_entities if args.list_entities else None
        viewer.list_all_entities(entity_type=entity_type)
    
    elif args.compare:
        viewer.compare_chunks(args.compare[0], args.compare[1])
    
    elif args.export:
        viewer.export_metadata_summary(output_file=args.export)
    
    else:
        # Default: show statistics
        print("No action specified. Showing statistics...\n")
        viewer.show_statistics()
        print("\nUse --help to see all available options")


if __name__ == "__main__":
    main()