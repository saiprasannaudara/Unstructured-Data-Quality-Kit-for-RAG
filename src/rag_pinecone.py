"""
RAG System with Pinecone and LLM Integration
"""

import logging
from typing import List, Dict
import sys

from sentence_transformers import SentenceTransformer
import pinecone
import openai
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


class PineconeRAG:
    """RAG system using Pinecone and OpenAI."""
    
    def __init__(self,
                 pinecone_api_key: str,
                 pinecone_environment: str,
                 index_name: str,
                 openai_api_key: str = None,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        # Initialize Pinecone
        logger.info("Initializing Pinecone...")
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
        self.index = pinecone.Index(index_name)
        logger.info(f"[SUCCESS] Connected to Pinecone index: {index_name}")
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        logger.info("[SUCCESS] Embedding model loaded")
        
        # Setup OpenAI
        openai.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            logger.warning("No OpenAI API key provided")
        else:
            logger.info("[SUCCESS] OpenAI initialized")
    
    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict]:
        """Retrieve relevant chunks from Pinecone."""
        logger.info(f"Retrieving for query: '{query[:50]}...'")
        
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        
        # Filter by score
        chunks = []
        for match in results['matches']:
            if match['score'] >= min_score:
                chunks.append({
                    'text': match['metadata'].get('text', ''),
                    'source': match['metadata'].get('source_file', 'Unknown'),
                    'quality': match['metadata'].get('quality_score', 0),
                    'topic': match['metadata'].get('primary_topic', ''),
                    'similarity': match['score']
                })
        
        logger.info(f"Retrieved {len(chunks)} chunks")
        return chunks
    
    def generate_answer(self, query: str, chunks: List[Dict], model: str = "gpt-4") -> Dict:
        """Generate answer using OpenAI."""
        
        # Build context from chunks
        context = "\n\n".join([
            f"[Source: {chunk['source']} | Quality: {chunk['quality']:.2f}]\n{chunk['text']}"
            for chunk in chunks
        ])
        
        # Create prompt
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context.
Always cite your sources by mentioning the source file name.
If the context doesn't contain relevant information, say so clearly."""
        
        user_prompt = f"""Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above. Cite your sources."""
        
        # Call OpenAI
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            logger.info(f"[SUCCESS] Answer generated ({tokens_used} tokens)")
            
            return {
                'answer': answer,
                'model': model,
                'tokens_used': tokens_used
            }
        
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                'answer': f"Error: {str(e)}",
                'model': model
            }
    
    def query(self, question: str, top_k: int = 5, min_score: float = 0.7) -> Dict:
        """Complete RAG pipeline: retrieve + generate."""
        logger.info("\n" + "="*70)
        logger.info(f"QUESTION: {question}")
        logger.info("="*70)
        
        # Retrieve
        chunks = self.retrieve(question, top_k=top_k, min_score=min_score)
        
        if not chunks:
            return {
                'question': question,
                'answer': "I couldn't find relevant information in the database to answer your question.",
                'sources': []
            }
        
        # Generate
        result = self.generate_answer(question, chunks)
        
        # Format response
        response = {
            'question': question,
            'answer': result['answer'],
            'model': result.get('model'),
            'tokens_used': result.get('tokens_used'),
            'sources': chunks
        }
        
        return response
    
    def print_response(self, response: Dict):
        """Pretty print response."""
        print("\n" + "="*70)
        print("QUESTION:")
        print("="*70)
        print(response['question'])
        
        print("\n" + "="*70)
        print("ANSWER:")
        print("="*70)
        print(response['answer'])
        
        if response.get('tokens_used'):
            print(f"\n[Model: {response.get('model')} | Tokens: {response['tokens_used']}]")
        
        print("\n" + "="*70)
        print(f"SOURCES ({len(response['sources'])} chunks):")
        print("="*70)
        
        for i, source in enumerate(response['sources'], 1):
            print(f"\n[{i}] {source['source']}")
            print(f"    Quality: {source['quality']:.2f} | Similarity: {source['similarity']:.3f}")
            print(f"    Topic: {source['topic']}")
            print(f"    Preview: {source['text'][:150]}...")
        
        print("\n" + "="*70 + "\n")
    
    def interactive(self):
        """Interactive query mode."""
        print("\n" + "="*70)
        print("RAG SYSTEM - Interactive Mode")
        print("="*70)
        print("\nCommands:")
        print("  - Type your question")
        print("  - 'topk=N' to change number of results")
        print("  - 'score=N' to set minimum similarity score")
        print("  - 'quit' to exit")
        print("="*70 + "\n")
        
        top_k = 5
        min_score = 0.7
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                # Parse commands
                if user_input.startswith('topk='):
                    top_k = int(user_input.split('=')[1])
                    print(f"✓ Top-k set to: {top_k}")
                    continue
                
                if user_input.startswith('score='):
                    min_score = float(user_input.split('=')[1])
                    print(f"✓ Min score set to: {min_score}")
                    continue
                
                # Process question
                print("\n🤖 Thinking...")
                response = self.query(user_input, top_k=top_k, min_score=min_score)
                
                print("\n🤖 Assistant:")
                print("-" * 70)
                print(response['answer'])
                print("-" * 70)
                
                # Show sources
                print(f"\n📚 Sources: {len(response['sources'])} chunks used")
                for source in response['sources'][:3]:
                    print(f"  • {source['source']} (Similarity: {source['similarity']:.3f})")
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG System with Pinecone')
    parser.add_argument('--pinecone-api-key', required=True, help='Pinecone API key')
    parser.add_argument('--pinecone-env', required=True, help='Pinecone environment')
    parser.add_argument('--index-name', default='rag-chunks', help='Index name')
    parser.add_argument('--openai-api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--query', help='Direct query (non-interactive)')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    parser.add_argument('--min-score', type=float, default=0.7, help='Minimum similarity score')
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG
        rag = PineconeRAG(
            pinecone_api_key=args.pinecone_api_key,
            pinecone_environment=args.pinecone_env,
            index_name=args.index_name,
            openai_api_key=args.openai_api_key
        )
        
        # Run query or interactive
        if args.query:
            response = rag.query(args.query, top_k=args.top_k, min_score=args.min_score)
            rag.print_response(response)
        else:
            rag.interactive()
    
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()