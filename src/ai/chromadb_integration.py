"""
High-level integration of ChromaDB with chatbot pipeline.
Handles semantic context retrieval and query enrichment for SQL generation.
"""
from src.ai.chromadb_manager import ChromaDBManager
from typing import Dict, Any, Optional

class ChatbotContextManager:
    def __init__(self, host: str = None, port: int = None):
        import os
        host = host or os.getenv('CHROMA_HOST', 'localhost')
        port = port or int(os.getenv('CHROMA_PORT', 8000))
        self.chroma = ChromaDBManager(host=host, port=port)

    def enrich_query_with_context(self, user_query: str, metadata=None):
        if not self.chroma or not self.chroma.collection:
            # Return basic context without ChromaDB enrichment
            return {
                "user_query": user_query,
                "similar_queries": [],
                "similar_metadata": [],
                "raw_results": [],
                "chromadb_available": False
            }
        
        try:
            # Store the query and metadata
            self.chroma.add_query(user_query, metadata)
            # Retrieve similar queries for semantic context
            similar = self.chroma.search_similar(user_query, top_k=3)
            # Build context payload for SQL engine
            context = {
                "user_query": user_query,
                "similar_queries": [item["query"] for item in similar],
                "similar_metadata": [item["metadata"] for item in similar],
                "raw_results": similar,
                "chromadb_available": True
            }
            return context
        except Exception as e:
            print(f"ChromaDB enrichment failed: {e}")
            return {
                "user_query": user_query,
                "similar_queries": [],
                "similar_metadata": [],
                "raw_results": [],
                "chromadb_available": False,
                "error": str(e)
            }

    def clear_context(self):
        if self.chroma:
            self.chroma.clear()

# Usage example:
# context_manager = ChatbotContextManager()
# context = context_manager.enrich_query_with_context("who are the top users of applications?", {"intent": "top_users"})
# print(context)
