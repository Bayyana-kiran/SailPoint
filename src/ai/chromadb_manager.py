"""
ChromaDB Manager for Chatbot Context and Semantic Search
Handles local vector database setup, schema, and semantic retrieval for chatbot queries.
"""
import os
from .chromadb_imports import CHROMADB_AVAILABLE, CHROMADB_ERROR

class ChromaDBManager:
    def __init__(self, host: str = None, port: int = None, persist_dir: str = "./chromadb_data"):
        # Get configuration from environment variables with defaults
        host = host or os.getenv('CHROMA_HOST', 'localhost')
        port = port or int(os.getenv('CHROMA_PORT', 8000))
        
        # Try to connect to ChromaDB server first, fallback to local if server not available
        self.client = None
        self.collection = None
        self.embedding_fn = None
        
        if not CHROMADB_AVAILABLE:
            print(f"❌ ChromaDB not available: {CHROMADB_ERROR}")
            print("💡 To fix this, you can:")
            print("   1. Start a ChromaDB server: chroma run --host 0.0.0.0 --port 8000")
            print("   2. Or upgrade Python's sqlite3 (requires Python rebuild)")
            print("🚫 ChromaDB functionality will be disabled")
            return
        
        # Import ChromaDB modules only if available
        from .chromadb_imports import chromadb, Settings, embedding_functions
        
        try:
            # Try server connection first
            try:
                self.client = chromadb.HttpClient(host=host, port=port)
                print(f"✅ Connected to ChromaDB server at {host}:{port}")
            except Exception as server_error:
                print(f"⚠️  Could not connect to ChromaDB server at {host}:{port}: {server_error}")
                print("🔄 Falling back to local ChromaDB instance")
                
                # Try local ChromaDB
                try:
                    self.client = chromadb.Client(Settings(persist_directory=persist_dir))
                    print(f"✅ Using local ChromaDB with persistence directory: {persist_dir}")
                except Exception as local_error:
                    print(f"❌ Could not initialize local ChromaDB: {local_error}")
                    print("🚫 ChromaDB functionality will be disabled")
                    return
            
            self.collection_name = "chatbot_context"
            self.collection = self._get_or_create_collection()
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
            print("✅ ChromaDB initialized successfully")
            
        except Exception as e:
            print(f"❌ ChromaDB initialization failed: {e}")
            print("🚫 ChromaDB functionality will be disabled")

    def _get_or_create_collection(self):
        if not self.client:
            return None
        if self.collection_name in self.client.list_collections():
            return self.client.get_collection(self.collection_name)
        return self.client.create_collection(self.collection_name)

    def add_query(self, query: str, metadata=None):
        if not self.client or not self.collection or not self.embedding_fn:
            print("ChromaDB not available, skipping query storage")
            return
        
        try:
            embedding = self.embedding_fn(query)
            doc_id = f"query_{hash(query)}"
            self.collection.add(
                documents=[query],
                embeddings=[embedding],
                ids=[doc_id],
                metadatas=[metadata or {}]
            )
        except Exception as e:
            print(f"Failed to add query to ChromaDB: {e}")

    def search_similar(self, query: str, top_k: int = 5):
        if not self.client or not self.collection or not self.embedding_fn:
            print("ChromaDB not available, returning empty results")
            return []
        
        try:
            embedding = self.embedding_fn(query)
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k
            )
            return [
                {
                    "query": doc,
                    "score": score,
                    "metadata": meta
                }
                for doc, score, meta in zip(results["documents"], results["distances"], results["metadatas"])
            ]
        except Exception as e:
            print(f"Failed to search ChromaDB: {e}")
            return []

    def clear(self):
        if not self.client or not self.collection:
            print("ChromaDB not available, cannot clear")
            return
        
        try:
            self.collection.delete()
            self.collection = self._get_or_create_collection()
        except Exception as e:
            print(f"Failed to clear ChromaDB: {e}")

# Usage example (for integration):
# chroma_manager = ChromaDBManager()
# chroma_manager.add_query("how many users are there?", {"intent": "count_records"})
# similar = chroma_manager.search_similar("number of identities")
# print(similar)
