"""
Safe ChromaDB imports with error handling
"""
CHROMADB_AVAILABLE = False
CHROMADB_ERROR = None
chromadb = None
Settings = None
embedding_functions = None

# Try to import ChromaDB safely
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
    CHROMADB_ERROR = None
except RuntimeError as e:
    if "sqlite3" in str(e):
        CHROMADB_AVAILABLE = False
        CHROMADB_ERROR = f"sqlite3 version issue: {e}"
        # Don't re-raise, just mark as unavailable
    else:
        # Re-raise non-sqlite3 RuntimeErrors
        raise
except ImportError as e:
    CHROMADB_AVAILABLE = False
    CHROMADB_ERROR = f"Import error: {e}"
except Exception as e:
    CHROMADB_AVAILABLE = False
    CHROMADB_ERROR = f"Unexpected error: {e}"
    # Don't re-raise unexpected errors, just mark as unavailable
