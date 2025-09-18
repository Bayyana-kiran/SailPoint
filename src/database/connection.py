"""
Database Connection Manager
Handles MySQL database connections with connection pooling and error handling.
"""

import logging
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Tuple
import pymysql
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError
import structlog

from config.database import DATABASE_CONFIG, POOL_CONFIG, get_database_url, validate_config

logger = structlog.get_logger()

class DatabaseConnection:
    """
    Production-ready database connection manager with connection pooling,
    error handling, and performance monitoring.
    """
    
    def __init__(self):
        self.engine: Optional[Engine] = None
        self.metadata: Optional[MetaData] = None
        self._connection_pool = None
        self._last_health_check = 0
        self._health_check_interval = 300  # 5 minutes
        
    def initialize(self) -> bool:
        """Initialize database connection and connection pool."""
        try:
            # Create SQLAlchemy engine with connection pooling
            self.engine = create_engine(
                get_database_url(),
                poolclass=QueuePool,
                pool_size=POOL_CONFIG['pool_size'],
                max_overflow=POOL_CONFIG['max_overflow'],
                pool_timeout=POOL_CONFIG['pool_timeout'],
                pool_recycle=POOL_CONFIG['pool_recycle'],
                pool_pre_ping=POOL_CONFIG['pool_pre_ping'],
                pool_reset_on_return='commit',
                echo=False  # Set to True for SQL query logging
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                
            logger.info("Database connection initialized successfully")
            return True
            
        except Exception as e:
            logger.error("Failed to initialize database connection", error=str(e))
            raise ConnectionError(f"Database initialization failed: {str(e)}")
    
    @contextmanager
    def get_connection(self):
        """
        Get a database connection from the pool.
        Use as context manager to ensure proper cleanup.
        """
        if not self.engine:
            self.initialize()
            
        connection = self.engine.connect()
        try:
            yield connection
        except Exception as e:
            logger.error("Error with database connection", error=str(e))
            raise
        finally:
            if connection:
                connection.close()
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> Tuple[List[Dict], List[str]]:
        """
        Execute a SELECT query safely with proper error handling.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Tuple of (results, column_names)
        """
        start_time = time.time()
        
        try:
            with self.get_connection() as conn:
                result = conn.execute(text(query), params or {})
                
                # Fetch results
                rows = result.fetchall()
                columns = list(result.keys()) if rows else []
                
                # Convert to list of dictionaries
                data = [dict(row._mapping) for row in rows]
                
                execution_time = time.time() - start_time
                
                logger.info(
                    "Query executed successfully",
                    query_length=len(query),
                    result_count=len(data),
                    execution_time=execution_time
                )
                
                return data, columns
                
        except SQLAlchemyError as e:
            execution_time = time.time() - start_time
            logger.error(
                "SQL execution error",
                error=str(e),
                query=query[:200],  # Log first 200 chars
                execution_time=execution_time
            )
            raise
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                "Unexpected error during query execution",
                error=str(e),
                query=query[:200],
                execution_time=execution_time
            )
            raise
    
    def test_connection(self) -> bool:
        """Test database connection health."""
        try:
            with self.get_connection() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error("Connection health check failed", error=str(e))
            return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection pool information."""
        if not self.engine:
            return {"status": "not_initialized"}
            
        pool = self.engine.pool
        return {
            "status": "connected",
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid()
        }
    
    def close(self):
        """Close all database connections."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")

# Global database connection instance
db_connection = DatabaseConnection()
