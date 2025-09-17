"""
Database Metadata Analyzer
Analyzes MySQL database schema, relationships, and generates intelligent context for LLM.
"""

import json
import time
import os
import pymysql
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from sqlalchemy import MetaData, Table, inspect, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv
import logging

from src.database.connection import db_connection

logger = logging.getLogger(__name__)
load_dotenv()

@dataclass
class ColumnInfo:
    """Information about a database column."""
    name: str
    type: str
    nullable: bool
    primary_key: bool
    foreign_key: Optional[str] = None
    default: Optional[str] = None
    comment: Optional[str] = None

@dataclass
class TableInfo:
    """Information about a database table."""
    name: str
    columns: List[ColumnInfo]
    primary_keys: List[str]
    foreign_keys: List[Dict[str, str]]
    indexes: List[str]
    row_count: Optional[int] = None
    comment: Optional[str] = None

@dataclass
class DatabaseSchema:
    """Complete database schema information."""
    tables: Dict[str, TableInfo]
    relationships: List[Dict[str, Any]]
    indexes: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    metadata_timestamp: float

class DatabaseMetadataAnalyzer:
    """
    Analyzes database schema and generates intelligent metadata for LLM context.
    """
    
    def __init__(self):
        self.schema_cache: Optional[DatabaseSchema] = None
        self.cache_timestamp: float = 0
        self.cache_ttl: int = 3600  # 1 hour cache TTL
        self.actual_schema_cache: Optional[Dict] = None
        
    def analyze_schema(self, force_refresh: bool = False) -> DatabaseSchema:
        """
        Analyze database schema and return comprehensive metadata.
        
        Args:
            force_refresh: Force refresh of cached schema
            
        Returns:
            DatabaseSchema object with complete schema information
        """
        current_time = time.time()
        
        # Check cache validity
        if (not force_refresh and 
            self.schema_cache and 
            (current_time - self.cache_timestamp) < self.cache_ttl):
            return self.schema_cache
        
        logger.info("Analyzing database schema...")
        
        try:
            # Get database tables
            tables = self._get_tables_info()
            
            # Analyze relationships
            relationships = self._analyze_relationships(tables)
            
            # Get indexes
            indexes = self._get_indexes_info()
            
            # Get constraints
            constraints = self._get_constraints_info()
            
            # Create schema object
            schema = DatabaseSchema(
                tables=tables,
                relationships=relationships,
                indexes=indexes,
                constraints=constraints,
                metadata_timestamp=current_time
            )
            
            # Update cache
            self.schema_cache = schema
            self.cache_timestamp = current_time
            
            logger.info(f"Schema analysis complete. Found {len(tables)} tables")
            return schema
            
        except Exception as e:
            logger.error(f"Schema analysis failed: {str(e)}")
            raise
    
    def _get_tables_info(self) -> Dict[str, TableInfo]:
        """Get detailed information about all tables."""
        tables = {}
        
        try:
            with db_connection.get_connection() as conn:
                # Get all table names
                result = conn.execute(text("""
                    SELECT TABLE_NAME, TABLE_COMMENT, TABLE_ROWS
                    FROM information_schema.TABLES 
                    WHERE TABLE_SCHEMA = DATABASE()
                    AND TABLE_TYPE = 'BASE TABLE'
                """))
                
                table_list = result.fetchall()
                
                for table_row in table_list:
                    table_name = table_row[0]
                    table_comment = table_row[1]
                    table_rows = table_row[2]
                    
                    # Get column information
                    columns = self._get_columns_info(conn, table_name)
                    
                    # Get primary keys
                    primary_keys = self._get_primary_keys(conn, table_name)
                    
                    # Get foreign keys
                    foreign_keys = self._get_foreign_keys(conn, table_name)
                    
                    # Get indexes
                    indexes = self._get_table_indexes(conn, table_name)
                    
                    tables[table_name] = TableInfo(
                        name=table_name,
                        columns=columns,
                        primary_keys=primary_keys,
                        foreign_keys=foreign_keys,
                        indexes=indexes,
                        row_count=table_rows,
                        comment=table_comment
                    )
                    
        except Exception as e:
            logger.error(f"Error getting tables info: {str(e)}")
            raise
            
        return tables
    
    def _get_columns_info(self, conn, table_name: str) -> List[ColumnInfo]:
        """Get detailed column information for a table."""
        columns = []
        
        try:
            result = conn.execute(text("""
                SELECT 
                    COLUMN_NAME,
                    DATA_TYPE,
                    IS_NULLABLE,
                    COLUMN_KEY,
                    COLUMN_DEFAULT,
                    COLUMN_COMMENT,
                    COLUMN_TYPE
                FROM information_schema.COLUMNS 
                WHERE TABLE_SCHEMA = DATABASE() 
                AND TABLE_NAME = :table_name
                ORDER BY ORDINAL_POSITION
            """), {"table_name": table_name})
            
            for row in result:
                columns.append(ColumnInfo(
                    name=row[0],
                    type=row[6],  # COLUMN_TYPE includes size info
                    nullable=row[2] == 'YES',
                    primary_key=row[3] == 'PRI',
                    default=row[4],
                    comment=row[5]
                ))
                
        except Exception as e:
            logger.error(f"Error getting columns for table {table_name}: {str(e)}")
            
        return columns
    
    def _get_primary_keys(self, conn, table_name: str) -> List[str]:
        """Get primary key columns for a table."""
        try:
            result = conn.execute(text("""
                SELECT COLUMN_NAME
                FROM information_schema.KEY_COLUMN_USAGE
                WHERE TABLE_SCHEMA = DATABASE()
                AND TABLE_NAME = :table_name
                AND CONSTRAINT_NAME = 'PRIMARY'
                ORDER BY ORDINAL_POSITION
            """), {"table_name": table_name})
            
            return [row[0] for row in result]
            
        except Exception as e:
            logger.error(f"Error getting primary keys for {table_name}: {str(e)}")
            return []
    
    def _get_foreign_keys(self, conn, table_name: str) -> List[Dict[str, str]]:
        """Get foreign key relationships for a table."""
        try:
            result = conn.execute(text("""
                SELECT 
                    COLUMN_NAME,
                    REFERENCED_TABLE_NAME,
                    REFERENCED_COLUMN_NAME,
                    CONSTRAINT_NAME
                FROM information_schema.KEY_COLUMN_USAGE
                WHERE TABLE_SCHEMA = DATABASE()
                AND TABLE_NAME = :table_name
                AND REFERENCED_TABLE_NAME IS NOT NULL
            """), {"table_name": table_name})
            
            foreign_keys = []
            for row in result:
                foreign_keys.append({
                    'column': row[0],
                    'referenced_table': row[1],
                    'referenced_column': row[2],
                    'constraint_name': row[3]
                })
                
            return foreign_keys
            
        except Exception as e:
            logger.error(f"Error getting foreign keys for {table_name}: {str(e)}")
            return []
    
    def _get_table_indexes(self, conn, table_name: str) -> List[str]:
        """Get index information for a table."""
        try:
            result = conn.execute(text("""
                SELECT DISTINCT INDEX_NAME
                FROM information_schema.STATISTICS
                WHERE TABLE_SCHEMA = DATABASE()
                AND TABLE_NAME = :table_name
                AND INDEX_NAME != 'PRIMARY'
            """), {"table_name": table_name})
            
            return [row[0] for row in result]
            
        except Exception as e:
            logger.error(f"Error getting indexes for {table_name}: {str(e)}")
            return []
    
    def _analyze_relationships(self, tables: Dict[str, TableInfo]) -> List[Dict[str, Any]]:
        """Analyze relationships between tables."""
        relationships = []
        
        for table_name, table_info in tables.items():
            for fk in table_info.foreign_keys:
                relationships.append({
                    'type': 'foreign_key',
                    'from_table': table_name,
                    'from_column': fk['column'],
                    'to_table': fk['referenced_table'],
                    'to_column': fk['referenced_column'],
                    'constraint_name': fk['constraint_name']
                })
        
        return relationships
    
    def _get_indexes_info(self) -> List[Dict[str, Any]]:
        """Get comprehensive index information."""
        indexes = []
        
        try:
            with db_connection.get_connection() as conn:
                result = conn.execute(text("""
                    SELECT 
                        TABLE_NAME,
                        INDEX_NAME,
                        COLUMN_NAME,
                        NON_UNIQUE,
                        INDEX_TYPE
                    FROM information_schema.STATISTICS
                    WHERE TABLE_SCHEMA = DATABASE()
                    ORDER BY TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX
                """))
                
                for row in result:
                    indexes.append({
                        'table': row[0],
                        'name': row[1],
                        'column': row[2],
                        'unique': row[3] == 0,
                        'type': row[4]
                    })
                    
        except Exception as e:
            logger.error(f"Error getting indexes info: {str(e)}")
            
        return indexes
    
    def _get_constraints_info(self) -> List[Dict[str, Any]]:
        """Get constraint information."""
        constraints = []
        
        try:
            with db_connection.get_connection() as conn:
                result = conn.execute(text("""
                    SELECT 
                        TABLE_NAME,
                        CONSTRAINT_NAME,
                        CONSTRAINT_TYPE
                    FROM information_schema.TABLE_CONSTRAINTS
                    WHERE TABLE_SCHEMA = DATABASE()
                """))
                
                for row in result:
                    constraints.append({
                        'table': row[0],
                        'name': row[1],
                        'type': row[2]
                    })
                    
        except Exception as e:
            logger.error(f"Error getting constraints info: {str(e)}")
            
        return constraints
    
    def get_actual_schema(self) -> Dict[str, List[Dict]]:
        """
        Get actual database schema by directly querying the database.
        Returns table names and their actual columns only.
        """
        if self.actual_schema_cache:
            return self.actual_schema_cache
            
        try:
            # Connect directly to MySQL
            connection = pymysql.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                user=os.getenv('DB_USER', 'root'), 
                password=os.getenv('DB_PASSWORD', ''),
                database=os.getenv('DB_NAME', 'identityiq_sample'),
                charset='utf8mb4'
            )
            
            schema_info = {}
            
            with connection.cursor() as cursor:
                # Get all tables
                cursor.execute('SHOW TABLES')
                tables = [row[0] for row in cursor.fetchall()]
                
                # Get columns for each table
                for table in tables:
                    cursor.execute(f'DESCRIBE {table}')
                    columns = cursor.fetchall()
                    schema_info[table] = []
                    
                    for col in columns:
                        column_info = {
                            'name': col[0],
                            'type': col[1],
                            'null': col[2] == 'YES',
                            'key': col[3],
                            'default': col[4],
                            'extra': col[5]
                        }
                        schema_info[table].append(column_info)
            
            connection.close()
            self.actual_schema_cache = schema_info
            logger.info(f"Loaded actual schema for {len(schema_info)} tables")
            return schema_info
            
        except Exception as e:
            logger.error(f"Error getting actual schema: {str(e)}")
            return {}
    
    def generate_llm_context(self, schema: Optional[DatabaseSchema] = None) -> str:
        """
        Generate comprehensive IdentityIQ-specific context for LLM based on ACTUAL database schema.
        Uses real table and column information directly from database.
        
        Args:
            schema: DatabaseSchema object (not used, kept for compatibility)
            
        Returns:
            Formatted context string for LLM with actual IdentityIQ schema
        """
        # Get actual schema from database
        actual_schema = self.get_actual_schema()
        
        if not actual_schema:
            return "ERROR: Could not load database schema"
        
        context_parts = []
        
        # IdentityIQ System Overview
        context_parts.append("SAILPOINT IDENTITYIQ DATABASE SCHEMA")
        context_parts.append("="*50)
        context_parts.append(f"This is a SailPoint IdentityIQ identity management system database.")
        context_parts.append(f"Total tables: {len(actual_schema)}")
        context_parts.append("")
        
        # ACTUAL DATABASE TABLES AND COLUMNS
        context_parts.append("DATABASE TABLES WITH ACTUAL COLUMNS:")
        context_parts.append("-" * 40)
        
        # Sort tables by importance for better AI understanding
        important_tables = ['spt_identity', 'spt_application', 'spt_audit_event', 'spt_bundle', 'spt_link', 'spt_account']
        other_tables = sorted([t for t in actual_schema.keys() if t not in important_tables])
        ordered_tables = [t for t in important_tables if t in actual_schema] + other_tables
        
        for table_name in ordered_tables:
            if table_name in actual_schema:
                columns = actual_schema[table_name]
                context_parts.append(f"\nTABLE: {table_name}")
                context_parts.append(f"Columns ({len(columns)}):")
                
                for col in columns:
                    key_info = ""
                    if col['key'] == 'PRI':
                        key_info = " [PRIMARY KEY]"
                    elif col['key'] == 'MUL':
                        key_info = " [INDEXED]"
                    elif col['key'] == 'UNI':
                        key_info = " [UNIQUE]"
                    
                    null_info = " [NULL]" if col['null'] else " [NOT NULL]"
                    context_parts.append(f"  - {col['name']} ({col['type']}){key_info}{null_info}")
        
        # SQL QUERY GUIDELINES
        context_parts.append("SQL QUERY GUIDELINES:")
        context_parts.append("- Use EXACT column names as listed above")
        context_parts.append("- Common patterns:")
        context_parts.append("  * Users: SELECT * FROM spt_identity WHERE name LIKE '%username%'")
        context_parts.append("  * Applications: SELECT * FROM spt_application WHERE name LIKE '%app%'")
        context_parts.append("  * Audit events: SELECT * FROM spt_audit_event WHERE action = 'login' ORDER BY created DESC")
        context_parts.append("  * Recent activity: SELECT * FROM spt_audit_event WHERE created >= DATE_SUB(NOW(), INTERVAL 7 DAY)")
        context_parts.append("")
        
        return "\n".join(context_parts)
    
    def get_table_sample_data(self, table_name: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get sample data from a table for context."""
        try:
            query = f"SELECT * FROM `{table_name}` LIMIT {limit}"
            results, columns = db_connection.execute_query(query)
            return results
        except Exception as e:
            logger.error(f"Error getting sample data for {table_name}: {str(e)}")
            return []
    
    def export_schema(self, format: str = 'json') -> str:
        """Export schema information in specified format."""
        schema = self.analyze_schema()
        
        if format == 'json':
            return json.dumps(asdict(schema), indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

# Global metadata analyzer instance
metadata_analyzer = DatabaseMetadataAnalyzer()
