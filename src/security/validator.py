"""
SQL Query Validator
Provides comprehensive validation and security checks for SQL queries.
"""

import re
import time
import logging
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from config.security import SECURITY_CONFIG, DANGEROUS_PATTERNS

logger = logging.getLogger(__name__)

class ValidationResult(Enum):
    """Query validation results."""
    VALID = "valid"
    BLOCKED = "blocked"
    WARNING = "warning"
    ERROR = "error"

@dataclass
class ValidationResponse:
    """Response from query validation."""
    result: ValidationResult
    message: str
    details: Optional[Dict[str, Any]] = None
    sanitized_query: Optional[str] = None

class SQLValidator:
    """
    Comprehensive SQL query validator with security checks and sanitization.
    """
    
    def __init__(self):
        self.allowed_operations = set(op.upper() for op in SECURITY_CONFIG['allowed_operations'])
        self.blocked_keywords = set(kw.upper() for kw in SECURITY_CONFIG['blocked_keywords'])
        self.sensitive_keywords = set(kw.upper() for kw in SECURITY_CONFIG['sensitive_keywords'])
        self.restricted_tables = set(SECURITY_CONFIG['restricted_tables'])
        self.dangerous_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in DANGEROUS_PATTERNS]
        self.schema = self._load_schema()
        
    def _load_schema(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load database schema from complete_schema.json."""
        schema_path = os.path.join(os.path.dirname(__file__), '..', '..', 'complete_schema.json')
        try:
            with open(schema_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load schema: {str(e)}")
            return {}
        
    def validate_query(self, query: str, user_context: Optional[Dict] = None) -> ValidationResponse:
        """
        Comprehensive query validation with security checks.
        
        Args:
            query: SQL query to validate
            user_context: Optional user context for additional checks
            
        Returns:
            ValidationResponse with validation result
        """
        start_time = time.time()
        
        try:
            # Basic input validation
            basic_check = self._basic_validation(query)
            if basic_check.result != ValidationResult.VALID:
                return basic_check
            
            # Normalize query
            normalized_query = self._normalize_query(query)
            
            # Check allowed operations
            operation_check = self._check_allowed_operations(normalized_query)
            if operation_check.result != ValidationResult.VALID:
                return operation_check
            
            # Check for blocked keywords
            keyword_check = self._check_blocked_keywords(normalized_query)
            if keyword_check.result != ValidationResult.VALID:
                return keyword_check
            
            # Check for dangerous patterns
            pattern_check = self._check_dangerous_patterns(normalized_query)
            if pattern_check.result != ValidationResult.VALID:
                return pattern_check
            
            # Check table access
            table_check = self._check_table_access(normalized_query)
            if table_check.result != ValidationResult.VALID:
                return table_check
            
            # Check for sensitive data access
            sensitive_check = self._check_sensitive_access(normalized_query)
            
            # Validate against schema
            schema_check = self._validate_against_schema(normalized_query)
            if schema_check.result != ValidationResult.VALID:
                return schema_check
            
            # Sanitize query
            sanitized_query = self._sanitize_query(normalized_query)
            
            validation_time = time.time() - start_time
            
            logger.info(
                f"Query validation completed - Result: {sensitive_check.result.value}, "
                f"Time: {validation_time:.3f}s, Query length: {len(query)}"
            )
            
            return ValidationResponse(
                result=sensitive_check.result,
                message=sensitive_check.message,
                details={
                    "validation_time": validation_time,
                    "query_length": len(query),
                    "normalized_length": len(normalized_query)
                },
                sanitized_query=sanitized_query
            )
            
        except Exception as e:
            logger.error(f"Query validation error: {str(e)}")
            return ValidationResponse(
                result=ValidationResult.ERROR,
                message=f"Validation error: {str(e)}"
            )
    
    def _basic_validation(self, query: str) -> ValidationResponse:
        """Basic input validation checks."""
        if not query or not query.strip():
            return ValidationResponse(
                result=ValidationResult.BLOCKED,
                message="Empty query not allowed"
            )
        
        if len(query) > SECURITY_CONFIG['max_query_length']:
            return ValidationResponse(
                result=ValidationResult.BLOCKED,
                message=f"Query too long (max {SECURITY_CONFIG['max_query_length']} characters)"
            )
        
        # Check for null bytes and control characters
        if '\x00' in query or any(ord(c) < 32 and c not in '\t\n\r ' for c in query):
            return ValidationResponse(
                result=ValidationResult.BLOCKED,
                message="Invalid characters detected in query"
            )
        
        return ValidationResponse(
            result=ValidationResult.VALID,
            message="Basic validation passed"
        )
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent analysis."""
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', query.strip())
        
        # Remove comments
        normalized = re.sub(r'--.*$', '', normalized, flags=re.MULTILINE)
        normalized = re.sub(r'/\*.*?\*/', '', normalized, flags=re.DOTALL)
        
        return normalized
    
    def _check_allowed_operations(self, query: str) -> ValidationResponse:
        """Check if query uses only allowed operations."""
        query_upper = query.upper().strip()
        
        # Extract the main operation
        first_word = query_upper.split()[0] if query_upper.split() else ""
        
        if first_word not in self.allowed_operations:
            return ValidationResponse(
                result=ValidationResult.BLOCKED,
                message=f"Operation '{first_word}' not allowed. Allowed operations: {', '.join(self.allowed_operations)}"
            )
        
        # Check for multiple statements (semicolon followed by another statement)
        statements = [s.strip() for s in query.split(';') if s.strip()]
        if len(statements) > 1:
            return ValidationResponse(
                result=ValidationResult.BLOCKED,
                message="Multiple statements not allowed"
            )
        
        return ValidationResponse(
            result=ValidationResult.VALID,
            message="Operation check passed"
        )
    
    def _check_blocked_keywords(self, query: str) -> ValidationResponse:
        """Check for blocked keywords in query."""
        query_upper = query.upper()
        
        for keyword in self.blocked_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', query_upper):
                return ValidationResponse(
                    result=ValidationResult.BLOCKED,
                    message=f"Blocked keyword detected: {keyword}"
                )
        
        return ValidationResponse(
            result=ValidationResult.VALID,
            message="Keyword check passed"
        )
    
    def _check_dangerous_patterns(self, query: str) -> ValidationResponse:
        """Check for dangerous SQL injection patterns."""
        for pattern in self.dangerous_patterns:
            if pattern.search(query):
                return ValidationResponse(
                    result=ValidationResult.BLOCKED,
                    message=f"Dangerous pattern detected: {pattern.pattern}"
                )
        
        return ValidationResponse(
            result=ValidationResult.VALID,
            message="Pattern check passed"
        )
    
    def _check_table_access(self, query: str) -> ValidationResponse:
        """Check for access to restricted tables."""
        query_upper = query.upper()
        
        for restricted_table in self.restricted_tables:
            if restricted_table.upper() in query_upper:
                return ValidationResponse(
                    result=ValidationResult.BLOCKED,
                    message=f"Access to restricted table: {restricted_table}"
                )
        
        return ValidationResponse(
            result=ValidationResult.VALID,
            message="Table access check passed"
        )
    
    def _check_sensitive_access(self, query: str) -> ValidationResponse:
        """Check for access to sensitive data."""
        query_upper = query.upper()
        detected_sensitive = []
        
        for keyword in self.sensitive_keywords:
            if keyword in query_upper:
                detected_sensitive.append(keyword)
        
        if detected_sensitive:
            return ValidationResponse(
                result=ValidationResult.WARNING,
                message=f"Sensitive data access detected: {', '.join(detected_sensitive)}",
                details={"sensitive_keywords": detected_sensitive}
            )
        
        return ValidationResponse(
            result=ValidationResult.VALID,
            message="Sensitive data check passed"
        )
    
    def _validate_against_schema(self, query: str) -> ValidationResponse:
        """Validate query against database schema for column and table references."""
        if not self.schema:
            return ValidationResponse(
                result=ValidationResult.VALID,
                message="Schema not available, skipping validation"
            )
        
        try:
            # Extract table aliases and their corresponding tables
            table_aliases = self._extract_table_aliases(query)
            
            # Extract column references
            column_refs = self._extract_column_references(query)
            
            invalid_refs = []
            
            for alias, column in column_refs:
                if alias:
                    # Check if alias exists and column is valid for that table
                    if alias not in table_aliases:
                        invalid_refs.append(f"{alias}.{column} (unknown alias '{alias}')")
                    else:
                        table_name = table_aliases[alias]
                        if not self._is_valid_column(table_name, column):
                            invalid_refs.append(f"{alias}.{column} (invalid column for {table_name})")
                else:
                    # No alias, check if column exists in any joined table
                    found = False
                    for table in table_aliases.values():
                        if self._is_valid_column(table, column):
                            found = True
                            break
                    if not found:
                        invalid_refs.append(f"{column} (column not found in any table)")
            
            if invalid_refs:
                return ValidationResponse(
                    result=ValidationResult.ERROR,
                    message=f"Schema validation failed: {', '.join(invalid_refs)}"
                )
            
            return ValidationResponse(
                result=ValidationResult.VALID,
                message="Schema validation passed"
            )
            
        except Exception as e:
            logger.error(f"Schema validation error: {str(e)}")
            return ValidationResponse(
                result=ValidationResult.ERROR,
                message=f"Schema validation error: {str(e)}"
            )
    
    def _extract_table_aliases(self, query: str) -> Dict[str, str]:
        """Extract table aliases and their corresponding table names."""
        aliases = {}
        
        # Normalize query for parsing
        normalized = re.sub(r'\s+', ' ', query.upper().strip())
        
        # Pattern to match FROM and JOIN clauses with optional aliases
        patterns = [
            r'FROM\s+(\w+)\s+(?:AS\s+)?(\w+)',
            r'FROM\s+(\w+)',
            r'JOIN\s+(\w+)\s+(?:AS\s+)?(\w+)',
            r'JOIN\s+(\w+)',
            # Handle comma-separated tables in FROM clause
            r'FROM\s+(.+?)\s+WHERE',
            r'FROM\s+(.+?)(?:\s+WHERE|\s+GROUP|\s+ORDER|\s+HAVING|\s+LIMIT|$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, normalized, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    table, alias = match
                    # Clean up table and alias
                    table = table.strip()
                    alias = alias.strip()
                    if alias and alias != table:
                        aliases[alias.lower()] = table.lower()
                    else:
                        aliases[table.lower()] = table.lower()
                elif len(match) == 1:
                    # Handle comma-separated tables
                    from_part = match[0]
                    # Split by comma and extract table alias pairs
                    tables = [t.strip() for t in from_part.split(',')]
                    for table_spec in tables:
                        parts = table_spec.split()
                        if len(parts) >= 2:
                            table = parts[0]
                            alias = parts[-1]  # Last part is usually the alias
                            if alias != table:
                                aliases[alias.lower()] = table.lower()
                            else:
                                aliases[table.lower()] = table.lower()
                        elif len(parts) == 1:
                            table = parts[0]
                            aliases[table.lower()] = table.lower()
        
        return aliases
    
    def _extract_column_references(self, query: str) -> List[Tuple[str, str]]:
        """Extract column references with their aliases."""
        refs = []
        
        # Pattern to match column references like alias.column or just column
        pattern = r'(\w+)\.(\w+)'
        matches = re.findall(pattern, query)
        
        for match in matches:
            alias, column = match
            refs.append((alias, column))
        
        return refs
    
    def _is_valid_column(self, table_name: str, column_name: str) -> bool:
        """Check if column exists in the table schema."""
        if table_name not in self.schema:
            return False
        
        for col in self.schema[table_name]:
            if col['name'] == column_name:
                return True
        
        return False
    
    def _sanitize_query(self, query: str) -> str:
        """Sanitize query for safe execution."""
        if not SECURITY_CONFIG['sanitize_input']:
            return query
        
        sanitized = query
        
        # Remove any remaining comments
        sanitized = re.sub(r'--.*$', '', sanitized, flags=re.MULTILINE)
        sanitized = re.sub(r'/\*.*?\*/', '', sanitized, flags=re.DOTALL)
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized.strip())
        
        # Add LIMIT if not present for SELECT queries
        if (sanitized.upper().startswith('SELECT') and 
            'LIMIT' not in sanitized.upper() and 
            SECURITY_CONFIG['max_result_rows']):
            sanitized += f" LIMIT {SECURITY_CONFIG['max_result_rows']}"
        
        return sanitized
    
    def extract_table_names(self, query: str) -> List[str]:
        """Extract table names from query."""
        # Simple regex to extract table names - could be enhanced
        table_pattern = r'\bFROM\s+`?(\w+)`?|\bJOIN\s+`?(\w+)`?'
        matches = re.findall(table_pattern, query, re.IGNORECASE)
        
        tables = []
        for match in matches:
            table_name = match[0] or match[1]  # Get non-empty group
            if table_name:
                tables.append(table_name)
        
        return list(set(tables))  # Remove duplicates
    
    def validate_syntax(self, query: str) -> ValidationResponse:
        """Basic SQL syntax validation."""
        if not SECURITY_CONFIG['validate_sql_syntax']:
            return ValidationResponse(
                result=ValidationResult.VALID,
                message="Syntax validation disabled"
            )
        
        # Basic syntax checks
        query_upper = query.upper().strip()
        
        # Check for balanced parentheses
        if query.count('(') != query.count(')'):
            return ValidationResponse(
                result=ValidationResult.ERROR,
                message="Unbalanced parentheses in query"
            )
        
        # Check for balanced quotes
        single_quotes = query.count("'") - query.count("\\'")
        double_quotes = query.count('"') - query.count('\\"')
        
        if single_quotes % 2 != 0:
            return ValidationResponse(
                result=ValidationResult.ERROR,
                message="Unbalanced single quotes in query"
            )
        
        if double_quotes % 2 != 0:
            return ValidationResponse(
                result=ValidationResult.ERROR,
                message="Unbalanced double quotes in query"
            )
        
        return ValidationResponse(
            result=ValidationResult.VALID,
            message="Basic syntax validation passed"
        )

# Global validator instance
sql_validator = SQLValidator()
