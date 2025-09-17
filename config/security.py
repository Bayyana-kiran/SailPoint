"""
Security Configuration
Defines security policies and restrictions for the database chatbot.
"""

import os
from typing import List, Dict, Any

# Security Configuration
SECURITY_CONFIG = {
    # Rate Limiting
    'max_queries_per_minute': int(os.getenv('MAX_QUERIES_PER_MINUTE', 60)),
    'max_queries_per_hour': int(os.getenv('MAX_QUERIES_PER_HOUR', 1000)),
    'max_queries_per_day': int(os.getenv('MAX_QUERIES_PER_DAY', 10000)),
    
    # Query Restrictions
    'query_timeout': int(os.getenv('QUERY_TIMEOUT', 30)),
    'max_result_rows': int(os.getenv('MAX_RESULT_ROWS', 1000)),
    'max_query_length': int(os.getenv('MAX_QUERY_LENGTH', 5000)),
    
    # Allowed Operations (READ-ONLY by default)
    'allowed_operations': [
        'SELECT', 'SHOW', 'DESCRIBE', 'EXPLAIN', 'WITH'
    ],
    
    # Blocked Keywords (Security)
    'blocked_keywords': [
        'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE',
        'TRUNCATE', 'REPLACE', 'LOAD', 'OUTFILE', 'INFILE',
        'GRANT', 'REVOKE', 'FLUSH', 'RESET', 'SHUTDOWN',
        'CALL', 'EXECUTE', 'PREPARE', 'DEALLOCATE'
    ],
    
    # Sensitive Keywords (Log and Monitor)
    'sensitive_keywords': [
        'PASSWORD', 'USER', 'MYSQL', 'INFORMATION_SCHEMA',
        'PERFORMANCE_SCHEMA', 'SYS', 'mysql'
    ],
    
    # Table Access Control
    'restricted_tables': [
        'mysql.user', 'mysql.db', 'mysql.tables_priv',
        'information_schema.user_privileges',
        'performance_schema.users'
    ],
    
    # Authentication
    'require_authentication': os.getenv('REQUIRE_AUTH', 'true').lower() == 'true',
    'session_timeout': int(os.getenv('SESSION_TIMEOUT', 3600)),  # 1 hour
    
    # Logging
    'log_all_queries': os.getenv('LOG_ALL_QUERIES', 'true').lower() == 'true',
    'log_failed_queries': True,
    'log_sensitive_access': True,
    
    # Input Validation
    'max_input_length': 10000,
    'sanitize_input': True,
    'validate_sql_syntax': True,
}

# Dangerous Patterns (Regex patterns to detect potential attacks)
DANGEROUS_PATTERNS = [
    r';\s*(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE)',  # Multiple statements
    r'UNION\s+.*SELECT',  # Union-based injection
    r'OR\s+1\s*=\s*1',    # Always true conditions
    r'--\s*$',            # SQL comments
    r'/\*.*\*/',          # Multi-line comments
    r'0x[0-9a-fA-F]+',    # Hexadecimal values
    r'LOAD_FILE\s*\(',    # File operations
    r'INTO\s+OUTFILE',    # File writing
    r'BENCHMARK\s*\(',    # Performance attacks
    r'SLEEP\s*\(',        # Time-based attacks
    r'WAITFOR\s+DELAY',   # Time delays
]

# Content Security Policy
CSP_RULES = {
    'default-src': "'self'",
    'script-src': "'self' 'unsafe-inline'",
    'style-src': "'self' 'unsafe-inline'",
    'img-src': "'self' data:",
    'connect-src': "'self'",
    'font-src': "'self'",
    'object-src': "'none'",
    'media-src': "'self'",
    'frame-src': "'none'",
}

def get_security_headers() -> Dict[str, str]:
    """Generate security headers for HTTP responses."""
    return {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': '; '.join([f'{k} {v}' for k, v in CSP_RULES.items()]),
        'Referrer-Policy': 'strict-origin-when-cross-origin',
    }
