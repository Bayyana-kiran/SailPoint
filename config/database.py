"""
Database Configuration
Manages database connection settings and environment variables.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database Configuration
DATABASE_CONFIG = {
    'host': os.getenv('MYSQL_HOST', 'localhost'),
    'port': int(os.getenv('MYSQL_PORT', 3306)),
    'database': os.getenv('MYSQL_DATABASE', ''),
    'user': os.getenv('MYSQL_USER', ''),
    'password': os.getenv('MYSQL_PASSWORD', ''),
    'charset': 'utf8mb4',
    'autocommit': True,
    'connect_timeout': 30,
    'read_timeout': 30,
    'write_timeout': 30,
}

# Connection Pool Configuration
POOL_CONFIG = {
    'pool_name': 'chatbot_pool',
    'pool_size': 10,
    'pool_reset_session': True,
    'pool_pre_ping': True,
    'max_overflow': 20,
    'pool_timeout': 30,
    'pool_recycle': 3600,  # 1 hour
}

# SQLAlchemy Database URL
def get_database_url() -> str:
    """Generate SQLAlchemy database URL from configuration."""
    return (
        f"mysql+pymysql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}"
        f"@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}"
        f"/{DATABASE_CONFIG['database']}?charset={DATABASE_CONFIG['charset']}"
    )

# Validation
def validate_config() -> bool:
    """Validate database configuration."""
    required_fields = ['host', 'database', 'user']
    
    for field in required_fields:
        if not DATABASE_CONFIG.get(field):
            raise ValueError(f"Missing required database configuration: {field}")
    
    # Password can be empty for local development
    if DATABASE_CONFIG.get('password') is None:
        raise ValueError("Missing required database configuration: password")
    
    return True
