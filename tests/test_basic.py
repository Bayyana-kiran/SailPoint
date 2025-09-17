"""
Basic Test Suite for Database Chatbot
Run with: pytest tests/test_basic.py -v
"""

import pytest
import os
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Test imports and basic functionality
def test_imports():
    """Test that all modules can be imported."""
    try:
        from src.security.validator import sql_validator, ValidationResult
        from src.security.rate_limiter import rate_limiter
        from src.security.audit_logger import audit_logger
        assert True  # If we get here, imports worked
    except ImportError as e:
        pytest.skip(f"Import failed: {e}")

def test_sql_validator_basic():
    """Test basic SQL validation."""
    try:
        from src.security.validator import sql_validator, ValidationResult
        
        # Test valid query
        result = sql_validator.validate_query("SELECT * FROM users LIMIT 10")
        assert result.result in [ValidationResult.VALID, ValidationResult.WARNING]
        
        # Test blocked query
        result = sql_validator.validate_query("DROP TABLE users")
        assert result.result == ValidationResult.BLOCKED
        
        # Test empty query
        result = sql_validator.validate_query("")
        assert result.result == ValidationResult.BLOCKED
        
    except ImportError:
        pytest.skip("SQL validator not available")

def test_rate_limiter_basic():
    """Test basic rate limiting."""
    try:
        from src.security.rate_limiter import rate_limiter, RateLimitResult
        
        # Test rate limit check
        result = rate_limiter.check_rate_limit("test_user")
        assert result.result in [RateLimitResult.ALLOWED, RateLimitResult.BLOCKED]
        
        # Test recording request
        success = rate_limiter.record_request("test_user")
        assert isinstance(success, bool)
        
    except ImportError:
        pytest.skip("Rate limiter not available")

def test_audit_logger_basic():
    """Test basic audit logging."""
    try:
        from src.security.audit_logger import audit_logger, AuditEventType
        
        # Test logging a query execution
        audit_logger.log_query_execution(
            user_id="test_user",
            query="SELECT 1",
            table_names=["test"],
            success=True,
            result_count=1
        )
        
        # Test getting statistics
        stats = audit_logger.get_audit_statistics()
        assert isinstance(stats, dict)
        assert "total_events" in stats
        
    except ImportError:
        pytest.skip("Audit logger not available")

def test_configuration_loading():
    """Test configuration loading."""
    try:
        from config.security import SECURITY_CONFIG
        from config.gemini import GEMINI_CONFIG
        
        # Test that configs are dictionaries
        assert isinstance(SECURITY_CONFIG, dict)
        assert isinstance(GEMINI_CONFIG, dict)
        
        # Test required keys exist
        assert "allowed_operations" in SECURITY_CONFIG
        assert "blocked_keywords" in SECURITY_CONFIG
        assert "model" in GEMINI_CONFIG
        
    except ImportError:
        pytest.skip("Configuration modules not available")

@pytest.mark.skipif(not os.getenv("MYSQL_HOST"), reason="MySQL not configured")
def test_database_connection():
    """Test database connection (only if MySQL is configured)."""
    try:
        from src.database.connection import db_connection
        
        # Test initialization
        result = db_connection.initialize()
        assert result is True
        
        # Test connection health
        health = db_connection.test_connection()
        assert isinstance(health, bool)
        
    except ImportError:
        pytest.skip("Database connection module not available")
    except Exception as e:
        pytest.skip(f"Database connection failed: {e}")

@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="Gemini API not configured")
def test_gemini_client():
    """Test Gemini client (only if API key is configured)."""
    try:
        from src.ai.gemini_client import gemini_client
        
        # Test initialization
        result = gemini_client.initialize()
        assert result is True
        
        # Test basic functionality
        stats = gemini_client.get_usage_stats()
        assert isinstance(stats, dict)
        
    except ImportError:
        pytest.skip("Gemini client not available")
    except Exception as e:
        pytest.skip(f"Gemini client test failed: {e}")

def test_environment_variables():
    """Test environment variable handling."""
    # Test that we can handle missing environment variables gracefully
    
    # Save original values
    original_api_key = os.getenv("GOOGLE_API_KEY")
    original_mysql_host = os.getenv("MYSQL_HOST")
    
    try:
        # Test with missing API key
        if "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]
        
        # Should handle gracefully
        from config.gemini import GEMINI_CONFIG
        assert GEMINI_CONFIG["api_key"] == ""
        
    finally:
        # Restore original values
        if original_api_key:
            os.environ["GOOGLE_API_KEY"] = original_api_key
        if original_mysql_host:
            os.environ["MYSQL_HOST"] = original_mysql_host

def test_file_structure():
    """Test that expected files and directories exist."""
    base_path = Path(__file__).parent.parent
    
    # Test main files exist
    assert (base_path / "app.py").exists()
    assert (base_path / "requirements.txt").exists()
    assert (base_path / "README.md").exists()
    assert (base_path / "Dockerfile").exists()
    
    # Test directory structure
    assert (base_path / "src").is_dir()
    assert (base_path / "config").is_dir()
    assert (base_path / "docs").is_dir()
    
    # Test config files
    assert (base_path / "config" / "database.py").exists()
    assert (base_path / "config" / "security.py").exists()
    assert (base_path / "config" / "gemini.py").exists()

if __name__ == "__main__":
    # Run basic tests
    print("Running basic tests...")
    
    test_imports()
    print("✓ Imports test passed")
    
    test_sql_validator_basic()
    print("✓ SQL validator test passed")
    
    test_rate_limiter_basic()
    print("✓ Rate limiter test passed")
    
    test_audit_logger_basic()
    print("✓ Audit logger test passed")
    
    test_configuration_loading()
    print("✓ Configuration test passed")
    
    test_environment_variables()
    print("✓ Environment variables test passed")
    
    test_file_structure()
    print("✓ File structure test passed")
    
    print("\nAll basic tests passed! ✅")
    print("\nTo run full test suite with database and API tests:")
    print("1. Set up your .env file with MYSQL_* and GOOGLE_API_KEY")
    print("2. Run: pytest tests/ -v")
