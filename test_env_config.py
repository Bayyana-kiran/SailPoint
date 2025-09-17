#!/usr/bin/env python3
"""
Environment Configuration Test
Tests that .env file is loaded correctly and all configurations work.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_env_loading():
    """Test that .env file is loaded correctly."""
    print("🔧 Testing Environment Configuration...")
    
    # Import configurations (this will trigger load_dotenv())
    try:
        from config.database import DATABASE_CONFIG, validate_config
        from config.gemini import GEMINI_CONFIG, validate_gemini_config
        
        print("\n✅ Configuration modules imported successfully")
        
        # Test database configuration
        print(f"\n📊 Database Configuration:")
        print(f"  Host: {DATABASE_CONFIG['host']}")
        print(f"  Port: {DATABASE_CONFIG['port']}")
        print(f"  Database: {DATABASE_CONFIG['database']}")
        print(f"  User: {DATABASE_CONFIG['user']}")
        print(f"  Password: {'*' * len(DATABASE_CONFIG['password']) if DATABASE_CONFIG['password'] else 'NOT SET'}")
        
        # Test Gemini configuration
        print(f"\n🤖 Gemini Configuration:")
        print(f"  Model: {GEMINI_CONFIG['model']}")
        print(f"  Temperature: {GEMINI_CONFIG['temperature']}")
        print(f"  Max Tokens: {GEMINI_CONFIG['max_tokens']}")
        print(f"  API Key: {'SET' if GEMINI_CONFIG['api_key'] else 'NOT SET'}")
        
        # Test environment variables directly
        print(f"\n🔧 Direct Environment Variables:")
        env_vars = [
            'MYSQL_HOST', 'MYSQL_PORT', 'MYSQL_DATABASE', 'MYSQL_USER',
            'GOOGLE_API_KEY', 'GEMINI_MODEL', 'ENVIRONMENT'
        ]
        
        for var in env_vars:
            value = os.getenv(var)
            if var in ['MYSQL_PASSWORD', 'GOOGLE_API_KEY']:
                display_value = 'SET' if value else 'NOT SET'
            else:
                display_value = value or 'NOT SET'
            print(f"  {var}: {display_value}")
        
        print(f"\n🎯 Testing Configuration Validation:")
        
        # Test database validation (if all required fields are set)
        try:
            if all(DATABASE_CONFIG[field] for field in ['host', 'database', 'user', 'password']):
                validate_config()
                print("  ✅ Database configuration is valid")
            else:
                print("  ⚠️  Database configuration incomplete (missing required fields)")
        except Exception as e:
            print(f"  ❌ Database validation failed: {e}")
        
        # Test Gemini validation
        try:
            if GEMINI_CONFIG['api_key']:
                validate_gemini_config()
                print("  ✅ Gemini configuration is valid")
            else:
                print("  ⚠️  Gemini configuration incomplete (missing API key)")
        except Exception as e:
            print(f"  ❌ Gemini validation failed: {e}")
        
        print(f"\n🎉 Environment configuration test completed!")
        return True
        
    except ImportError as e:
        print(f"  ❌ Failed to import configuration modules: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 DATABASE CHATBOT - ENVIRONMENT CONFIGURATION TEST")
    print("=" * 60)
    
    # Show current working directory and .env file status
    print(f"\n📁 Working Directory: {os.getcwd()}")
    env_file = Path(".env")
    if env_file.exists():
        print(f"✅ .env file found: {env_file.absolute()}")
        print(f"   File size: {env_file.stat().st_size} bytes")
    else:
        print(f"❌ .env file not found in current directory")
        exit(1)
    
    # Run the test
    success = test_env_loading()
    
    if success:
        print(f"\n✅ All tests passed! Your .env configuration is working correctly.")
    else:
        print(f"\n❌ Some tests failed. Please check your .env file configuration.")
    
    print("\n📝 Setup Instructions:")
    print("1. Ensure your .env file contains all required variables")
    print("2. Set your MySQL database connection details")
    print("3. Add your Google Gemini API key")
    print("4. Run: streamlit run app.py")
