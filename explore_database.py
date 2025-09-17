#!/usr/bin/env python3
"""
Database Table Explorer
Script to display all tables and their structure from your MySQL database.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def display_database_tables():
    """Display all tables and their structure from the database."""
    print("🔍 MYSQL DATABASE TABLE EXPLORER")
    print("=" * 60)
    
    try:
        # Import database connection
        from src.database.connection import db_connection
        from src.database.metadata import metadata_analyzer
        
        print(f"\n📊 Database Connection Details:")
        print(f"  Host: {os.getenv('MYSQL_HOST', 'localhost')}")
        print(f"  Port: {os.getenv('MYSQL_PORT', '3306')}")
        print(f"  Database: {os.getenv('MYSQL_DATABASE', 'Not Set')}")
        print(f"  User: {os.getenv('MYSQL_USER', 'Not Set')}")
        print(f"  Password: {'SET' if os.getenv('MYSQL_PASSWORD') else 'NOT SET'}")
        
        # Test database connection
        print(f"\n🔗 Testing Database Connection...")
        if not db_connection.test_connection():
            print("❌ Database connection failed!")
            print("Please check your database credentials in the .env file.")
            return False
        
        print("✅ Database connection successful!")
        
        # Get database metadata
        print(f"\n📋 Analyzing Database Structure...")
        
        # Initialize metadata analyzer
        if not metadata_analyzer.initialize():
            print("❌ Failed to initialize metadata analyzer")
            return False
        
        # Get all tables
        schema_info = metadata_analyzer.get_full_schema_info()
        
        if not schema_info:
            print("❌ No schema information found. Database might be empty.")
            return False
        
        print("✅ Schema analysis complete!")
        
        # Display table information
        print(f"\n📊 DATABASE TABLES AND STRUCTURE:")
        print("=" * 60)
        
        # Parse and display tables
        tables = []
        current_table = None
        
        for line in schema_info.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Table:'):
                current_table = line.replace('Table:', '').strip()
                tables.append(current_table)
                print(f"\n🏷️  TABLE: {current_table}")
                print("-" * 40)
            elif line.startswith('Columns:'):
                print("   📂 COLUMNS:")
            elif line.startswith('- '):
                # Column information
                column_info = line[2:].strip()
                print(f"      • {column_info}")
            elif line.startswith('Relationships:'):
                print("   🔗 RELATIONSHIPS:")
            elif 'foreign key' in line.lower() or 'references' in line.lower():
                print(f"      → {line}")
        
        # Summary
        print(f"\n📈 DATABASE SUMMARY:")
        print("=" * 60)
        print(f"  Total Tables Found: {len(tables)}")
        print(f"  Table Names: {', '.join(tables) if tables else 'None'}")
        
        # Additional database statistics
        try:
            with db_connection.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get row counts for each table
                print(f"\n📊 TABLE ROW COUNTS:")
                print("-" * 40)
                
                for table in tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
                        count = cursor.fetchone()[0]
                        print(f"  {table}: {count:,} rows")
                    except Exception as e:
                        print(f"  {table}: Unable to count ({str(e)[:50]}...)")
                
        except Exception as e:
            print(f"⚠️  Could not get table statistics: {str(e)}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Database exploration failed: {str(e)}")
        print(f"\nFull error traceback:")
        traceback.print_exc()
        return False

def show_raw_table_list():
    """Show raw table list using direct SQL query."""
    print(f"\n🔧 RAW TABLE LIST (Direct SQL Query):")
    print("=" * 60)
    
    try:
        import pymysql
        
        # Connection parameters from environment
        config = {
            'host': os.getenv('MYSQL_HOST', 'localhost'),
            'port': int(os.getenv('MYSQL_PORT', 3306)),
            'database': os.getenv('MYSQL_DATABASE', ''),
            'user': os.getenv('MYSQL_USER', ''),
            'password': os.getenv('MYSQL_PASSWORD', ''),
            'charset': 'utf8mb4'
        }
        
        # Connect directly
        connection = pymysql.connect(**config)
        
        with connection.cursor() as cursor:
            # Show all tables
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            
            print(f"\n📋 Found {len(tables)} tables:")
            
            for i, (table_name,) in enumerate(tables, 1):
                print(f"\n{i}. 🏷️  TABLE: {table_name}")
                
                # Show table structure
                cursor.execute(f"DESCRIBE `{table_name}`")
                columns = cursor.fetchall()
                
                print("   📂 COLUMNS:")
                for col in columns:
                    field, type_, null, key, default, extra = col
                    key_info = f" [{key}]" if key else ""
                    null_info = "NULL" if null == "YES" else "NOT NULL"
                    default_info = f" DEFAULT: {default}" if default else ""
                    extra_info = f" {extra}" if extra else ""
                    
                    print(f"      • {field}: {type_} {null_info}{key_info}{default_info}{extra_info}")
                
                # Show row count
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
                    count = cursor.fetchone()[0]
                    print(f"   📊 ROWS: {count:,}")
                except:
                    print(f"   📊 ROWS: Unable to count")
        
        connection.close()
        return True
        
    except Exception as e:
        print(f"❌ Direct SQL query failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Database Table Explorer...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("Please create a .env file with your database credentials.")
        exit(1)
    
    # Try advanced analysis first
    print("\n🔍 Method 1: Using Advanced Database Analyzer")
    success = display_database_tables()
    
    if not success:
        print("\n🔧 Method 2: Using Direct SQL Queries")
        success = show_raw_table_list()
    
    if success:
        print(f"\n✅ Database exploration completed successfully!")
    else:
        print(f"\n❌ Database exploration failed.")
        print("\n📝 Troubleshooting:")
        print("1. Check your .env file has correct database credentials")
        print("2. Ensure MySQL server is running")
        print("3. Verify database name and user permissions")
        print("4. Check network connectivity to database server")
