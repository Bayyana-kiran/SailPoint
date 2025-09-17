#!/usr/bin/env python3
"""
Simple Database Table Explorer
Direct MySQL connection to display tables and their structure.
"""

import os
import pymysql
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def connect_to_database():
    """Create a direct MySQL connection."""
    connection_configs = [
        # Try with mysql_native_password first
        {
            'host': os.getenv('MYSQL_HOST', 'localhost'),
            'port': int(os.getenv('MYSQL_PORT', 3306)),
            'database': os.getenv('MYSQL_DATABASE', ''),
            'user': os.getenv('MYSQL_USER', ''),
            'password': os.getenv('MYSQL_PASSWORD', ''),
            'charset': 'utf8mb4',
            'connect_timeout': 10,
            'auth_plugin_map': {
                'mysql_native_password': '',
                'caching_sha2_password': ''
            }
        },
        # Try without auth_plugin_map
        {
            'host': os.getenv('MYSQL_HOST', 'localhost'),
            'port': int(os.getenv('MYSQL_PORT', 3306)),
            'database': os.getenv('MYSQL_DATABASE', ''),
            'user': os.getenv('MYSQL_USER', ''),
            'password': os.getenv('MYSQL_PASSWORD', ''),
            'charset': 'utf8mb4',
            'connect_timeout': 10
        },
        # Try with SSL disabled
        {
            'host': os.getenv('MYSQL_HOST', 'localhost'),
            'port': int(os.getenv('MYSQL_PORT', 3306)),
            'database': os.getenv('MYSQL_DATABASE', ''),
            'user': os.getenv('MYSQL_USER', ''),
            'password': os.getenv('MYSQL_PASSWORD', ''),
            'charset': 'utf8mb4',
            'connect_timeout': 10,
            'ssl_disabled': True
        }
    ]
    
    for i, config in enumerate(connection_configs, 1):
        try:
            print(f"  Attempt {i}: Trying connection with config {i}...")
            connection = pymysql.connect(**config)
            print(f"  ✅ Connection successful with config {i}")
            return connection
        except Exception as e:
            print(f"  ❌ Config {i} failed: {str(e)}")
            continue
    
    print(f"❌ All connection attempts failed")
    return None

def explore_database():
    """Explore database tables and structure."""
    print("🔍 SIMPLE DATABASE TABLE EXPLORER")
    print("=" * 60)
    
    # Display connection info
    print(f"\n📊 Connection Details:")
    print(f"  Host: {os.getenv('MYSQL_HOST', 'localhost')}")
    print(f"  Port: {os.getenv('MYSQL_PORT', '3306')}")
    print(f"  Database: {os.getenv('MYSQL_DATABASE', 'Not Set')}")
    print(f"  User: {os.getenv('MYSQL_USER', 'Not Set')}")
    print(f"  Password: {'SET' if os.getenv('MYSQL_PASSWORD') else 'NOT SET'}")
    
    # Connect to database
    print(f"\n🔗 Connecting to database...")
    connection = connect_to_database()
    
    if not connection:
        return False
    
    print("✅ Connected successfully!")
    
    try:
        with connection.cursor() as cursor:
            # Get all tables
            print(f"\n📋 Fetching table list...")
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            
            if not tables:
                print("❌ No tables found in the database.")
                return True
            
            print(f"✅ Found {len(tables)} tables")
            print("\n" + "=" * 60)
            print("📊 DATABASE TABLES AND STRUCTURE")
            print("=" * 60)
            
            for i, (table_name,) in enumerate(tables, 1):
                print(f"\n{i}. 🏷️  TABLE: {table_name}")
                print("-" * 50)
                
                # Get table structure
                cursor.execute(f"DESCRIBE `{table_name}`")
                columns = cursor.fetchall()
                
                print("   📂 COLUMNS:")
                for col in columns:
                    field, type_, null, key, default, extra = col
                    
                    # Format column info
                    key_info = ""
                    if key == "PRI":
                        key_info = " 🔑 PRIMARY KEY"
                    elif key == "UNI":
                        key_info = " 🔸 UNIQUE"
                    elif key == "MUL":
                        key_info = " 🔗 INDEX"
                    
                    null_info = "NULL" if null == "YES" else "NOT NULL"
                    default_info = f" (Default: {default})" if default else ""
                    extra_info = f" {extra}" if extra else ""
                    
                    print(f"      • {field}: {type_} - {null_info}{key_info}{default_info}{extra_info}")
                
                # Get row count
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
                    count = cursor.fetchone()[0]
                    print(f"   📊 TOTAL ROWS: {count:,}")
                except Exception as e:
                    print(f"   📊 TOTAL ROWS: Unable to count - {str(e)}")
                
                # Get sample data (first 3 rows)
                try:
                    cursor.execute(f"SELECT * FROM `{table_name}` LIMIT 3")
                    sample_rows = cursor.fetchall()
                    
                    if sample_rows:
                        print(f"   📝 SAMPLE DATA (First 3 rows):")
                        for j, row in enumerate(sample_rows, 1):
                            row_data = ", ".join([str(val)[:50] + "..." if len(str(val)) > 50 else str(val) for val in row])
                            print(f"      Row {j}: {row_data}")
                    else:
                        print(f"   📝 SAMPLE DATA: No data found")
                        
                except Exception as e:
                    print(f"   📝 SAMPLE DATA: Unable to fetch - {str(e)}")
            
            # Database summary
            print(f"\n📈 DATABASE SUMMARY:")
            print("=" * 60)
            print(f"  📊 Total Tables: {len(tables)}")
            print(f"  🏷️  Table Names: {', '.join([table[0] for table in tables])}")
            
            # Get database size info
            try:
                cursor.execute("""
                    SELECT 
                        ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) AS 'DB Size (MB)'
                    FROM information_schema.tables 
                    WHERE table_schema = %s
                """, (os.getenv('MYSQL_DATABASE'),))
                
                size_result = cursor.fetchone()
                if size_result and size_result[0]:
                    print(f"  💾 Database Size: {size_result[0]} MB")
            except Exception as e:
                print(f"  💾 Database Size: Unable to calculate - {str(e)}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error exploring database: {str(e)}")
        return False
    finally:
        connection.close()
        print(f"\n🔒 Database connection closed.")

if __name__ == "__main__":
    print("🚀 Starting Simple Database Explorer...")
    
    # Check environment variables (password can be empty)
    required_vars = ['MYSQL_HOST', 'MYSQL_DATABASE', 'MYSQL_USER']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file and ensure all database credentials are set.")
        exit(1)
    
    success = explore_database()
    
    if success:
        print(f"\n✅ Database exploration completed successfully!")
        print(f"\n📝 Next Steps:")
        print("1. Review the table structure above")
        print("2. Update your .env file if needed")
        print("3. Run: streamlit run app.py")
    else:
        print(f"\n❌ Database exploration failed.")
        print(f"\n📝 Troubleshooting Tips:")
        print("1. Verify MySQL server is running")
        print("2. Check database credentials in .env file")
        print("3. Ensure database exists and user has permissions")
        print("4. Test connection with MySQL client: mysql -h HOST -u USER -p DATABASE")
