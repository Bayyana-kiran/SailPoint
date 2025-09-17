import streamlit as st
import time
import json
import logging
import traceback
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from src.database.connection import db_connection
    from src.database.metadata import metadata_analyzer
    from src.security.validator import sql_validator, ValidationResult
    from src.ai.gemini_client import sql_engine
except ImportError as e:
    st.error(f"Module import error: {e}")
    st.error("Please ensure all dependencies are installed: `pip install -r requirements.txt`")
    st.stop()

st.set_page_config(
    page_title="Database Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    
    .chat-message.user {
        background-color: #f0f2f6;
        margin-left: 2rem;
    }
    
    .chat-message.assistant {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        margin-right: 2rem;
    }
    
    .chat-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: white;
    }
    
    .user-avatar {
        background-color: #ff6b6b;
    }
    
    .assistant-avatar {
        background-color: #4ecdc4;
    }
    
    .chat-content {
        flex-grow: 1;
    }
    
    .query-box {
        background-color: #1e1e1e;
        color: #f8f8f2;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        margin: 1rem 0;
    }
    
    .error-box {
        background-color: #ffe6e6;
        border-left: 4px solid #ff4444;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'database_connected' not in st.session_state:
        st.session_state.database_connected = False
    
    if 'schema_analyzed' not in st.session_state:
        st.session_state.schema_analyzed = False
    
    if 'gemini_initialized' not in st.session_state:
        st.session_state.gemini_initialized = False
    
    if 'schema_context' not in st.session_state:
        st.session_state.schema_context = ""
    
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

def check_environment():
    import os
    
    required_vars = ['GOOGLE_API_KEY', 'MYSQL_HOST', 'MYSQL_DATABASE', 'MYSQL_USER']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if os.getenv('MYSQL_PASSWORD') is None:
        missing_vars.append('MYSQL_PASSWORD')
    
    if missing_vars:
        st.error(f"Missing environment variables: {', '.join(missing_vars)}")
        st.info("Please set the required environment variables and restart the application.")
        return False
    
    return True

def initialize_connections():
    try:
        if not st.session_state.database_connected:
            with st.spinner("Connecting to database..."):
                db_connection.initialize()
                st.session_state.database_connected = True
                st.success("✅ Database connected successfully!")
        
        if not st.session_state.schema_analyzed:
            with st.spinner("Analyzing database schema..."):
                schema = metadata_analyzer.analyze_schema()
                st.session_state.schema_context = metadata_analyzer.generate_llm_context(schema)
                st.session_state.schema_analyzed = True
                st.success(f"✅ Schema analyzed! Found {len(schema.tables)} tables.")
        
        if not st.session_state.gemini_initialized:
            with st.spinner("Initializing AI assistant..."):
                sql_engine.initialize()
                st.session_state.gemini_initialized = True
                st.success("✅ AI assistant ready!")
        
        return True
        
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        logger.error(f"Initialization error: {str(e)}")
        return False

def render_chat_message(message: Dict[str, Any]):
    """Render a chat message with proper styling."""
    role = message['role']
    content = message['content']
    timestamp = message.get('timestamp', time.time())
    
    avatar_class = "user-avatar" if role == "user" else "assistant-avatar"
    message_class = "user" if role == "user" else "assistant"
    avatar_text = "U" if role == "user" else "AI"
    
    st.markdown(f"""
    <div class="chat-message {message_class}">
        <div class="chat-avatar {avatar_class}">
            {avatar_text}
        </div>
        <div class="chat-content">
            <strong>{'You' if role == 'user' else 'Assistant'}</strong>
            <br>
            {content}
            <br>
            <small style="color: #666;">
                {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}
            </small>
        </div>
    </div>
    """, unsafe_allow_html=True)

def execute_sql_query(query: str) -> Dict[str, Any]:
    """Execute SQL query with validation and error handling."""
    try:
        # Validate query
        validation_result = sql_validator.validate_query(query)
        
        if validation_result.result == ValidationResult.BLOCKED:
            return {
                'success': False,
                'error': f"Query blocked: {validation_result.message}",
                'type': 'security'
            }
        
        if validation_result.result == ValidationResult.ERROR:
            return {
                'success': False,
                'error': f"Validation error: {validation_result.message}",
                'type': 'validation'
            }
        
        # Use sanitized query if available
        final_query = validation_result.sanitized_query or query
        
        # Execute query
        start_time = time.time()
        results, columns = db_connection.execute_query(final_query)
        execution_time = time.time() - start_time
        
        # Add to query history
        st.session_state.query_history.append({
            'query': final_query,
            'timestamp': time.time(),
            'execution_time': execution_time,
            'result_count': len(results),
            'success': True
        })
        
        return {
            'success': True,
            'data': results,
            'columns': columns,
            'execution_time': execution_time,
            'row_count': len(results),
            'validation_warning': validation_result.message if validation_result.result == ValidationResult.WARNING else None
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Query execution failed: {error_msg}")
        
        # Add failed query to history
        st.session_state.query_history.append({
            'query': query,
            'timestamp': time.time(),
            'execution_time': 0,
            'result_count': 0,
            'success': False,
            'error': error_msg
        })
        
        return {
            'success': False,
            'error': error_msg,
            'type': 'execution'
        }

def process_user_message(user_input: str):
    """Process user message and generate response."""
    try:
        # Add user message to chat
        st.session_state.messages.append({
            'role': 'user',
            'content': user_input,
            'timestamp': time.time()
        })
        
        with st.spinner("Generating SQL query..."):
            # Generate SQL query using Gemini
            response = sql_engine.generate_sql_query(
                user_question=user_input,
                schema_context=st.session_state.schema_context
            )
            
            # Extract SQL query from the SQLQuery object
            if response.sql_query and hasattr(response.sql_query, 'sql'):
                sql_query = response.sql_query.sql.strip()
                query_confidence = getattr(response.sql_query, 'confidence_score', 0.5)
                potential_issues = getattr(response.sql_query, 'potential_issues', [])
            else:
                # Fallback to parsing from explanation
                sql_query = response.explanation.strip()
                query_confidence = 0.5
                potential_issues = []
                
                # Extract SQL query from response (remove markdown formatting)
                if "```sql" in sql_query.lower():
                    sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
                elif "```" in sql_query:
                    sql_query = sql_query.split("```")[1].strip()
        
        # Check for potential issues with the generated query
        warning_msg = None
        if query_confidence < 0.5 or potential_issues:
            warning_msg = f"⚠️ **Query Quality Warning:**\n"
            warning_msg += f"- Confidence Score: {query_confidence:.1%}\n"
            if potential_issues:
                warning_msg += "- Issues: " + "; ".join(potential_issues) + "\n"
            warning_msg += "\nThe generated query may have problems. Do you want to proceed anyway?"
            
            # For now, we'll proceed but show the warning
            # In a production app, you might want to ask for confirmation
        
        # Execute the query
        execution_result = execute_sql_query(sql_query)
        
        if execution_result['success']:
            # Format successful response
            response_content = f"""
**Generated SQL Query:**
```sql
{sql_query}
```

**Query Results:**
- Rows returned: {execution_result['row_count']}
- Execution time: {execution_result['execution_time']:.3f} seconds
"""
            
            if 'warning_msg' in locals():
                response_content += f"\n{warning_msg}"
            
            if execution_result.get('validation_warning'):
                response_content += f"\n⚠️ **Validation Warning:** {execution_result['validation_warning']}"
            
            # Add assistant response
            st.session_state.messages.append({
                'role': 'assistant',
                'content': response_content,
                'timestamp': time.time(),
                'query': sql_query,
                'results': execution_result['data'],
                'columns': execution_result['columns']
            })
            
        else:
            # Handle query execution error
            error_type = execution_result.get('type', 'unknown')
            error_msg = execution_result['error']
            
            if error_type in ['security', 'validation']:
                response_content = f"""
**Query Generation Failed:**

{error_msg}

Please rephrase your question or ensure you're asking for data that exists in the database.
"""
            else:
                # Try to get error analysis from Gemini
                try:
                    error_analysis = sql_engine.analyze_error(
                        error_message=error_msg,
                        query=sql_query,
                        schema_context=st.session_state.schema_context
                    )
                    response_content = f"""
**Query Execution Error:**

{error_msg}

**AI Analysis:**
{error_analysis.explanation}
"""
                except:
                    response_content = f"""
**Query Execution Error:**

{error_msg}

Please check your question and try again.
"""
            
            st.session_state.messages.append({
                'role': 'assistant',
                'content': response_content,
                'timestamp': time.time(),
                'error': True
            })
        
    except Exception as e:
        logger.error(f"Message processing failed: {str(e)}")
        st.session_state.messages.append({
            'role': 'assistant',
            'content': f"I apologize, but I encountered an error while processing your request: {str(e)}",
            'timestamp': time.time(),
            'error': True
        })

def render_sidebar():
    """Render sidebar with database info and controls."""
    st.sidebar.title("Database Chatbot")
    
    # Connection status
    st.sidebar.subheader("📊 Status")
    if st.session_state.database_connected:
        st.sidebar.success("Database: Connected")
    else:
        st.sidebar.error("Database: Disconnected")
    
    if st.session_state.gemini_initialized:
        st.sidebar.success("AI Assistant: Ready")
    else:
        st.sidebar.error("AI Assistant: Not Ready")
    
    # Database info
    if st.session_state.database_connected:
        st.sidebar.subheader("🗄️ Database Info")
        try:
            conn_info = db_connection.get_connection_info()
            st.sidebar.json(conn_info)
        except:
            st.sidebar.error("Could not retrieve connection info")
    
    # Query history
    st.sidebar.subheader("📈 Query Statistics")
    if st.session_state.query_history:
        total_queries = len(st.session_state.query_history)
        successful_queries = len([q for q in st.session_state.query_history if q['success']])
        avg_execution_time = sum([q['execution_time'] for q in st.session_state.query_history]) / total_queries
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Total Queries", total_queries)
            st.metric("Success Rate", f"{(successful_queries/total_queries)*100:.1f}%")
        with col2:
            st.metric("Avg Time", f"{avg_execution_time:.3f}s")
    else:
        st.sidebar.info("No queries executed yet")
    
    # Controls
    st.sidebar.subheader("🔧 Controls")
    
    if st.sidebar.button("🔄 Refresh Schema"):
        with st.spinner("Refreshing schema..."):
            try:
                schema = metadata_analyzer.analyze_schema(force_refresh=True)
                st.session_state.schema_context = metadata_analyzer.generate_llm_context(schema)
                st.sidebar.success("Schema refreshed!")
            except Exception as e:
                st.sidebar.error(f"Schema refresh failed: {str(e)}")
    
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        sql_engine.clear_conversation()
        st.sidebar.success("Chat cleared!")
        st.rerun()
    
    if st.sidebar.button("📊 Export Data"):
        if st.session_state.messages:
            chat_data = json.dumps(st.session_state.messages, indent=2, default=str)
            st.sidebar.download_button(
                label="Download Chat History",
                data=chat_data,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Check environment
    if not check_environment():
        return
    
    # Initialize connections
    if not initialize_connections():
        return
    
    # Render sidebar
    render_sidebar()
    
    # Main chat interface
    st.title("💬 Database Chatbot")
    st.markdown("Ask questions about your database in natural language!")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            render_chat_message(message)
            
            # Show query results if available
            if message['role'] == 'assistant' and 'results' in message:
                if message['results']:
                    st.subheader("Query Results:")
                    results = message['results']
                    
                    # Display results without PyArrow dependency
                    if len(results) > 0:
                        st.info(f"📊 **{len(results)} rows returned**")
                        
                        # Convert to DataFrame for easier handling
                        df = pd.DataFrame(results)
                        
                        # Display as HTML table to avoid PyArrow
                        if len(df) > 50:
                            st.write(f"**Showing first 50 rows of {len(df)} total results:**")
                            display_df = df.head(50)
                        else:
                            display_df = df
                        
                        # Create HTML table manually
                        html_table = "<div style='max-height: 400px; overflow-y: auto;'>"
                        html_table += "<table style='width: 100%; border-collapse: collapse; font-size: 12px;'>"
                        
                        # Header
                        html_table += "<thead style='background-color: #f0f2f6; position: sticky; top: 0;'><tr>"
                        for col in display_df.columns:
                            html_table += f"<th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>{col}</th>"
                        html_table += "</tr></thead>"
                        
                        # Rows
                        html_table += "<tbody>"
                        for idx, row in display_df.iterrows():
                            html_table += "<tr>"
                            for col in display_df.columns:
                                value = str(row[col]) if pd.notna(row[col]) else ""
                                html_table += f"<td style='border: 1px solid #ddd; padding: 8px;'>{value}</td>"
                            html_table += "</tr>"
                        html_table += "</tbody></table></div>"
                        
                        st.markdown(html_table, unsafe_allow_html=True)
                        
                        # Show summary for numeric columns
                        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                        if len(numeric_cols) > 0:
                            with st.expander("📈 View Summary Statistics"):
                                for col in numeric_cols:
                                    col_data = df[col].dropna()
                                    if len(col_data) > 0:
                                        st.write(f"**{col}:**")
                                        st.write(f"- Count: {len(col_data)}")
                                        st.write(f"- Mean: {col_data.mean():.2f}")
                                        st.write(f"- Min: {col_data.min()}")
                                        st.write(f"- Max: {col_data.max()}")
                                        st.write("---")
                    else:
                        st.info("Query executed successfully but returned no results.")
                else:
                    st.info("Query executed successfully but returned no results.")
    
    # Chat input
    user_input = st.chat_input("Ask a question about your database...")
    
    if user_input:
        process_user_message(user_input)
        st.rerun()
    
    # Example queries
    if not st.session_state.messages:
        st.subheader("💡 Example Questions")
        examples = [
            "Show me all tables in the database",
            "What are the top 10 customers by sales?",
            "How many orders were placed last month?",
            "Which products have low inventory?",
            "Show me employee information from the HR department"
        ]
        
        cols = st.columns(len(examples))
        for i, example in enumerate(examples):
            with cols[i % len(cols)]:
                if st.button(example, key=f"example_{i}"):
                    process_user_message(example)
                    st.rerun()

if __name__ == "__main__":
    main()
