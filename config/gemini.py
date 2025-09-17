"""
Gemini API Configuration
Configuration settings for Google Gemini API integration.
"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini API Configuration
GEMINI_CONFIG = {
    'api_key': os.getenv('GOOGLE_API_KEY', ''),
    'model': os.getenv('GEMINI_MODEL', 'gemini-2.5-flash'),
    'temperature': float(os.getenv('GEMINI_TEMPERATURE', 0.1)),
    'max_tokens': int(os.getenv('GEMINI_MAX_TOKENS', 8192)),
    'top_p': float(os.getenv('GEMINI_TOP_P', 0.8)),
    'top_k': int(os.getenv('GEMINI_TOP_K', 40)),
}

# Available Models
AVAILABLE_MODELS = [
    'gemini-2.5-flash',
    'gemini-2.5-pro',
    'gemini-1.5-pro',
    'gemini-1.5-flash'
]

# System Instructions for SQL Generation
SYSTEM_INSTRUCTIONS = """
You are an expert SQL database assistant. Your role is to help users query MySQL databases safely and efficiently.

CRITICAL RULES:
1. ONLY generate SELECT, SHOW, DESCRIBE, and EXPLAIN queries
2. NEVER generate INSERT, UPDATE, DELETE, DROP, or any data modification queries
3. Always use proper SQL syntax for MySQL
4. Include appropriate LIMIT clauses to prevent large result sets
5. Use table aliases for better readability
6. Consider database performance in your queries
7. Validate that referenced tables and columns exist
8. Provide clear explanations of what each query does
9. If unsure about table structure, ask for clarification
10. Handle ambiguous requests by asking specific questions

RESPONSE FORMAT:
- Provide the SQL query in a code block
- Explain what the query does
- Mention any assumptions made
- Suggest alternatives if applicable

SAFETY MEASURES:
- Always validate table and column names against the schema
- Use parameterized queries when possible
- Limit result sets appropriately
- Avoid expensive operations like full table scans when possible
"""

# Prompt Templates
PROMPT_TEMPLATES = {
    'sql_generation': """
Based on the database schema and user question, generate a safe MySQL SELECT query.

Database Schema:
{schema_context}

User Question: {user_question}

Additional Context: {additional_context}

Generate a safe, efficient MySQL query that answers the user's question.
""",
    
    'query_explanation': """
Explain this SQL query in simple terms:

Query: {query}

Schema Context: {schema_context}

Provide a clear explanation of:
1. What data the query retrieves
2. Which tables are involved
3. Any joins or conditions used
4. Expected result format
""",
    
    'error_analysis': """
Analyze this database error and provide a helpful solution:

Error: {error_message}
Query: {query}
Schema: {schema_context}

Provide:
1. What caused the error
2. How to fix it
3. A corrected query if applicable
""",
    
    'schema_summary': """
Summarize this database schema in user-friendly terms:

Schema: {schema_info}

Provide:
1. Overview of what the database contains
2. Key tables and their purpose
3. Important relationships
4. Common query patterns possible
"""
}

# Safety Configuration
SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

# Generation Configuration
GENERATION_CONFIG = {
    "temperature": GEMINI_CONFIG['temperature'],
    "top_p": GEMINI_CONFIG['top_p'],
    "top_k": GEMINI_CONFIG['top_k'],
    "max_output_tokens": GEMINI_CONFIG['max_tokens'],
    "response_mime_type": "text/plain",
}

def validate_gemini_config() -> bool:
    """Validate Gemini API configuration."""
    if not GEMINI_CONFIG['api_key']:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    
    if GEMINI_CONFIG['model'] not in AVAILABLE_MODELS:
        raise ValueError(f"Invalid model. Available models: {AVAILABLE_MODELS}")
    
    return True
