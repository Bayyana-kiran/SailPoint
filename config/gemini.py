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
    'model': os.getenv('GEMINI_MODEL', 'gemini-1.5-flash'),  # Changed to more stable model
    'temperature': float(os.getenv('GEMINI_TEMPERATURE', 0.1)),
    'max_tokens': int(os.getenv('GEMINI_MAX_TOKENS', 2048)),  # Reduced for faster responses
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
You are an expert SQL assistant for SailPoint IdentityIQ databases. Generate ONLY safe SELECT queries.

CRITICAL REQUIREMENTS:
1. ONLY generate SELECT, SHOW, DESCRIBE, or EXPLAIN queries - NEVER any data modification queries
2. Always format SQL in ```sql code blocks
3. Include LIMIT clauses to prevent large results (max 100 rows)
4. Use proper MySQL syntax
5. Reference only tables that exist in the schema
6. Provide clear explanations

SCHEMA CONTEXT:
{schema_context}

USER QUESTION: {user_question}

ADDITIONAL CONTEXT: {additional_context}

INSTRUCTIONS:
- If the question is about policies, query the spt_policy table
- If about users/identities, query spt_identity table  
- If about applications, query spt_application table
- If about roles, query spt_bundle table where type='role'
- Always include an explanation of what the query does

Generate the SQL query in this exact format:
```sql
SELECT [columns] FROM [table] WHERE [conditions] LIMIT [number];
```

Then provide an explanation.
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

# Structured Generation Configuration for JSON output
STRUCTURED_GENERATION_CONFIG = {
    "temperature": GEMINI_CONFIG['temperature'],
    "top_p": GEMINI_CONFIG['top_p'],
    "top_k": GEMINI_CONFIG['top_k'],
    "max_output_tokens": GEMINI_CONFIG['max_tokens'],
    "response_mime_type": "application/json",
    "response_schema": {
        "type": "object",
        "properties": {
            "sql": {"type": "string", "description": "The generated SQL query"},
            "explanation": {"type": "string", "description": "Explanation of what the query does"},
            "confidence": {"type": "number", "description": "Confidence score between 0 and 1"},
            "query_type": {"type": "string", "description": "Type of query (SELECT, SHOW, etc.)"},
            "tables_used": {"type": "array", "items": {"type": "string"}, "description": "List of tables used in the query"},
            "is_valid": {"type": "boolean", "description": "Whether the SQL is valid"}
        },
        "required": ["sql", "explanation", "is_valid"]
    }
}

def validate_gemini_config() -> bool:
    """Validate Gemini API configuration."""
    if not GEMINI_CONFIG['api_key']:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    
    if GEMINI_CONFIG['model'] not in AVAILABLE_MODELS:
        raise ValueError(f"Invalid model. Available models: {AVAILABLE_MODELS}")
    
    return True
