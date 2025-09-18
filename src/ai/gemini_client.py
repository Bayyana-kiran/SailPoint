"""
Simplified and Optimized SQL Generation Engine
Uses direct text generation with optimized prompts and token management.
"""

import time
import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

try:
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from config.gemini import (
    GEMINI_CONFIG, SAFETY_SETTINGS, STRUCTURED_GENERATION_CONFIG, validate_gemini_config
)

logger = logging.getLogger(__name__)


@dataclass
class SQLQuery:
    """Structured SQL query representation."""
    sql: str
    query_type: str
    tables_used: List[str]
    columns_used: List[str]
    is_valid: bool
    confidence_score: float
    explanation: str
    potential_issues: List[str]
    optimization_suggestions: List[str]
    token_count: Optional[int] = None


@dataclass
class GenerationResponse:
    """Enhanced response with structured data."""
    sql_query: Optional[SQLQuery] = None
    explanation: str = ""
    usage_metadata: Optional[Dict[str, Any]] = None
    safety_ratings: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = None
    generation_time: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None


class OptimizedSQLEngine:
    """
    Optimized SQL generation engine using direct text generation with token management.
    """

    def __init__(self):
        self.model = None
        self.conversation_history = []
        self.total_tokens_used = 0
        self.request_count = 0
        self._initialized = False
        self.last_schema_context = ""
        self.query_cache = {}

    def initialize(self) -> bool:
        """Initialize the optimized SQL engine."""
        try:
            validate_gemini_config()

            genai.configure(api_key=GEMINI_CONFIG['api_key'])

            self.model = genai.GenerativeModel(
                model_name=GEMINI_CONFIG['model'],
                generation_config=STRUCTURED_GENERATION_CONFIG,
                safety_settings=SAFETY_SETTINGS
            )

            self._initialized = True
            logger.info(f"Optimized SQL engine initialized with model: {GEMINI_CONFIG['model']}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize SQL engine: {str(e)}")
            self._initialized = False
            return False

    def generate_sql_query(
        self,
        user_question: str,
        schema_context: str,
        additional_context: Optional[str] = None
    ) -> GenerationResponse:
        """
        Generate SQL query using optimized direct text generation.
        """
        if not self._initialized:
            self.initialize()

        start_time = time.time()

        try:
            # Check cache first
            cache_key = f"{user_question}_{hash(schema_context)}"
            if cache_key in self.query_cache:
                cached_result = self.query_cache[cache_key]
                return GenerationResponse(
                    success=True,
                    sql_query=cached_result,
                    explanation=cached_result.explanation,
                    generation_time=0.0,
                    usage_metadata={'cached': True}
                )

            # Prepare optimized prompt
            prompt = self._build_optimized_prompt(user_question, schema_context, additional_context)

            # Generate response
            response = self._generate_with_retry(prompt)

            # Process response
            sql_result = self._process_response(response, user_question)

            # Cache result
            if sql_result and sql_result.is_valid:
                self.query_cache[cache_key] = sql_result

            generation_time = time.time() - start_time
            self.request_count += 1

            # Update token usage
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                tokens_used = getattr(response.usage_metadata, 'total_token_count', 0)
                self.total_tokens_used += tokens_used

            logger.info(
                f"SQL query generated in {generation_time:.2f}s - "
                f"Confidence: {sql_result.confidence_score:.2f}, Tokens: {tokens_used}"
            )

            return GenerationResponse(
                success=True,
                sql_query=sql_result,
                explanation=sql_result.explanation,
                generation_time=generation_time,
                usage_metadata=getattr(response, 'usage_metadata', {})
            )

        except Exception as e:
            logger.error(f"SQL generation failed: {str(e)}")
            return GenerationResponse(
                success=False,
                error_message=f"SQL generation error: {str(e)}",
                generation_time=time.time() - start_time
            )

    def _build_optimized_prompt(self, question: str, schema_context: str, additional_context: Optional[str] = None) -> str:
        """Build optimized prompt for SQL generation."""
        # Extract key schema information
        schema_summary = self._extract_schema_summary(schema_context)

        prompt_parts = [
            "You are an expert SQL assistant for SailPoint IdentityIQ databases.",
            "",
            "CRITICAL RULES:",
            "1. ONLY generate SELECT, SHOW, DESCRIBE, or EXPLAIN queries",
            "2. NEVER generate INSERT, UPDATE, DELETE, DROP, or any data modification",
            "3. Always use proper MySQL syntax",
            "4. Include LIMIT 100 for safety",
            "5. Use table aliases for readability",
            "",
            "DATABASE SCHEMA:",
            schema_summary,
            "",
            f"USER QUESTION: {question}",
        ]

        if additional_context:
            prompt_parts.extend([
                "",
                "ADDITIONAL CONTEXT:",
                additional_context
            ])

        prompt_parts.extend([
            "",
            "OUTPUT FORMAT:",
            "Respond with a valid JSON object containing:",
            "- sql: The SQL query string",
            "- explanation: Explanation of what the query does",
            "- confidence: Confidence score (0.0 to 1.0)",
            "- query_type: Type of query (SELECT, SHOW, DESCRIBE, EXPLAIN)",
            "- tables_used: Array of table names used",
            "- is_valid: Boolean indicating if SQL is valid",
            "",
            "Example:",
            '{"sql": "SELECT * FROM users LIMIT 100;", "explanation": "Retrieves all user records", "confidence": 0.9, "query_type": "SELECT", "tables_used": ["users"], "is_valid": true}',
            "",
            "Do not include any text outside the JSON object."
        ])

        return "\n".join(prompt_parts)

    def _extract_schema_summary(self, schema_context: str) -> str:
        """Extract key schema information for the prompt."""
        lines = schema_context.split('\n')
        summary_lines = []

        # Extract table definitions
        in_tables_section = False
        for line in lines:
            if 'TABLE:' in line:
                in_tables_section = True
                summary_lines.append(line)
            elif in_tables_section and line.strip().startswith('-'):
                summary_lines.append(line)
            elif line.strip() == '' and in_tables_section:
                # Stop after first table section to keep prompt concise
                break

        # If no structured tables found, use first 20 lines
        if not summary_lines:
            summary_lines = lines[:20]

        return '\n'.join(summary_lines)

    def _generate_with_retry(self, prompt: str, max_retries: int = 3) -> Any:
        """Generate response with exponential backoff."""
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                response = self.model.generate_content(prompt)

                # Check for safety issues
                if hasattr(response, 'safety_ratings'):
                    for rating in response.safety_ratings:
                        if rating.get('blocked', False):
                            raise ValueError("Response blocked by safety filters")

                return response

            except google_exceptions.ResourceExhausted as e:
                logger.warning(f"Rate limit exceeded (attempt {attempt + 1})")
                last_exception = e
                if attempt < max_retries:
                    delay = 1 * (2 ** attempt)
                    time.sleep(delay)
                    continue

            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {str(e)}")
                last_exception = e
                if attempt < max_retries:
                    delay = 1 * (2 ** attempt)
                    time.sleep(delay)
                    continue

        raise RuntimeError(f"Generation failed after {max_retries + 1} attempts: {str(last_exception)}")

    def _process_response(self, response, user_question: str) -> SQLQuery:
        """Process Gemini JSON response into SQL query."""
        try:
            # Parse JSON response
            response_text = response.text if hasattr(response, 'text') else str(response)
            response_data = json.loads(response_text)

            # Extract fields
            sql_query = response_data.get('sql', '')
            explanation = response_data.get('explanation', 'Query generated successfully.')
            confidence = response_data.get('confidence', 0.5)
            query_type = response_data.get('query_type', 'UNKNOWN')
            tables_used = response_data.get('tables_used', [])
            is_valid = response_data.get('is_valid', True)

            # Validate and clean SQL
            sql_query, is_valid = self._validate_and_clean_sql(sql_query)

            # If confidence not provided, calculate it
            if 'confidence' not in response_data:
                confidence = self._calculate_confidence(sql_query, user_question, explanation)

            # If query_type not provided, identify it
            if 'query_type' not in response_data:
                query_type = self._identify_query_type(sql_query)

            # If tables_used not provided, extract from SQL
            if not tables_used:
                tables_used = self._extract_tables_from_sql(sql_query)

            return SQLQuery(
                sql=sql_query,
                query_type=query_type,
                tables_used=tables_used,
                columns_used=[],
                is_valid=is_valid,
                confidence_score=confidence,
                explanation=explanation,
                potential_issues=[],
                optimization_suggestions=[]
            )

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)} - Response: {response_text}")
            return SQLQuery(
                sql="SELECT 'Error: Invalid JSON response' as message",
                query_type="error",
                tables_used=[],
                columns_used=[],
                is_valid=False,
                confidence_score=0.0,
                explanation=f"Failed to parse JSON response: {str(e)}",
                potential_issues=[str(e)],
                optimization_suggestions=[]
            )
        except Exception as e:
            logger.error(f"Response processing error: {str(e)}")
            return SQLQuery(
                sql="SELECT 'Error processing response' as message",
                query_type="error",
                tables_used=[],
                columns_used=[],
                is_valid=False,
                confidence_score=0.0,
                explanation=f"Failed to process response: {str(e)}",
                potential_issues=[str(e)],
                optimization_suggestions=[]
            )

    def _extract_sql_from_response(self, response_text: str) -> str:
        """Extract SQL query from response."""
        # Look for SQL code blocks
        sql_patterns = [
            r'```sql\s*(.*?)\s*```',
            r'```\s*(SELECT.*?;?)\s*```',
            r'(SELECT\s+.*?;)',
            r'(SELECT\s+.*?\n)',
        ]

        for pattern in sql_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                sql = match.strip()
                if sql and self._is_valid_sql_start(sql):
                    return sql

        # Fallback: look for any line starting with SELECT
        for line in response_text.split('\n'):
            line = line.strip()
            if line.upper().startswith('SELECT'):
                return line

        return ""

    def _is_valid_sql_start(self, sql: str) -> bool:
        """Check if SQL starts with allowed operation."""
        allowed_starts = ('SELECT', 'SHOW', 'DESCRIBE', 'EXPLAIN')
        return any(sql.upper().strip().startswith(op) for op in allowed_starts)

    def _validate_and_clean_sql(self, sql: str) -> Tuple[str, bool]:
        """Validate and clean SQL query."""
        if not sql:
            return "SELECT 'No SQL query generated' as message", False

        # Remove code block markers
        sql = re.sub(r'```sql', '', sql, flags=re.IGNORECASE).strip()
        sql = re.sub(r'```', '', sql).strip()

        # Ensure it ends with semicolon
        if not sql.endswith(';'):
            sql += ';'

        # Add LIMIT if not present and it's a SELECT
        if sql.upper().startswith('SELECT') and 'LIMIT' not in sql.upper():
            sql = sql.rstrip(';') + ' LIMIT 100;'

        return sql, True

    def _calculate_confidence(self, sql: str, question: str, response: str) -> float:
        """Calculate confidence score."""
        confidence = 0.5

        # Check for SQL keywords
        if 'SELECT' in sql.upper():
            confidence += 0.2

        # Check if response mentions uncertainty
        uncertainty_words = ['unsure', 'uncertain', 'maybe', 'perhaps', 'not sure']
        if any(word in response.lower() for word in uncertainty_words):
            confidence -= 0.3

        # Check for explanation
        if len(response.split()) > 20:
            confidence += 0.1

        return max(0.0, min(1.0, confidence))

    def _extract_explanation(self, response_text: str) -> str:
        """Extract explanation from response."""
        # Remove SQL code blocks
        text = re.sub(r'```.*?```', '', response_text, flags=re.DOTALL)

        # Get the remaining text as explanation
        explanation = text.strip()

        if not explanation:
            return "Query generated successfully."

        return explanation

    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        """Extract table names from SQL."""
        tables = []

        # Simple pattern matching
        patterns = [
            r'FROM\s+(\w+)',
            r'JOIN\s+(\w+)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, sql, re.IGNORECASE)
            tables.extend(matches)

        return list(set(tables))

    def _identify_query_type(self, sql: str) -> str:
        """Identify query type."""
        sql_upper = sql.upper().strip()

        if sql_upper.startswith('SELECT'):
            return 'SELECT'
        elif sql_upper.startswith('SHOW'):
            return 'SHOW'
        elif sql_upper.startswith('DESCRIBE'):
            return 'DESCRIBE'
        elif sql_upper.startswith('EXPLAIN'):
            return 'EXPLAIN'
        else:
            return 'UNKNOWN'

    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_requests": self.request_count,
            "total_tokens_used": self.total_tokens_used,
            "cached_queries": len(self.query_cache),
            "initialized": self._initialized
        }


# Global optimized SQL engine instance
sql_engine = OptimizedSQLEngine()
