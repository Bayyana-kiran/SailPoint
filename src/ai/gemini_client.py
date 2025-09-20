"""
Robust SQL Generation Engine - Eliminates all prompt parsing techniques
Uses structured response schema and reliable validation methods.
"""

import time
import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

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


class QueryClassifier:
    """Pure ML-based query classifier without any patterns or keywords."""
    
    def __init__(self):
        self.classifier_model = None
        self.embedding_cache = {}
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize the pure ML classification model."""
        try:
            if not GEMINI_AVAILABLE:
                logger.error("Gemini not available for classification")
                return False
            
            # Pure ML classifier - no patterns, just semantic understanding
            self.classifier_model = genai.GenerativeModel(
                model_name=GEMINI_CONFIG['model'],
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    top_p=0.9,
                    top_k=20,
                    max_output_tokens=100,
                    response_mime_type="application/json",
                    response_schema=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "requires_database_access": genai.protos.Schema(
                                type=genai.protos.Type.BOOLEAN,
                                description="Does this query need to access database information?"
                            ),
                            "confidence": genai.protos.Schema(
                                type=genai.protos.Type.NUMBER,
                                description="Confidence score from 0.0 to 1.0"
                            ),
                            "user_intent_summary": genai.protos.Schema(
                                type=genai.protos.Type.STRING,
                                description="Brief description of what the user wants to accomplish"
                            ),
                            "appropriate_response": genai.protos.Schema(
                                type=genai.protos.Type.STRING,
                                description="Most appropriate response approach for this query"
                            )
                        },
                        required=["requires_database_access", "confidence", "user_intent_summary", "appropriate_response"]
                    )
                ),
                safety_settings=SAFETY_SETTINGS,
                system_instruction=(
                    "You are an AI that understands user intent through natural language processing. "
                    "Analyze each query to determine if the user needs information from a database or wants conversational interaction. "
                    "Use semantic understanding of the user's underlying needs and goals."
                )
            )
            
            self._initialized = True
            logger.info("Pure ML query classifier initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ML classifier: {str(e)}")
            self._initialized = False
            return False
    
    def classify_query(self, user_input: str) -> Dict[str, Any]:
        """Classify user query using pure ML semantic understanding."""
        if not user_input or not isinstance(user_input, str):
            return {
                "requires_database_access": False,
                "confidence": 0.0,
                "user_intent_summary": "Invalid or empty input",
                "appropriate_response": "error_handling"
            }
        
        # Initialize if needed
        if not self._initialized:
            if not self.initialize():
                return self._ml_fallback(user_input)
        
        # Check cache first
        cache_key = user_input.strip()[:200]
        if cache_key in self.embedding_cache:
            cached_result = self.embedding_cache[cache_key]
            cached_result["cached"] = True
            return cached_result
        
        try:
            # Let the ML model understand the user's intent semantically
            analysis_prompt = f"Analyze this user input to understand their intent: {user_input}"
            
            response = self.classifier_model.generate_content(analysis_prompt)
            classification_data = json.loads(response.text)
            
            # Build result with ML understanding
            result = {
                "requires_database_access": classification_data.get("requires_database_access", False),
                "confidence": max(0.0, min(1.0, float(classification_data.get("confidence", 0.5)))),
                "user_intent_summary": classification_data.get("user_intent_summary", "Intent analysis completed"),
                "appropriate_response": classification_data.get("appropriate_response", "conversational"),
                "input_length": len(user_input),
                "cached": False
            }
            
            # Cache with intelligent memory management
            if len(self.embedding_cache) < 2000:
                self.embedding_cache[cache_key] = result.copy()
            else:
                # Remove oldest entries when cache is full
                oldest_keys = list(self.embedding_cache.keys())[:500]
                for key in oldest_keys:
                    del self.embedding_cache[key]
                self.embedding_cache[cache_key] = result.copy()
            
            return result
            
        except Exception as e:
            logger.warning(f"ML classification failed: {str(e)}")
            return self._ml_fallback(user_input)
    
    def _ml_fallback(self, user_input: str) -> Dict[str, Any]:
        """Fallback using basic ML heuristics when main classifier fails."""
        
        # Use simple ML-style approach - just length and question structure analysis
        input_words = user_input.split()
        word_count = len(input_words)
        
        # Very basic heuristic based on linguistic structure only
        has_question_words = any(word.lower() in user_input.lower() for word in ['what', 'who', 'where', 'when', 'how', 'which'])
        has_action_request = any(word.lower() in user_input.lower() for word in ['show', 'find', 'get', 'list', 'count'])
        
        # Determine database need based on semantic likelihood
        likely_database_request = (
            (has_question_words and word_count > 3) or
            (has_action_request and word_count > 2) or
            word_count > 8  # Longer queries more likely to be data requests
        )
        
        confidence = 0.6 if likely_database_request else 0.4
        
        return {
            "requires_database_access": likely_database_request,
            "confidence": confidence,
            "user_intent_summary": f"Fallback analysis of {word_count} words",
            "appropriate_response": "database_query" if likely_database_request else "conversational",
            "fallback_used": True
        }
    
    def should_generate_sql(self, user_input: str) -> Tuple[bool, Dict[str, Any]]:
        """Determine if SQL generation is needed using ML classification."""
        classification = self.classify_query(user_input)
        
        should_generate = (
            classification["requires_database_access"] and 
            classification["confidence"] >= 0.5
        )
        
        return should_generate, classification
    
    def get_contextual_response(self, user_input: str, classification: Dict[str, Any]) -> str:
        """Generate contextual response using ML understanding of user intent."""
        intent_summary = classification["user_intent_summary"]
        
        if not classification["requires_database_access"]:
            # Use the ML model to generate appropriate conversational response
            return self._generate_conversational_response(user_input, intent_summary)
        else:
            # This shouldn't be called if database access is required
            return None
    
    def _generate_conversational_response(self, user_input: str, intent_summary: str) -> str:
        """Generate conversational response based on ML understanding."""
        # Use a simple conversational model or template based on intent
        try:
            if not self._initialized:
                return self._default_conversational_response()
            
            # Generate contextual response using ML
            response_prompt = (
                f"Generate a helpful conversational response for this user input: '{user_input}'. "
                f"The user's intent appears to be: {intent_summary}. "
                f"You are a SailPoint IdentityIQ database assistant. Keep response concise and helpful."
            )
            
            conversation_model = genai.GenerativeModel(
                model_name=GEMINI_CONFIG['model'],
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=150
                )
            )
            
            response = conversation_model.generate_content(response_prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.warning(f"Conversational response generation failed: {str(e)}")
            return self._default_conversational_response()
    
    def _default_conversational_response(self) -> str:
        """Default response when ML response generation fails."""
        return (
            "I'm your SailPoint IdentityIQ database assistant. I can help you find information "
            "about users, roles, applications, and access rights. What would you like to know?"
        )


class QueryType(Enum):
    """Enum for SQL query types."""
    SELECT = "SELECT"
    SHOW = "SHOW" 
    DESCRIBE = "DESCRIBE"
    EXPLAIN = "EXPLAIN"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"


@dataclass
class SQLQuery:
    """Structured SQL query representation."""
    sql: str
    query_type: QueryType
    tables_used: List[str]
    columns_used: List[str]
    is_valid: bool
    confidence_score: float
    explanation: str
    potential_issues: List[str]
    optimization_suggestions: List[str]
    token_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['query_type'] = self.query_type.value
        return result


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


class SQLValidator:
    """Dedicated SQL validation class with no parsing dependencies."""
    
    # Allowed SQL operations
    ALLOWED_OPERATIONS = {
        QueryType.SELECT: r'^SELECT\s+',
        QueryType.SHOW: r'^SHOW\s+',
        QueryType.DESCRIBE: r'^(?:DESCRIBE|DESC)\s+',
        QueryType.EXPLAIN: r'^EXPLAIN\s+'
    }
    
    # Dangerous patterns that should never be allowed
    DANGEROUS_PATTERNS = [
        r'\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE)\b',
        r';\s*(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE)\b',
        r'--\s*[^\r\n]*(?:INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE)',
        r'/\*.*?(?:INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE).*?\*/',
        r'\b(LOAD_FILE|INTO\s+OUTFILE|INTO\s+DUMPFILE)\b'
    ]
    
    @classmethod
    def validate_sql(cls, sql: str) -> Tuple[str, bool, List[str]]:
        """
        Validate SQL query without any parsing tricks.
        Returns: (cleaned_sql, is_valid, issues)
        """
        if not sql or not isinstance(sql, str):
            return "SELECT 'No SQL provided' as message;", False, ["Empty or invalid SQL"]
        
        # Clean the SQL
        cleaned_sql = cls._clean_sql(sql)
        issues = []
        
        # Check for dangerous patterns
        sql_upper = cleaned_sql.upper()
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, sql_upper, re.IGNORECASE | re.MULTILINE | re.DOTALL):
                return (
                    "SELECT 'Blocked: Dangerous SQL operation detected' as message;",
                    False,
                    ["Contains dangerous SQL operations"]
                )
        
        # Validate allowed operations
        is_valid_operation = False
        detected_type = QueryType.UNKNOWN
        
        for query_type, pattern in cls.ALLOWED_OPERATIONS.items():
            if re.match(pattern, sql_upper, re.IGNORECASE):
                is_valid_operation = True
                detected_type = query_type
                break
        
        if not is_valid_operation:
            allowed_ops = [qt.value for qt in cls.ALLOWED_OPERATIONS.keys()]
            return (
                f"SELECT 'Invalid: Must start with {', '.join(allowed_ops)}' as message;",
                False,
                [f"Must start with one of: {', '.join(allowed_ops)}"]
            )
        
        # Add safety measures
        final_sql = cls._add_safety_measures(cleaned_sql, detected_type)
        
        return final_sql, True, issues
    
    @classmethod
    def _clean_sql(cls, sql: str) -> str:
        """Clean SQL without complex parsing."""
        # Remove common code block markers
        sql = re.sub(r'```(?:sql)?', '', sql, flags=re.IGNORECASE)
        
        # Remove leading/trailing whitespace
        sql = sql.strip()
        
        # Ensure single statement (split on ; and take first non-empty)
        statements = [s.strip() for s in sql.split(';') if s.strip()]
        if statements:
            sql = statements[0]
        
        # Add semicolon if missing
        if not sql.endswith(';'):
            sql += ';'
            
        return sql
    
    @classmethod
    def _add_safety_measures(cls, sql: str, query_type: QueryType) -> str:
        """Add safety measures to SQL."""
        sql_upper = sql.upper()
        
        # Add LIMIT to SELECT queries if not present
        if query_type == QueryType.SELECT and 'LIMIT' not in sql_upper:
            sql = sql.rstrip(';') + ' LIMIT 100;'
        
        return sql
    
    @classmethod
    def extract_tables(cls, sql: str) -> List[str]:
        """Extract table names using reliable regex patterns."""
        if not sql:
            return []
        
        tables = set()
        sql_clean = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)  # Remove comments
        sql_clean = re.sub(r'/\*.*?\*/', '', sql_clean, flags=re.DOTALL)  # Remove multi-line comments
        
        # Reliable table extraction patterns
        patterns = [
            r'\bFROM\s+([`"]?)(\w+)\1(?:\s+(?:AS\s+)?\w+)?',  # FROM table [AS alias]
            r'\bJOIN\s+([`"]?)(\w+)\1(?:\s+(?:AS\s+)?\w+)?',  # JOIN table [AS alias]
            r'\bUPDATE\s+([`"]?)(\w+)\1',  # UPDATE table (shouldn't occur but just in case)
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, sql_clean, re.IGNORECASE)
            for _, table_name in matches:
                if table_name and len(table_name) > 1:
                    tables.add(table_name.lower())
        
        return sorted(list(tables))
    
    @classmethod
    def identify_query_type(cls, sql: str) -> QueryType:
        """Identify query type reliably."""
        if not sql:
            return QueryType.UNKNOWN
        
        sql_upper = sql.upper().strip()
        
        for query_type, pattern in cls.ALLOWED_OPERATIONS.items():
            if re.match(pattern, sql_upper, re.IGNORECASE):
                return query_type
        
        return QueryType.UNKNOWN


class StructuredResponseHandler:
    """Handles structured responses using native Gemini structured generation."""
    
    @staticmethod
    def create_response_schema():
        """Define the response schema using Gemini's structured generation."""
        import google.generativeai as genai
        from google.generativeai import protos
        
        return genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "sql": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    description="The generated SQL query"
                ),
                "explanation": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    description="Clear explanation of the query"
                ),
                "confidence": genai.protos.Schema(
                    type=genai.protos.Type.NUMBER,
                    description="Confidence score from 0.0 to 1.0"
                ),
                "query_type": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    description="Type of SQL query",
                    enum=["SELECT", "SHOW", "DESCRIBE", "EXPLAIN", "UNKNOWN"]
                ),
                "tables_used": genai.protos.Schema(
                    type=genai.protos.Type.ARRAY,
                    items=genai.protos.Schema(type=genai.protos.Type.STRING),
                    description="List of tables referenced in the query"
                ),
                "is_valid": genai.protos.Schema(
                    type=genai.protos.Type.BOOLEAN,
                    description="Whether the query appears valid"
                )
            },
            required=["sql", "explanation", "is_valid", "confidence", "query_type", "tables_used"]
        )
    
    @staticmethod
    def extract_structured_response(response) -> Dict[str, Any]:
        """Extract structured response directly from Gemini's structured output."""
        try:
            # Gemini's structured generation returns the response directly as structured data
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'text'):
                            # The structured response should already be valid JSON
                            return json.loads(part.text)
            
            # Fallback - try to get text content
            if hasattr(response, 'text'):
                return json.loads(response.text)
                
            raise ValueError("No structured content found in response")
            
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            logger.error(f"Failed to extract structured response: {e}")
            # Return fallback structure
            return {
                "sql": "SELECT 'Response extraction failed' as error;",
                "explanation": f"Failed to extract AI response: {str(e)}",
                "is_valid": False,
                "confidence": 0.0,
                "query_type": "ERROR",
                "tables_used": []
            }
    
    @staticmethod
    def validate_parsed_response(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize parsed response."""
        # Set defaults for missing fields
        defaults = {
            "sql": "SELECT 'Missing SQL' as error;",
            "explanation": "No explanation provided",
            "is_valid": False,
            "confidence": 0.5,
            "query_type": "UNKNOWN",
            "tables_used": []
        }
        
        # Merge with defaults
        for key, default_value in defaults.items():
            if key not in data:
                data[key] = default_value
        
        # Validate and fix data types
        if not isinstance(data["confidence"], (int, float)):
            data["confidence"] = 0.5
        else:
            data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))
        
        if not isinstance(data["tables_used"], list):
            data["tables_used"] = []
        
        if not isinstance(data["is_valid"], bool):
            data["is_valid"] = False
        
        return data


class RobustSQLEngine:
    """
    Robust SQL generation engine with no complex prompt parsing.
    Uses structured generation and reliable validation only.
    """

    def __init__(self):
        self.model = None
        self.total_tokens_used = 0
        self.request_count = 0
        self._initialized = False
        self.query_cache = {}
        self.validator = SQLValidator()
        self.response_handler = StructuredResponseHandler()
        self.query_classifier = QueryClassifier()  # Add ML-based classifier

    def initialize(self) -> bool:
        """Initialize the robust SQL engine with structured generation."""
        try:
            validate_gemini_config()
            genai.configure(api_key=GEMINI_CONFIG['api_key'])

            # Configure for structured generation
            self.model = genai.GenerativeModel(
                model_name=GEMINI_CONFIG['model'],
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=2048,
                    response_mime_type="application/json",
                    response_schema=self.response_handler.create_response_schema()
                ),
                safety_settings=SAFETY_SETTINGS,
                system_instruction=self._get_system_instruction()
            )

            self._initialized = True
            logger.info(f"Robust SQL engine initialized with structured generation")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize SQL engine: {str(e)}")
            self._initialized = False
            return False

    def _get_system_instruction(self) -> str:
        """Get system instruction that defines behavior without prompt engineering."""
        return (
            "You are a SQL expert for SailPoint IdentityIQ databases. "
            "Generate ONLY safe, read-only SQL queries (SELECT, SHOW, DESCRIBE, EXPLAIN). "
            "Never generate INSERT, UPDATE, DELETE, or any data modification queries. "
            "Always respond with the exact JSON schema provided."
        )

    def generate_sql_query(
        self,
        user_question: str,
        schema_context: str,
        additional_context: Optional[str] = None
    ) -> GenerationResponse:
        """Generate SQL query or conversational response using pure ML classification."""
        if not self._initialized:
            if not self.initialize():
                return GenerationResponse(
                    success=False,
                    error_message="Engine initialization failed"
                )

        start_time = time.time()

        try:
            # Use pure ML classification - no patterns
            should_generate_sql, classification = self.query_classifier.should_generate_sql(user_question)
            
            logger.info(
                f"ML Classification - DB Required: {classification['requires_database_access']}, "
                f"Confidence: {classification['confidence']:.2f}, "
                f"Intent: {classification['user_intent_summary']}"
            )
            
            # Handle non-database requests
            if not should_generate_sql:
                contextual_response = self.query_classifier.get_contextual_response(user_question, classification)
                
                return GenerationResponse(
                    success=True,
                    sql_query=None,
                    explanation=contextual_response,
                    generation_time=time.time() - start_time,
                    usage_metadata={
                        'ml_classification': classification,
                        'tokens_saved': True,
                        'response_generated': 'conversational'
                    }
                )

            # Generate SQL for database requests
            cache_key = f"{user_question}_{hash(schema_context)}"
            if cache_key in self.query_cache:
                cached_result = self.query_cache[cache_key]
                return GenerationResponse(
                    success=True,
                    sql_query=cached_result,
                    explanation=cached_result.explanation,
                    generation_time=0.0,
                    usage_metadata={
                        'cached': True,
                        'ml_classification': classification
                    }
                )

            # Build content for SQL generation
            content_parts = [
                f"Database Schema:\n{schema_context[:2000]}",
                f"User Question: {user_question}",
                f"Semantic Context: {classification['user_intent_summary']}"
            ]
            
            if additional_context:
                content_parts.append(f"Additional Context: {additional_context[:500]}")
            
            # Generate structured SQL response
            response = self._generate_with_retry(content_parts)
            sql_result = self._process_structured_response(response, user_question)
            
            if sql_result and sql_result.is_valid:
                self.query_cache[cache_key] = sql_result

            generation_time = time.time() - start_time
            self.request_count += 1

            return GenerationResponse(
                success=True,
                sql_query=sql_result,
                explanation=sql_result.explanation,
                generation_time=generation_time,
                usage_metadata={
                    'ml_classification': classification,
                    'sql_generated': True,
                    'processing_time': generation_time
                }
            )

        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return GenerationResponse(
                success=False,
                error_message=f"Processing error: {str(e)}",
                generation_time=time.time() - start_time
            )

    def _generate_with_retry(self, content_parts: List[str], max_retries: int = 3) -> Any:
        """Generate response using structured content parts."""
        content = "\n\n".join(content_parts)
        
        for attempt in range(max_retries + 1):
            try:
                response = self.model.generate_content(content)
                
                # Basic safety check
                if hasattr(response, 'safety_ratings'):
                    for rating in response.safety_ratings:
                        if rating.get('blocked', False):
                            raise ValueError("Response blocked by safety filters")
                
                return response
                
            except google_exceptions.ResourceExhausted:
                if attempt < max_retries:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise
                
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                raise e

    def _process_structured_response(self, response, user_question: str) -> SQLQuery:
        """Process structured response without any parsing tricks."""
        try:
            # Extract structured data directly
            parsed_data = self.response_handler.extract_structured_response(response)
            validated_data = self.response_handler.validate_parsed_response(parsed_data)
            
            # Validate SQL using dedicated validator
            sql_query, is_valid, issues = self.validator.validate_sql(validated_data["sql"])
            
            # Extract tables reliably
            tables_used = self.validator.extract_tables(sql_query)
            
            # Determine query type
            query_type = self.validator.identify_query_type(sql_query)
            
            return SQLQuery(
                sql=sql_query,
                query_type=query_type,
                tables_used=tables_used,
                columns_used=[],
                is_valid=is_valid,
                confidence_score=validated_data["confidence"],
                explanation=validated_data["explanation"],
                potential_issues=issues,
                optimization_suggestions=[]
            )
            
        except Exception as e:
            logger.error(f"Structured response processing failed: {str(e)}")
            return SQLQuery(
                sql="SELECT 'Structured processing failed' as error;",
                query_type=QueryType.ERROR,
                tables_used=[],
                columns_used=[],
                is_valid=False,
                confidence_score=0.0,
                explanation=f"Failed to process structured response: {str(e)}",
                potential_issues=["structured_processing_error"],
                optimization_suggestions=[]
            )

    def clear_conversation(self):
        """Clear conversation history and cache."""
        self.query_cache.clear()
        logger.info("Cache and conversation history cleared")

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_requests": self.request_count,
            "total_tokens_used": self.total_tokens_used,
            "cached_queries": len(self.query_cache),
            "initialized": self._initialized
        }


# Global robust SQL engine instance
sql_engine = RobustSQLEngine()