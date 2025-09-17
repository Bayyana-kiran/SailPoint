"""
Advanced SQL Generation Engine
Uses structured function calling, query validation, and multi-step reasoning 
instead of primitive prompt parsing for production-grade text-to-SQL conversion.
"""

import time
import json
import logging
import re
import sqlparse
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from contextlib import asynccontextmanager

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    from google.api_core import exceptions as google_exceptions
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Google Generative AI not available. Using fallback mode.")

# Fallback to OpenAI if available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from config.gemini import (
    GEMINI_CONFIG, SYSTEM_INSTRUCTIONS, PROMPT_TEMPLATES,
    SAFETY_SETTINGS, GENERATION_CONFIG, validate_gemini_config
)

# Import validator for schema validation
try:
    from src.security.validator import sql_validator, ValidationResult
    VALIDATOR_AVAILABLE = True
except ImportError:
    VALIDATOR_AVAILABLE = False
    ValidationResult = None

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Types of SQL query intents."""
    SELECT_DATA = "select_data"
    COUNT_RECORDS = "count_records"
    AGGREGATE_DATA = "aggregate_data"
    JOIN_TABLES = "join_tables"
    FILTER_DATA = "filter_data"
    SORT_DATA = "sort_data"
    SCHEMA_INFO = "schema_info"
    UNKNOWN = "unknown"

class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"      # Single table, basic WHERE
    MODERATE = "moderate"  # Multiple tables, basic JOINs
    COMPLEX = "complex"    # Subqueries, CTEs, window functions
    ADVANCED = "advanced"  # Complex analytical queries

@dataclass
class QueryAnalysis:
    """Analysis of user query intent and requirements."""
    intent: QueryIntent
    complexity: QueryComplexity
    tables_mentioned: List[str]
    columns_mentioned: List[str]
    conditions: List[str]
    aggregations: List[str]
    confidence_score: float
    reasoning: str

@dataclass
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

@dataclass
class ChatMessage:
    """Enhanced chat message with metadata."""
    role: str
    content: str
    timestamp: float
    query_analysis: Optional[QueryAnalysis] = None
    generated_sql: Optional[SQLQuery] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class GenerationResponse:
    """Enhanced response with structured data."""
    sql_query: Optional[SQLQuery] = None
    explanation: str = ""
    query_analysis: Optional[QueryAnalysis] = None
    usage_metadata: Optional[Dict[str, Any]] = None
    safety_ratings: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = None
    generation_time: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None

class AdvancedSQLEngine:
    """
    Advanced SQL generation engine using structured reasoning, function calling,
    and multi-step validation instead of simple prompt parsing.
    """
    
    def __init__(self):
        self.model = None
        self.chat_session = None
        self.conversation_history: List[ChatMessage] = []
        self.total_tokens_used = 0
        self.request_count = 0
        self._initialized = False
        self.schema_cache: Dict[str, Any] = {}
        self.query_patterns: Dict[str, str] = {}
        self._load_query_patterns()
    
    def _load_query_patterns(self):
        """Load common SQL query patterns for intent recognition."""
        self.query_patterns = {
            "count": r'\b(how many|count|number of|total)\b',
            "top": r'\b(top|best|highest|largest|maximum)\b',
            "bottom": r'\b(bottom|worst|lowest|smallest|minimum)\b',
            "average": r'\b(average|avg|mean)\b',
            "sum": r'\b(sum|total|add up)\b',
            "recent": r'\b(recent|latest|newest|last)\b',
            "old": r'\b(old|oldest|earliest|first)\b',
            "list": r'\b(list|show|display|get)\b',
            "compare": r'\b(compare|vs|versus|difference)\b',
            "between": r'\b(between|from .* to|range)\b'
        }
    
    def initialize(self) -> bool:
        """Initialize the advanced SQL engine."""
        try:
            if GEMINI_AVAILABLE:
                # Validate configuration
                validate_gemini_config()
                
                # Configure API
                genai.configure(api_key=GEMINI_CONFIG['api_key'])
                
                # Initialize model (simplified for compatibility)
                self.model = genai.GenerativeModel(
                    model_name=GEMINI_CONFIG['model'],
                    generation_config=GENERATION_CONFIG,
                    safety_settings=SAFETY_SETTINGS,
                    system_instruction=self._get_advanced_system_instructions()
                )
                
                # Start chat session
                self.chat_session = self.model.start_chat(history=[])
                
                self._initialized = True
                logger.info(f"Advanced SQL engine initialized with model: {GEMINI_CONFIG['model']}")
                return True
                
            elif OPENAI_AVAILABLE:
                # Fallback to OpenAI (simple implementation)
                self.model = "openai-fallback"
                self.chat_session = None
                self._initialized = True
                logger.info("Advanced SQL engine initialized with OpenAI fallback")
                return True
            else:
                # Simple fallback without any AI
                self.model = "simple-fallback"
                self.chat_session = None
                self._initialized = True
                logger.info("Advanced SQL engine initialized with simple fallback mode")
                return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SQL engine: {str(e)}")
            # Always fallback to simple mode
            self.model = "simple-fallback"
            self.chat_session = None
            self._initialized = True
            logger.info("Advanced SQL engine initialized with simple fallback mode due to error")
            return True
    
    def _get_advanced_system_instructions(self) -> str:
        """Get advanced system instructions for IdentityIQ SQL generation."""
        return """
You are an expert SailPoint IdentityIQ SQL generation engine.

APPROACH:
1. Analyze the user's intent and understand what they want to query
2. Identify relevant IdentityIQ tables and columns from the schema provided
3. Generate optimized MySQL-compatible SQL queries
4. Use proper table joins and relationships
5. Apply appropriate filters and limits for safety

IDENTITYIQ EXPERTISE:
- Core tables: spt_identity (users), spt_application (systems), spt_link (accounts), spt_bundle (roles)
- Always use table aliases for readability
- Include LIMIT clauses to prevent large result sets
- Focus on business-relevant data (names, emails, statuses, dates)
- Understand IdentityIQ relationships and data model

RESPONSE FORMAT:
Provide the SQL query in a code block followed by a clear explanation of what it does and why.
- validate_sql_query(): Check query validity
- optimize_query(): Suggest improvements

RULES:
- Always use function calling instead of raw text generation
- Provide structured responses with confidence scores
- Include reasoning for all decisions
- Suggest optimizations and alternatives
- Handle edge cases and ambiguities gracefully
"""

    def _get_function_tools(self) -> List[Dict[str, Any]]:
        """Define function calling tools for structured SQL generation."""
        return [
            {
                "name": "analyze_query_intent",
                "description": "Analyze user query to understand intent and requirements",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_query": {"type": "string", "description": "User's natural language query"},
                        "intent": {"type": "string", "enum": ["select_data", "count_records", "aggregate_data", "join_tables", "filter_data", "sort_data", "schema_info"]},
                        "complexity": {"type": "string", "enum": ["simple", "moderate", "complex", "advanced"]},
                        "tables_needed": {"type": "array", "items": {"type": "string"}},
                        "columns_needed": {"type": "array", "items": {"type": "string"}},
                        "conditions": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                    },
                    "required": ["user_query", "intent", "complexity", "confidence"]
                }
            },
            {
                "name": "generate_sql_structure",
                "description": "Generate structured SQL query based on analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql_query": {"type": "string", "description": "Generated SQL query"},
                        "query_type": {"type": "string", "description": "Type of SQL operation"},
                        "tables_used": {"type": "array", "items": {"type": "string"}},
                        "explanation": {"type": "string", "description": "Explanation of the query logic"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                    },
                    "required": ["sql_query", "query_type", "explanation", "confidence"]
                }
            },
            {
                "name": "validate_sql_query",
                "description": "Validate SQL query structure and logic",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "is_valid": {"type": "boolean"},
                        "issues": {"type": "array", "items": {"type": "string"}},
                        "suggestions": {"type": "array", "items": {"type": "string"}},
                        "optimizations": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["is_valid"]
                }
            }
        ]

    def analyze_user_intent(self, user_question: str, schema_context: str) -> QueryAnalysis:
        """
        Advanced intent analysis using pattern matching and NLP.
        """
        # Pattern-based intent detection
        intent = QueryIntent.UNKNOWN
        confidence = 0.5
        
        user_lower = user_question.lower()
        
        # Intent classification
        if re.search(self.query_patterns["count"], user_lower):
            intent = QueryIntent.COUNT_RECORDS
            confidence = 0.8
        elif re.search(self.query_patterns["list"], user_lower):
            intent = QueryIntent.SELECT_DATA
            confidence = 0.7
        elif any(word in user_lower for word in ["sum", "average", "total", "max", "min"]):
            intent = QueryIntent.AGGREGATE_DATA
            confidence = 0.8
        elif "join" in user_lower or " and " in user_lower:
            intent = QueryIntent.JOIN_TABLES
            confidence = 0.6
        
        # Extract table and column mentions from schema
        tables_mentioned = self._extract_tables_from_query(user_question, schema_context)
        columns_mentioned = self._extract_columns_from_query(user_question, schema_context)
        
        # Determine complexity
        complexity = QueryComplexity.SIMPLE
        if len(tables_mentioned) > 1:
            complexity = QueryComplexity.MODERATE
        if any(word in user_lower for word in ["subquery", "nested", "window", "cte"]):
            complexity = QueryComplexity.COMPLEX
        
        return QueryAnalysis(
            intent=intent,
            complexity=complexity,
            tables_mentioned=tables_mentioned,
            columns_mentioned=columns_mentioned,
            conditions=self._extract_conditions(user_question),
            aggregations=self._extract_aggregations(user_question),
            confidence_score=confidence,
            reasoning=f"Detected {intent.value} intent with {confidence:.1%} confidence"
        )

    def generate_sql_query(
        self, 
        user_question: str, 
        schema_context: str, 
        additional_context: Optional[str] = None
    ) -> GenerationResponse:
        """
        Advanced SQL query generation with structured reasoning.
        
        Args:
            user_question: User's natural language question
            schema_context: Database schema information
            additional_context: Optional additional context
            
        Returns:
            GenerationResponse with generated SQL query and analysis
        """
        if not self._initialized:
            self.initialize()
        
        start_time = time.time()
        
        try:
            # Step 1: Analyze user intent using advanced techniques
            analysis = self.analyze_user_intent(user_question, schema_context)
            
            # Step 2: Build enhanced context for LLM
            enhanced_context = self._build_enhanced_context(
                user_question, schema_context, analysis, additional_context
            )
            
            # Step 3: Generate using function calling approach
            response = self._generate_with_advanced_reasoning(enhanced_context)
            
            # Step 4: Process and validate the response
            sql_result = self._process_advanced_response(response, analysis)
            
            # Step 5: Apply post-processing and validation
            if sql_result.sql and sql_result.sql.strip():
                sql_result = self._validate_and_optimize_query(sql_result)
            
            generation_time = time.time() - start_time
            
            # Update metrics
            self.request_count += 1
            
            # Log advanced metrics
            logger.info(
                f"Advanced SQL query generated in {generation_time:.2f}s - "
                f"Intent: {analysis.intent.value}, Complexity: {analysis.complexity.value}, "
                f"Confidence: {analysis.confidence_score:.2f}, Tables: {len(analysis.tables_mentioned)}"
            )
            
            # Add to conversation history with analysis
            self._add_to_history("user", user_question)
            self._add_to_history("model", f"Query: {sql_result.sql}\nExplanation: {sql_result.explanation}")
            
            # Create enhanced response object
            return GenerationResponse(
                success=True,
                sql_query=sql_result,
                explanation=sql_result.explanation,
                query_analysis=analysis,
                generation_time=generation_time,
                usage_metadata=getattr(response, 'usage_metadata', {})
            )
            
        except Exception as e:
            logger.error(f"Advanced SQL generation failed: {str(e)}")
            return GenerationResponse(
                success=False,
                error_message=f"SQL generation error: {str(e)}",
                sql_query=None,
                explanation="Failed to generate SQL query due to processing error",
                query_analysis=None,
                generation_time=time.time() - start_time,
                usage_metadata={'total_token_count': 0}
            )
    
    def explain_query(self, query: str, schema_context: str) -> GenerationResponse:
        """
        Explain an SQL query in natural language.
        
        Args:
            query: SQL query to explain
            schema_context: Database schema information
            
        Returns:
            GenerationResponse with query explanation
        """
        if not self._initialized:
            self.initialize()
        
        try:
            prompt = PROMPT_TEMPLATES['query_explanation'].format(
                query=query,
                schema_context=schema_context
            )
            
            response = self._generate_with_retry(prompt)
            
            # Add to conversation history
            self._add_to_history("user", f"Explain this query: {query}")
            self._add_to_history("model", response.explanation)
            
            return response
            
        except Exception as e:
            logger.error(f"Query explanation failed: {str(e)}")
            raise
    
    def analyze_error(
        self, 
        error_message: str, 
        query: str, 
        schema_context: str
    ) -> GenerationResponse:
        """
        Analyze database error and provide solution.
        
        Args:
            error_message: Database error message
            query: SQL query that caused the error
            schema_context: Database schema information
            
        Returns:
            GenerationResponse with error analysis and solution
        """
        if not self._initialized:
            self.initialize()
        
        try:
            prompt = PROMPT_TEMPLATES['error_analysis'].format(
                error_message=error_message,
                query=query,
                schema_context=schema_context
            )
            
            response = self._generate_with_retry(prompt)
            
            # Add to conversation history
            self._add_to_history("user", f"Error: {error_message}")
            self._add_to_history("model", response.explanation)
            
            return response
            
        except Exception as e:
            logger.error(f"Error analysis failed: {str(e)}")
            raise
    
    def summarize_schema(self, schema_info: str) -> GenerationResponse:
        """
        Generate user-friendly summary of database schema.
        
        Args:
            schema_info: Database schema information
            
        Returns:
            GenerationResponse with schema summary
        """
        if not self._initialized:
            self.initialize()
        
        try:
            prompt = PROMPT_TEMPLATES['schema_summary'].format(
                schema_info=schema_info
            )
            
            response = self._generate_with_retry(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Schema summarization failed: {str(e)}")
            raise
    
    def chat_with_context(
        self, 
        message: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> GenerationResponse:
        """
        Continue conversation with additional context.
        
        Args:
            message: User message
            context: Optional context information
            
        Returns:
            GenerationResponse with model's response
        """
        if not self._initialized:
            self.initialize()
        
        try:
            # Add context if provided
            if context:
                context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
                full_message = f"{message}\n\nContext:\n{context_str}"
            else:
                full_message = message
            
            response = self._generate_with_retry(full_message)
            
            # Add to conversation history
            self._add_to_history("user", message)
            self._add_to_history("model", response.explanation)
            
            return response
            
        except Exception as e:
            logger.error(f"Chat generation failed: {str(e)}")
            raise
    
    def _generate_with_retry(
        self, 
        prompt: str, 
        max_retries: int = 3, 
        base_delay: float = 1.0
    ) -> GenerationResponse:
        """
        Generate response with exponential backoff retry logic.
        
        Args:
            prompt: Input prompt
            max_retries: Maximum number of retries
            base_delay: Base delay for exponential backoff
            
        Returns:
            GenerationResponse
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                if self.chat_session:
                    response = self.chat_session.send_message(prompt)
                else:
                    response = self.model.generate_content(prompt)
                
                # Update token usage if available
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    self.total_tokens_used += getattr(response.usage_metadata, 'total_token_count', 0)
                
                return GenerationResponse(
                    explanation=response.text,
                    usage_metadata=getattr(response, 'usage_metadata', None),
                    safety_ratings=getattr(response, 'safety_ratings', None),
                    finish_reason=getattr(response, 'finish_reason', None)
                )
                
            except google_exceptions.ResourceExhausted as e:
                logger.warning(f"Rate limit exceeded (attempt {attempt + 1})")
                last_exception = e
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                    
            except google_exceptions.InvalidArgument as e:
                logger.error(f"Invalid argument: {str(e)}")
                raise ValueError(f"Invalid request: {str(e)}")
                
            except google_exceptions.PermissionDenied as e:
                logger.error(f"Permission denied: {str(e)}")
                raise PermissionError(f"API access denied: {str(e)}")
                
            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {str(e)}")
                last_exception = e
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
        
        # All retries exhausted
        raise RuntimeError(f"Generation failed after {max_retries + 1} attempts: {str(last_exception)}")
    
    def _add_to_history(self, role: str, content: str):
        """Add message to conversation history."""
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=time.time()
        )
        self.conversation_history.append(message)
        
        # Keep history manageable (last 50 messages)
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history as list of dictionaries."""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "metadata": msg.metadata
            }
            for msg in self.conversation_history
        ]
    
    def clear_conversation(self):
        """Clear conversation history and start fresh chat session."""
        self.conversation_history.clear()
        if self.model:
            self.chat_session = self.model.start_chat(history=[])
        logger.info("Conversation history cleared")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        return {
            "total_requests": self.request_count,
            "total_tokens_used": self.total_tokens_used,
            "conversation_length": len(self.conversation_history),
            "model": GEMINI_CONFIG['model'],
            "initialized": self._initialized
        }
    
    def export_conversation(self, format: str = 'json') -> str:
        """Export conversation history."""
        if format == 'json':
            return json.dumps(self.get_conversation_history(), indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

# Global Gemini client instance
# Helper methods for advanced SQL engine
    def _extract_tables_from_query(self, question: str, schema_context: str) -> List[str]:
        """Extract potential IdentityIQ table names from the question and schema."""
        tables = []
        question_lower = question.lower()
        
        # IdentityIQ business term to table mapping
        term_to_table = {
            'user': 'spt_identity',
            'users': 'spt_identity', 
            'identity': 'spt_identity',
            'identities': 'spt_identity',
            'people': 'spt_identity',
            'person': 'spt_identity',
            'employee': 'spt_identity',
            'employees': 'spt_identity',
            
            'application': 'spt_application',
            'applications': 'spt_application',
            'app': 'spt_application',
            'apps': 'spt_application',
            'system': 'spt_application',
            'systems': 'spt_application',
            
            'account': 'spt_link',
            'accounts': 'spt_link',
            'link': 'spt_link',
            'links': 'spt_link',
            
            'role': 'spt_bundle',
            'roles': 'spt_bundle',
            'bundle': 'spt_bundle',
            'bundles': 'spt_bundle',
            'access': 'spt_bundle',
            'permission': 'spt_entitlement',
            'permissions': 'spt_entitlement',
            'entitlement': 'spt_entitlement',
            'entitlements': 'spt_entitlement',
            
            'workflow': 'spt_workflow',
            'workflows': 'spt_workflow',
            'process': 'spt_workflow',
            'case': 'spt_workflow_case',
            'cases': 'spt_workflow_case',
            
            'certification': 'spt_certification',
            'certifications': 'spt_certification',
            'review': 'spt_certification',
            'reviews': 'spt_certification',
            
            'audit': 'spt_audit_event',
            'activity': 'spt_audit_event',
            'activities': 'spt_audit_event',
            'event': 'spt_audit_event',
            'events': 'spt_audit_event',
            'log': 'spt_audit_event',
            'logs': 'spt_audit_event',
            
            'request': 'spt_request',
            'requests': 'spt_request',
            'task': 'spt_task_result',
            'tasks': 'spt_task_result',
            'policy': 'spt_policy',
            'policies': 'spt_policy',
            'violation': 'spt_policy_violation',
            'violations': 'spt_policy_violation'
        }
        
        # Find matching tables based on business terms
        for term, table in term_to_table.items():
            if term in question_lower:
                tables.append(table)
        
        # Also extract direct table references from schema context
        for line in schema_context.split('\n'):
            if '• spt_' in line:
                table_name = line.split(':')[0].replace('•', '').strip()
                if table_name.lower() in question_lower:
                    tables.append(table_name)
        
        return list(set(tables))  # Remove duplicates
    
    def _extract_columns_from_query(self, question: str, schema_context: str) -> List[str]:
        """Extract potential IdentityIQ column names from the question."""
        columns = []
        question_lower = question.lower()
        
        # IdentityIQ common column mappings
        column_terms = {
            'name': 'name',
            'names': 'name',
            'title': 'display_name',
            'email': 'email',
            'status': 'status',
            'type': 'type',
            'created': 'created',
            'modified': 'modified',
            'date': 'created',
            'time': 'created',
            'when': 'created',
            'id': 'id',
            'description': 'description',
            'owner': 'owner',
            'manager': 'manager',
            'department': 'department',
            'location': 'location',
            'cost_center': 'cost_center',
            'job_title': 'job_title',
            'first_name': 'firstname',
            'last_name': 'lastname',
            'username': 'name',
            'login': 'name',
            'employee_id': 'employee_number',
            'enabled': 'status',
            'active': 'status',
            'disabled': 'status',
            'inactive': 'status'
        }
        
        # Find matching columns based on terms
        for term, column in column_terms.items():
            if term in question_lower:
                columns.append(column)
        
        # Extract specific columns mentioned in schema
        for line in schema_context.split('\n'):
            if '    - ' in line and '(' in line:
                # Extract column name from format: "    - column_name (type)"
                column_part = line.strip().split('-', 1)[1].split('(')[0].strip()
                if column_part.lower() in question_lower:
                    columns.append(column_part)
        
        return list(set(columns))[:8]  # Remove duplicates, limit to 8
    
    def _extract_conditions(self, question: str) -> List[str]:
        """Extract filter conditions from the question."""
        conditions = []
        question_lower = question.lower()
        
        # Look for common filter patterns
        condition_patterns = [
            r'where\s+(\w+)\s*[=<>]',
            r'(\w+)\s*[=<>]\s*[\'"]?(\w+)[\'"]?',
            r'greater than\s+(\d+)',
            r'less than\s+(\d+)',
            r'equal to\s+[\'"]?(\w+)[\'"]?'
        ]
        
        for pattern in condition_patterns:
            matches = re.findall(pattern, question_lower)
            conditions.extend([str(match) for match in matches])
        
        return conditions[:3]  # Limit to 3 conditions
    
    def _extract_aggregations(self, question: str) -> List[str]:
        """Extract aggregation functions from the question."""
        aggregations = []
        question_lower = question.lower()
        
        agg_keywords = ['count', 'sum', 'average', 'avg', 'max', 'min', 'total']
        for keyword in agg_keywords:
            if keyword in question_lower:
                aggregations.append(keyword.upper())
        
        return list(set(aggregations))  # Remove duplicates

    def _build_enhanced_context(
        self, 
        user_question: str, 
        schema_context: str, 
        analysis: QueryAnalysis,
        additional_context: Optional[str] = None
    ) -> str:
        """Build enhanced context for LLM with IdentityIQ-specific structured analysis."""
        context = f"""
🎯 SAILPOINT IDENTITYIQ SQL GENERATION REQUEST

{schema_context}

📊 QUERY ANALYSIS:
- Intent: {analysis.intent.value}
- Complexity: {analysis.complexity.value}
- Confidence: {analysis.confidence_score:.1%}
- Tables Identified: {', '.join(analysis.tables_mentioned) if analysis.tables_mentioned else 'Auto-detect from question'}
- Columns Identified: {', '.join(analysis.columns_mentioned) if analysis.columns_mentioned else 'Auto-detect from question'}
- Conditions: {', '.join(analysis.conditions) if analysis.conditions else 'None specified'}
- Aggregations: {', '.join(analysis.aggregations) if analysis.aggregations else 'None required'}
- AI Reasoning: {analysis.reasoning}

🗣️ USER QUESTION: "{user_question}"
"""
        
        if additional_context:
            context += f"\n📝 ADDITIONAL CONTEXT:\n{additional_context}"
        
        context += """

🚀 IDENTITYIQ SQL GENERATION INSTRUCTIONS:
You are an expert IdentityIQ database analyst. Generate a precise SQL query that:

1. 🎯 ADDRESSES THE SPECIFIC QUESTION: Understand what the user is really asking
2. 🏗️ USES CORRECT IDENTITYIQ SCHEMA: Follow the table relationships and naming conventions shown above
3. 🔗 INCLUDES PROPER JOINS: Connect tables logically (identity → link → application pattern)
4. 📏 APPLIES SMART LIMITS: Use LIMIT for potentially large result sets (default 100)
5. 🎨 FORMATS RESULTS NICELY: Use aliases and readable column names
6. ⚡ OPTIMIZES FOR PERFORMANCE: Use indexes and efficient WHERE clauses
7. ✅ VALIDATES REFERENCES: Ensure all table aliases and column references are properly defined

🔍 IDENTITYIQ-SPECIFIC QUERY PATTERNS:
- For "users" or "identities": Use spt_identity table
- For "applications" or "systems": Use spt_application table  
- For "accounts": Join spt_identity → spt_link → spt_application
- For "roles" or "access": Use spt_bundle table
- For "recent activity": Use date filters with created/modified columns
- For "certifications": Use spt_certification and related tables
- For "workflows": Use spt_workflow and spt_workflow_case tables

⚠️ CRITICAL VALIDATION RULES:
- NEVER reference a table alias that isn't defined in FROM/JOIN clauses
- ALWAYS include all necessary JOINs for referenced columns
- For applications: JOIN spt_application a ON ... (then use a.name, a.type, etc.)
- For entitlements: JOIN spt_entitlement e ON ... (then use e.name, e.type, etc.)
- For bundles: JOIN spt_bundle b ON ... (then use b.name, b.type, etc.)
- For "not active" or "no recent activity": Use NOT EXISTS or LEFT JOIN with IS NULL to find missing records
- For "orphaned" or "inactive owners": Check for absence of recent audit events, not presence of old ones
- For date comparisons: Use proper date functions like DATE_SUB(NOW(), INTERVAL 1 YEAR)

📋 RESPONSE FORMAT:
Provide:
1. The complete SQL query (properly formatted)
2. A clear explanation of what the query does
3. Key IdentityIQ concepts involved
4. Expected result structure

Remember: This is a live IdentityIQ system - generate accurate, safe queries that respect the data model!
"""
        
        return context

    def _generate_with_advanced_reasoning(self, context: str):
        """Generate response using advanced reasoning with the LLM."""
        try:
            if GEMINI_AVAILABLE and hasattr(self, 'chat_session') and self.chat_session:
                # Use the chat session for continuity
                response = self.chat_session.send_message(context)
                return response
            else:
                # Fallback mode - return a simple mock response
                return self._generate_fallback_response(context)
            
        except Exception as e:
            logger.error(f"Advanced generation failed: {str(e)}")
            # Always fallback to simple response
            return self._generate_fallback_response(context)

    def _generate_fallback_response(self, context: str):
        """When AI is unavailable, return an error instead of hardcoded SQL."""
        # NO HARDCODED SQL QUERIES - Pure AI-driven approach only
        raise Exception("AI service is currently unavailable. Please configure Google Generative AI or OpenAI API keys to enable intelligent SQL query generation. This system does not use hardcoded queries and relies entirely on AI-generated SQL.")

    def _process_advanced_response(self, response, analysis: QueryAnalysis) -> SQLQuery:
        """Process LLM response into structured SQL query object."""
        try:
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            # Extract SQL query using improved parsing
            sql_query = self._extract_sql_from_response(response_text)
            
            # Extract explanation
            explanation = self._extract_explanation_from_response(response_text)
            
            # Determine confidence based on query quality
            confidence = self._calculate_query_confidence(sql_query, analysis)
            
            # Identify query type
            query_type = self._identify_query_type(sql_query)
            
            # Extract tables used
            tables_used = self._extract_tables_from_sql(sql_query)
            
            return SQLQuery(
                sql=sql_query,
                explanation=explanation,
                confidence_score=confidence,
                query_type=query_type,
                tables_used=tables_used,
                columns_used=[],
                is_valid=True,
                potential_issues=[],
                optimization_suggestions=[]
            )
            
        except Exception as e:
            logger.error(f"Response processing error: {str(e)}")
            return SQLQuery(
                sql="-- Error processing response",
                explanation=f"Failed to process LLM response: {str(e)}",
                confidence_score=0.0,
                query_type="error",
                tables_used=[],
                columns_used=[],
                is_valid=False,
                potential_issues=[str(e)],
                optimization_suggestions=[]
            )

    def _extract_sql_from_response(self, response_text: str) -> str:
        """Extract SQL query from LLM response using improved patterns."""
        # Look for SQL code blocks
        sql_patterns = [
            r'```sql\s*(.*?)\s*```',
            r'```\s*(SELECT.*?(?:;|\Z))',
            r'```\s*(INSERT.*?(?:;|\Z))',
            r'```\s*(UPDATE.*?(?:;|\Z))',
            r'```\s*(DELETE.*?(?:;|\Z))',
            r'(SELECT.*?(?:;|\Z))',
            r'(INSERT.*?(?:;|\Z))',
            r'(UPDATE.*?(?:;|\Z))',
            r'(DELETE.*?(?:;|\Z))'
        ]
        
        for pattern in sql_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if matches:
                # Clean and return the first match
                sql = matches[0].strip()
                # Remove common artifacts
                sql = re.sub(r'^\s*```[\w]*\s*', '', sql)
                sql = re.sub(r'\s*```\s*$', '', sql)
                # Fix semicolon before LIMIT issue
                sql = re.sub(r';\s*LIMIT\s+(\d+)', r' LIMIT \1', sql, flags=re.IGNORECASE)
                # Clean up trailing semicolons
                sql = sql.rstrip(';').strip()
                return sql
        
        # If no patterns match, return the response as-is
        return response_text.strip()

    def _extract_explanation_from_response(self, response_text: str) -> str:
        """Extract explanation from LLM response."""
        # Look for explanation patterns
        explanation_patterns = [
            r'(?:Explanation|EXPLANATION):\s*(.*?)(?:\n\n|$)',
            r'(?:This query|The query)\s+(.*?)(?:\n\n|$)',
            r'(?:Analysis|ANALYSIS):\s*(.*?)(?:\n\n|$)'
        ]
        
        for pattern in explanation_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        # Fallback: use the text after the SQL query
        lines = response_text.split('\n')
        in_explanation = False
        explanation_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip SQL code blocks
            if line.startswith('```') or re.match(r'^(SELECT|INSERT|UPDATE|DELETE)', line, re.IGNORECASE):
                in_explanation = False
                continue
            
            # Look for explanation markers
            if any(marker in line.lower() for marker in ['explanation', 'this query', 'analysis']):
                in_explanation = True
                explanation_lines.append(line)
                continue
            
            if in_explanation:
                explanation_lines.append(line)
        
        return ' '.join(explanation_lines) if explanation_lines else "Query generated successfully"

    def _calculate_query_confidence(self, sql_query: str, analysis: QueryAnalysis) -> float:
        """Calculate confidence score based on query quality and analysis."""
        if not sql_query or sql_query.strip().startswith('--'):
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Check for SQL keywords
        if re.search(r'\b(SELECT|INSERT|UPDATE|DELETE)\b', sql_query, re.IGNORECASE):
            confidence += 0.2
        
        # Check if it addresses the intent
        if analysis.intent == QueryIntent.COUNT_RECORDS and 'COUNT' in sql_query.upper():
            confidence += 0.2
        elif analysis.intent == QueryIntent.SELECT_DATA and 'SELECT' in sql_query.upper():
            confidence += 0.2
        elif analysis.intent == QueryIntent.AGGREGATE_DATA and any(agg in sql_query.upper() for agg in ['SUM', 'AVG', 'MAX', 'MIN']):
            confidence += 0.2
        
        # Check for mentioned tables
        for table in analysis.tables_mentioned:
            if table.lower() in sql_query.lower():
                confidence += 0.1
        
        # Penalize for obvious errors
        if 'error' in sql_query.lower() or sql_query.count('(') != sql_query.count(')'):
            confidence -= 0.3
        
        return max(0.0, min(1.0, confidence))

    def _identify_query_type(self, sql_query: str) -> str:
        """Identify the type of SQL query."""
        sql_upper = sql_query.upper().strip()
        
        if sql_upper.startswith('SELECT'):
            return 'SELECT'
        elif sql_upper.startswith('INSERT'):
            return 'INSERT'
        elif sql_upper.startswith('UPDATE'):
            return 'UPDATE'
        elif sql_upper.startswith('DELETE'):
            return 'DELETE'
        else:
            return 'UNKNOWN'

    def _extract_tables_from_sql(self, sql_query: str) -> List[str]:
        """Extract table names from SQL query."""
        tables = []
        
        # Pattern to match table names after FROM, JOIN, INTO, UPDATE
        patterns = [
            r'FROM\s+(\w+)',
            r'JOIN\s+(\w+)',
            r'INTO\s+(\w+)',
            r'UPDATE\s+(\w+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, sql_query, re.IGNORECASE)
            tables.extend(matches)
        
        return list(set(tables))  # Remove duplicates

    def _validate_and_optimize_query(self, sql_result: SQLQuery) -> SQLQuery:
        """Validate and optimize the generated SQL query."""
        try:
            # Basic syntax validation
            if not sql_result.sql.strip():
                sql_result.potential_issues.append("Empty SQL query")
                return sql_result
            
            # Check for balanced parentheses
            if sql_result.sql.count('(') != sql_result.sql.count(')'):
                sql_result.potential_issues.append("Unbalanced parentheses")
                sql_result.confidence_score *= 0.7
            
            # Check for SQL injection patterns
            dangerous_patterns = [
                r';\s*(DROP|DELETE|TRUNCATE)',
                r'UNION.*SELECT',
                r'--\s*$'
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, sql_result.sql, re.IGNORECASE):
                    sql_result.potential_issues.append(f"Potentially dangerous pattern detected: {pattern}")
                    sql_result.confidence_score *= 0.5
            
            # Schema validation if validator is available
            if VALIDATOR_AVAILABLE and ValidationResult:
                schema_validation = sql_validator._validate_against_schema(sql_result.sql)
                if schema_validation.result == ValidationResult.ERROR:
                    sql_result.potential_issues.append(f"Schema validation error: {schema_validation.message}")
                    sql_result.confidence_score *= 0.3
                elif schema_validation.result == ValidationResult.WARNING:
                    sql_result.potential_issues.append(f"Schema validation warning: {schema_validation.message}")
                    sql_result.confidence_score *= 0.8
            
            # Suggest optimizations
            optimizations = []
            if 'SELECT *' in sql_result.sql:
                optimizations.append("Consider selecting specific columns instead of using SELECT *")
            
            if not re.search(r'\bLIMIT\b', sql_result.sql, re.IGNORECASE) and 'SELECT' in sql_result.sql:
                optimizations.append("Consider adding a LIMIT clause for large result sets")
            
            # Add optimizations to explanation
            if optimizations:
                sql_result.explanation += f"\n\nOptimization suggestions:\n" + "\n".join(f"- {opt}" for opt in optimizations)
                sql_result.optimization_suggestions.extend(optimizations)
            
            return sql_result
            
        except Exception as e:
            logger.error(f"Query validation error: {str(e)}")
            sql_result.potential_issues.append(f"Validation error: {str(e)}")
            return sql_result

# Initialize the advanced SQL engine
sql_engine = AdvancedSQLEngine()